/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "global_kvcache_mgr.h"

#include <nlohmann/json.hpp>

#include "common/hash_util.h"

namespace {
inline size_t round_down(size_t n, size_t multiple) {
  return (n / multiple) * multiple;
}

std::string ETCD_CACHE_PREFIX = "XLLM:CACHE:";
}  // namespace

namespace xllm_service {

GlobalKVCacheMgr::GlobalKVCacheMgr(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    const bool is_master_service)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client) {
  if (!is_master_service_) {
    auto handle_kvcache = std::bind(&GlobalKVCacheMgr::update_kvcache,
                                    this,
                                    std::placeholders::_1,
                                    std::placeholders::_2);
    etcd_client_->add_watch(ETCD_CACHE_PREFIX, handle_kvcache);
  }

  {
    std::unique_lock<std::shared_mutex> lock(kvcache_mutex_);
    etcd_client_->get_prefix(ETCD_CACHE_PREFIX, &kvcache_infos_);
    DLOG(INFO) << "Load etcd cache infos:" << kvcache_infos_.size();
  }
}

GlobalKVCacheMgr::~GlobalKVCacheMgr() {
  exited_ = true;
  etcd_client_->remove_watch(ETCD_CACHE_PREFIX);
}

void set_score(const std::unordered_set<std::string>& instance_names,
               const uint32_t& match_length,
               std::unordered_map<std::string, uint32_t>* scores,
               std::unordered_set<std::string>* instances) {
  for (const auto& name : instance_names) {
    if (scores->count(name) == 0) {
      scores->insert_or_assign(name, match_length);
    } else {
      (*scores)[name] = match_length;
    }
    instances->insert(name);
  }
}

void GlobalKVCacheMgr::match(const Slice<int32_t>& token_ids,
                             OverlapScores* overlap_scores) {
  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), options_.block_size());
  if (n_tokens == 0) {
    return;
  }

  overlap_scores->max_block_num = n_tokens / options_.block_size();

  std::shared_lock lock(kvcache_mutex_);
  XXH3Key token_hash_key;
  for (size_t i = 0; i < n_tokens; i += options_.block_size()) {
    if (i == 0) {
      xxh3_128bits_hash(nullptr,
                        token_ids.slice(i, i + options_.block_size()),
                        token_hash_key.data);
    } else {
      xxh3_128bits_hash(token_hash_key.data,
                        token_ids.slice(i, i + options_.block_size()),
                        token_hash_key.data);
    }

    auto iter = kvcache_infos_.find(token_hash_key);
    if (iter != kvcache_infos_.end() && !iter->second.empty()) {
      if (!iter->second.hbm_instance_set.empty()) {
        set_score(iter->second.hbm_instance_set,
                  i / options_.block_size() + 1,
                  &(overlap_scores->hbm_instance_score),
                  &(overlap_scores->instances));
        overlap_scores->max_matched_instance_name =
            *iter->second.hbm_instance_set.begin();
        overlap_scores->max_matched_block_num = i / options_.block_size() + 1;
      }

      if (!iter->second.dram_instance_set.empty()) {
        set_score(iter->second.dram_instance_set,
                  i / options_.block_size() + 1,
                  &(overlap_scores->dram_instance_score),
                  &(overlap_scores->instances));
        overlap_scores->max_matched_instance_name =
            *iter->second.hbm_instance_set.begin();
        overlap_scores->max_matched_block_num = i / options_.block_size() + 1;
      }

      if (!iter->second.ssd_instance_set.empty()) {
        set_score(iter->second.ssd_instance_set,
                  i / options_.block_size() + 1,
                  &(overlap_scores->ssd_instance_score),
                  &(overlap_scores->instances));
        overlap_scores->max_matched_instance_name =
            *iter->second.hbm_instance_set.begin();
        overlap_scores->max_matched_block_num = i / options_.block_size() + 1;
      }
    } else {
      break;
    }
  }
}

void GlobalKVCacheMgr::update_kvcache(const etcd::Response& response,
                                      const uint64_t prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    XXH3KeyCacheMap put_map;
    std::vector<XXH3Key> delete_list;

    for (const auto& event : response.events()) {
      auto key = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        CacheLocations cachelocations;
        auto json_str = event.kv().as_string();
        if (!cachelocations.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert_or_assign(XXH3Key{key.c_str()},
                                 std::move(cachelocations));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.emplace_back(XXH3Key{key.c_str()});
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(kvcache_mutex_);
      for (auto& iter : put_map) {
        kvcache_infos_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        kvcache_infos_.erase(iter);
      }
    }
  });
}

void GlobalKVCacheMgr::record_updated_kvcaches(
    const std::string& instance_name,
    const proto::KvCacheEvent& kvcache_event) {
  std::lock_guard<std::mutex> update_lock(update_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(kvcache_mutex_);
  for (int i = 0; i < kvcache_event.stored_cache_size(); i++) {
    XXH3Key key(kvcache_event.stored_cache(i).c_str());
    if (updated_kvcaches_.count(key) == 0) {
      if (kvcache_infos_.count(key) == 0) {
        updated_kvcaches_.insert_or_assign(key, CacheLocations());
      } else {
        updated_kvcaches_.insert_or_assign(key, kvcache_infos_[key]);
      }
    }
    updated_kvcaches_.at(key).hbm_instance_set.insert(instance_name);
  }

  for (int i = 0; i < kvcache_event.offload_cache_size(); i++) {
    XXH3Key key(kvcache_event.offload_cache(i).c_str());
    if (updated_kvcaches_.count(key) == 0) {
      if (kvcache_infos_.count(key) == 0) {
        continue;
      } else {
        updated_kvcaches_.insert_or_assign(key, kvcache_infos_[key]);
      }
    }
    if (updated_kvcaches_.at(key).hbm_instance_set.count(instance_name) != 0) {
      updated_kvcaches_.at(key).hbm_instance_set.erase(instance_name);
      updated_kvcaches_.at(key).dram_instance_set.insert(instance_name);
    } else {
      updated_kvcaches_.at(key).dram_instance_set.erase(instance_name);
      updated_kvcaches_.at(key).ssd_instance_set.insert(instance_name);
    }
  }

  for (int i = 0; i < kvcache_event.removed_cache_size(); i++) {
    XXH3Key key(kvcache_event.removed_cache(i).c_str());
    if (updated_kvcaches_.count(key) == 0) {
      if (kvcache_infos_.count(key) == 0) {
        continue;
      } else {
        updated_kvcaches_.insert_or_assign(key, kvcache_infos_[key]);
      }
    }
    updated_kvcaches_.at(key).hbm_instance_set.erase(instance_name);
    updated_kvcaches_.at(key).dram_instance_set.erase(instance_name);
    updated_kvcaches_.at(key).ssd_instance_set.erase(instance_name);
  }
}

bool GlobalKVCacheMgr::upload_kvcache() {
  std::lock_guard<std::mutex> update_lock(update_mutex_);
  if (updated_kvcaches_.empty()) {
    return true;
  }
  bool rt = etcd_client_->set(ETCD_CACHE_PREFIX, updated_kvcaches_);
  {
    std::unique_lock<std::shared_mutex> metric_lock(kvcache_mutex_);
    for (auto& iter : updated_kvcaches_) {
      if (iter.second.empty()) {
        kvcache_infos_.erase(iter.first);
      } else {
        kvcache_infos_.insert_or_assign(iter.first, std::move(iter.second));
      }
    }
  }
  if (rt) {
    updated_kvcaches_.clear();
  }
  return rt;
}

void GlobalKVCacheMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_CACHE_PREFIX);
}

}  // namespace xllm_service
