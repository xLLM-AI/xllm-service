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

#pragma once

#include <etcd/KeepAlive.hpp>
#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>
#include <etcd/v3/Transaction.hpp>
#include <string>
#include <unordered_set>

#include "common/hash_util.h"
#include "common/types.h"

namespace xllm_service {

using Callback = std::function<void(const etcd::Response&, const uint64_t&)>;

std::string get_event_key(const etcd::Event& event);

std::string get_event_value(const etcd::Event& event);

std::string get_event_key_suffix(const etcd::Event& event, uint64_t prefix_len);

class EtcdClient {
 public:
  EtcdClient(const std::string& etcd_addr,
             const std::string& etcd_namespace = "");
  EtcdClient(const std::string& etcd_addr,
             const std::string& username,
             const std::string& password,
             const std::string& etcd_namespace = "");
  ~EtcdClient();

  template <typename T>
  bool set(const std::string& key, const T& value) {
    auto response =
        client_.put(namespaced_key(key), value.serialize_to_json().dump());
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd set " << key
                 << " failed: " << response.error_message();
      return false;
    }

    return true;
  }

  template <typename T>
  bool set(const std::string& key_prefix,
           const unordered_map<std::string, T>& values) {
    bool rt = true;
    const std::string namespaced_prefix = namespaced_key(key_prefix);
    for (const auto& iter : values) {
      const std::string full_key = namespaced_prefix + iter.first;
      if (iter.second.empty()) {
        rt = rt && client_.rm(full_key).is_ok();
      } else {
        rt = rt && client_.put(full_key, iter.second.serialize_to_json().dump())
                       .is_ok();
      }
    }
    return rt;
  }

  bool set(const std::string& key_prefix, const XXH3KeyCacheMap& values);

  bool set(const std::string& key, const std::string& value);

  // create key-value with lease and transaction
  bool set(const std::string& key, const std::string& value, const int ttl);

  bool rm(const std::string& key);

  bool rm(const std::string& key_prefix,
          const std::unordered_set<std::string>& keys);

  template <typename T>
  bool get(const std::string& key, T* value) {
    auto response = client_.get(namespaced_key(key));
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd get " << key
                 << " failed: " << response.error_message();
      return false;
    }
    if (value) {
      return value->parse_from_json(response.value().as_string());
    } else {
      return true;
    }
  }

  bool get(const std::string& key_prefix, std::string* value);

  template <typename T>
  bool get_prefix(const std::string& key_prefix,
                  std::unordered_map<std::string, T>* values) {
    const std::string full_prefix = namespaced_key(key_prefix);
    auto response = client_.ls(full_prefix);
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd get " << key_prefix
                 << " failed: " << response.error_message();
      return false;
    }

    const size_t prefix_len = full_prefix.size();
    for (int i = 0; i < response.keys().size(); i++) {
      auto key_str = response.key(i).substr(prefix_len);
      auto json_str = response.value(i).as_string();

      T value;
      if (!value.parse_from_json(json_str)) {
        LOG(ERROR) << "Parse json fail: " << json_str;
        continue;
      }

      values->insert_or_assign(std::move(key_str), std::move(value));
    }
    return true;
  }

  bool get_prefix(const std::string& key_prefix, XXH3KeyCacheMap* values);

  bool get_prefix(const std::string& key_prefix,
                  std::unordered_map<std::string, std::string>* values);

  void add_watch(const std::string& key_prefix,
                 Callback callback,
                 bool recursive = true);

  void remove_watch(const std::string& key_prefix);

  void stop_watch();

 private:
  std::string namespaced_key(const std::string& logical_key) const;

  struct WatcherInfo {
    std::unique_ptr<etcd::Watcher> watcher;
    Callback callback;
  };

  etcd::SyncClient client_;
  std::string etcd_addr_;
  std::string etcd_namespace_prefix_;
  std::mutex watchers_mutex_;
  std::map<std::string, WatcherInfo> watchers_;
  std::vector<std::shared_ptr<etcd::KeepAlive>> keep_alives_;
};

}  // namespace xllm_service
