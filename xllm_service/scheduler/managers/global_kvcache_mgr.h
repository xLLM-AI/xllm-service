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

#include <shared_mutex>
#include <thread>

#include "../etcd_client/etcd_client.h"
#include "common/hash_util.h"
#include "common/macros.h"
#include "common/options.h"
#include "common/slice.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class GlobalKVCacheMgr final {
 public:
  explicit GlobalKVCacheMgr(const Options& options,
                            const std::shared_ptr<EtcdClient>& etcd_client,
                            const bool is_master_service);
  ~GlobalKVCacheMgr();

  void match(const Slice<int32_t>& token_ids, OverlapScores* overlap_scores);

  void record_updated_kvcaches(const std::string& instance_name,
                               const proto::KvCacheEvent& kvcache_event);
  bool upload_kvcache();

  void set_as_master();

 private:
  DISALLOW_COPY_AND_ASSIGN(GlobalKVCacheMgr);

  void update_kvcache(const etcd::Response& response,
                      const uint64_t prefix_len);

 private:
  Options options_;
  std::atomic_bool is_master_service_ = false;
  bool exited_ = false;
  std::shared_mutex kvcache_mutex_;
  XXH3KeyCacheMap kvcache_infos_;
  std::shared_ptr<EtcdClient> etcd_client_;  // not own

  std::mutex update_mutex_;
  XXH3KeyCacheMap updated_kvcaches_;

  ThreadPool threadpool_;
};

}  // namespace xllm_service
