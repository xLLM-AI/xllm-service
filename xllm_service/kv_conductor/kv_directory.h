/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "cluster/instance_types.h"
#include "kv_conductor/kv_index.h"

namespace xllm_service {

struct KVEventSource {
  std::string instance_name;
  std::string incarnation_id;
  uint64_t register_ts_ms = 0;
  std::string endpoint;
  std::string replay_endpoint;
  std::string topic;
  int32_t dp_rank = 0;
  uint32_t block_size = 128;
  uint64_t hash_seed = 1024;
};

struct KVEventInstance {
  std::string name;
  std::string incarnation_id;
  uint64_t register_ts_ms = 0;
  int32_t dp_size = 1;
};

std::vector<KVEventInstance> make_kv_event_instances(
    const std::vector<InstanceView>& instances);

struct KVDiscoveryConfig {
  bool enabled = false;
  std::string topic;
  uint32_t block_size = 128;
  uint64_t hash_seed = 1024;
  uint32_t event_port_offset = 10000;
  uint32_t replay_port_offset = 30000;
  int32_t refresh_interval_ms = 1000;
};

struct KVDirectoryConfig {
  std::vector<KVEventSource> sources;
  KVDiscoveryConfig discovery;
  size_t max_streams = 128;
  size_t max_blocks_per_stream = 262144;
  size_t max_message_bytes = 8 * 1024 * 1024;
  size_t max_buffered_bytes = 16 * 1024 * 1024;
  int32_t replay_timeout_ms = 2000;
  int32_t resync_interval_ms = 30000;
};

struct KVDirectoryMatch {
  std::string instance_name;
  int32_t dp_rank = 0;
  KVPrefixMatch prefix;
  std::string incarnation_id;
  uint64_t register_ts_ms = 0;
};

bool load_kv_directory_config(const std::string& path,
                              KVDirectoryConfig* config,
                              std::string* error);

// Each source owns its socket thread and index. Discovery reconciles a local
// instance snapshot; request handlers never wait for discovery or recovery.
class KVDirectory final {
 public:
  using InstanceProvider = std::function<std::vector<KVEventInstance>()>;
  explicit KVDirectory(KVDirectoryConfig config);
  ~KVDirectory();
  KVDirectory(const KVDirectory&) = delete;
  KVDirectory& operator=(const KVDirectory&) = delete;

  void start(InstanceProvider provider = {});
  void stop();
  std::vector<KVDirectoryMatch> match(
      const std::vector<int32_t>& token_ids) const;

 private:
  class Stream;
  void discover(const InstanceProvider& provider);
  void reconcile(const std::vector<KVEventInstance>& instances);

  KVDirectoryConfig config_;
  mutable std::shared_mutex streams_mutex_;
  std::vector<std::shared_ptr<Stream>> streams_;
  std::mutex lifecycle_mutex_;
  std::mutex wait_mutex_;
  std::condition_variable wakeup_;
  std::thread discovery_thread_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopped_{false};
};

}  // namespace xllm_service
