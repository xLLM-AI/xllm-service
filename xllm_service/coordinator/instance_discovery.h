/* Copyright 2026 The xLLM Authors.

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

#include <atomic>
#include <functional>
#include <memory>

#include "cluster/instance_types.h"
#include "common/threadpool.h"
#include "coordinator/etcd_client.h"

namespace xllm_service {
class InstanceState;
class InstanceDiscovery final {
 public:
  using InstanceCallback = std::function<void(InstanceEvent)>;
  using LoadCallback = std::function<void(LoadEvent)>;
  InstanceDiscovery(std::shared_ptr<EtcdClient> etcd,
                    std::shared_ptr<InstanceState> instances,
                    InstanceCallback instance_callback,
                    LoadCallback load_callback);
  ~InstanceDiscovery();
  void start(bool master);
  void stop();
  void set_as_master();
  bool upload_load_metrics();

 private:
  void handle_instances(const etcd::Response& response, uint64_t prefix_len);
  void handle_load(const etcd::Response& response, uint64_t prefix_len);
  std::shared_ptr<EtcdClient> etcd_client_;
  std::shared_ptr<InstanceState> instances_;
  InstanceCallback instance_callback_;
  LoadCallback load_callback_;
  std::atomic<bool> stopped_{true};
  std::atomic<bool> master_{false};
  // Declared last: joins callbacks before their captures can be destroyed.
  std::unique_ptr<ThreadPool> events_;
};
}  // namespace xllm_service
