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
#include <mutex>
#include <thread>

#include "cluster/instance_client.h"
#include "cluster/instance_state.h"
#include "config/service_config.h"

namespace xllm_service {
namespace proto {
class HeartbeatRequest;
}
class InstanceController final {
 public:
  using RetirementCallback =
      std::function<bool(const std::string&, const std::string&)>;
  InstanceController(const ServiceConfig& options,
                     std::shared_ptr<InstanceState> instances,
                     InstanceClient& client,
                     RetirementCallback retire);
  ~InstanceController();
  void start();
  void stop();
  void on_instance_event(InstanceEvent event);
  void on_load_event(LoadEvent event);
  bool handle_instance_heartbeat(const proto::HeartbeatRequest* request);

 private:
  friend class ServiceTestPeer;
  bool register_instance(InstanceMetaInfo info);
  void deregister_instance(const std::string& name,
                           const std::string& incarnation_id = "");
  std::vector<std::pair<std::string, InstanceMetaInfo>> link_operations(
      const InstanceMetaInfo& info,
      bool removing) const;
  void reconcile_instance_states();
  ServiceConfig options_;
  std::shared_ptr<InstanceState> instances_;
  // Owned by the composition root, which outlives this controller.
  InstanceClient& client_;
  RetirementCallback retire_;
  std::mutex lifecycle_mutex_;
  std::atomic<bool> exited_{false};
  std::unique_ptr<std::thread> state_reconcile_thread_;
};
}  // namespace xllm_service
