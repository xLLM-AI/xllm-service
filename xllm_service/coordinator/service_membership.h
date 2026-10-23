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
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "common/threadpool.h"
#include "config/service_config.h"
#include "coordinator/etcd_client.h"
namespace xllm_service {
class ServiceMembership final {
 public:
  ServiceMembership(const ServiceConfig& options,
                    std::shared_ptr<EtcdClient> etcd,
                    std::function<void()> on_master,
                    std::function<void()> on_heartbeat);
  ~ServiceMembership();
  bool start();
  void stop();
  bool is_master() const { return master_.load(); }

 private:
  bool register_current_service();
  void elect_master();
  void heartbeat_loop();
  ServiceConfig options_;
  std::shared_ptr<EtcdClient> etcd_client_;
  std::function<void()> on_master_;
  std::function<void()> on_heartbeat_;
  std::atomic<bool> stopped_{true};
  std::atomic<bool> master_{false};
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  std::unique_ptr<std::thread> heartbeat_;
  std::unique_ptr<ThreadPool> events_;
};
}  // namespace xllm_service
