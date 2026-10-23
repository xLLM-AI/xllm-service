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
#include <brpc/server.h>

#include <atomic>
#include <memory>
#include <thread>

#include "cluster/instance_controller.h"
#include "config/service_config.h"
#include "coordinator/instance_discovery.h"
#include "coordinator/service_membership.h"
#include "http_service/service.h"
#include "router/request_manager.h"
#include "router/router.h"
#include "rpc_service/service.h"
namespace xllm_service {
class ServiceApp final {
 public:
  explicit ServiceApp(const ServiceConfig& options);
  ~ServiceApp();
  bool start();
  void stop();

 private:
  bool setup_http_server();
  bool start_rpc_server();
  void manage_http_server_lifecycle();
  ServiceConfig options_;
  std::atomic<bool> stopped_{true};
  bool started_ = false;
  std::shared_ptr<EtcdClient> etcd_;
  std::shared_ptr<InstanceState> instances_;
  std::shared_ptr<RoutingState> routing_state_;
  std::unique_ptr<InstanceClient> instance_client_;
  std::unique_ptr<RequestManager> requests_;
  std::unique_ptr<InstanceController> controller_;
  std::unique_ptr<InstanceDiscovery> discovery_;
  std::unique_ptr<ServiceMembership> membership_;
  std::unique_ptr<KVDirectory> kv_directory_;
  std::unique_ptr<Router> router_;
  std::unique_ptr<XllmHttpServiceImpl> http_service_;
  std::unique_ptr<XllmRpcService> rpc_service_;
  std::unique_ptr<std::thread> readiness_thread_;
  brpc::Server http_server_;
  brpc::Server rpc_server_;
  bool rpc_running_ = false;
  brpc::ServerOptions http_options_;
  butil::EndPoint http_endpoint_;
  std::string http_server_address_;
  std::string rpc_server_address_;
};
}  // namespace xllm_service
