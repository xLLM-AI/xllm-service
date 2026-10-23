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
#include <memory>

#include "cluster/instance_state.h"
#include "config/service_config.h"
#include "kv_conductor/kv_directory.h"
#include "request/request_preprocessor.h"
#include "router/routing_policy.h"
#include "router/routing_state.h"
namespace xllm_service {
class Router final {
 public:
  Router(const ServiceConfig& options,
         std::shared_ptr<RoutingState> routing,
         const KVDirectory* directory);
  bool schedule(std::shared_ptr<Request> request);

 private:
  std::shared_ptr<RoutingState> routing_;
  const KVDirectory* directory_;
  RequestPreprocessor preprocessor_;
  std::unique_ptr<RoutingPolicy> policy_;
};
}  // namespace xllm_service
