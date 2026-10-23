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
#include <mutex>
#include <string>
#include <unordered_map>

#include "cluster/instance_state.h"
#include "common/time_predictor.h"
#include "config/service_config.h"
#include "request/request.h"
#include "router/routing_policy.h"

namespace xllm_service {
// Service-local reservations and predictors, separate from engine observations.
class RoutingState final {
 public:
  RoutingState(const ServiceConfig& options,
               std::shared_ptr<InstanceState> instances);
  bool select_route(Request& request,
                    RoutingPolicy& policy,
                    std::vector<KVDirectoryMatch> matches);
  bool bind_request(Request& request) const;
  void update_request_metrics(const std::shared_ptr<Request>& request,
                              RequestAction action);
  void forget_instance(const std::string& name,
                       const std::string& incarnation_id);
  RequestMetrics request_metrics(const std::string& name) const;

 private:
  class Reservation final {
   public:
    explicit Reservation(const InstanceMetaInfo& info);
    std::string incarnation_id;
    RequestMetrics metrics;
    TimePredictor predictor;
  };
  Reservation& reservation_for(const InstanceMetaInfo& info);
  void update_locked(const Request& request, RequestAction action);
  const bool enabled_;
  std::shared_ptr<InstanceState> instances_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Reservation>> reservations_;
};
}  // namespace xllm_service
