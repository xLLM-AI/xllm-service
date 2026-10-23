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

#include "router/router.h"

#include <glog/logging.h>

#include <utility>

#include "router/policies/cache_aware_routing.h"
#include "router/policies/round_robin.h"
#include "router/policies/slo_aware.h"
namespace xllm_service {
Router::Router(const ServiceConfig& options,
               std::shared_ptr<RoutingState> routing,
               const KVDirectory* directory)
    : routing_(std::move(routing)),
      directory_(options.load_balance_policy() == "CAR" ? directory : nullptr),
      preprocessor_(options) {
  if (options.load_balance_policy() == "CAR") {
    policy_ = std::make_unique<CacheAwareRouting>(options.car_host_hit_weight(),
                                                  options.car_prefill_lambda());
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    policy_ = std::make_unique<SloAware>(options.target_ttft(),
                                         options.target_tpot());
  } else {
    policy_ = std::make_unique<RoundRobin>();
  }
}

bool Router::schedule(std::shared_ptr<Request> request) {
  if (!preprocessor_.process(*request)) {
    return false;
  }
  std::vector<KVDirectoryMatch> matches;
  if (directory_ != nullptr && !request->token_ids.empty()) {
    matches = directory_->match(request->token_ids);
  }
  return routing_->select_route(*request, *policy_, std::move(matches));
}
}  // namespace xllm_service
