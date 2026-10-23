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

#include <atomic>
#include <memory>

#include "kv_conductor/kv_directory.h"
#include "router/routing_policy.h"

namespace xllm_service {

class CacheAwareRouting final : public RoutingPolicy {
 public:
  explicit CacheAwareRouting(double host_hit_weight = 0.5,
                             double prefill_lambda = 0.5);
  bool select(const RoutingInput& input, RoutingDecision* decision) override;

 private:
  const double host_hit_weight_;
  const double prefill_lambda_;
  std::atomic<uint64_t> next_selection_{0};
};

}  // namespace xllm_service
