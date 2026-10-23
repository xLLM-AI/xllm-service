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
#include "router/policies/round_robin.h"
namespace xllm_service {
class SloAware final : public RoutingPolicy {
 public:
  SloAware(int64_t target_ttft, int64_t target_tpot);
  bool select(const RoutingInput& input, RoutingDecision* decision) override;

 private:
  int64_t target_ttft_;
  int64_t target_tpot_;
  RoundRobin round_robin_;
};
}  // namespace xllm_service
