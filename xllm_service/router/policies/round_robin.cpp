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

#include "router/policies/round_robin.h"
namespace xllm_service {
bool RoundRobin::select(const RoutingInput& input, RoutingDecision* decision) {
  const auto& candidates = input.candidates;
  if (candidates.prefill.empty()) {
    return false;
  }
  const auto& prefill =
      candidates
          .prefill[next_prefill_.fetch_add(1) % candidates.prefill.size()];
  if (candidates.decode.empty() && prefill.type != InstanceType::DEFAULT) {
    return false;
  }
  decision->routing.prefill_name = prefill.name;
  decision->routing.decode_name.clear();
  if (!candidates.decode.empty()) {
    decision->routing.decode_name =
        candidates.decode[next_decode_.fetch_add(1) % candidates.decode.size()]
            .name;
  }
  return true;
}
}  // namespace xllm_service
