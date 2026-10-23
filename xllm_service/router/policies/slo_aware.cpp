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

#include "router/policies/slo_aware.h"

#include <algorithm>
namespace xllm_service {
SloAware::SloAware(int64_t target_ttft, int64_t target_tpot)
    : target_ttft_(target_ttft), target_tpot_(target_tpot) {}

bool SloAware::select(const RoutingInput& input, RoutingDecision* decision) {
  if (input.prompt_tokens == 0) {
    return round_robin_.select(input, decision);
  }
  const auto& prefill = input.candidates.prefill;
  const auto& decode = input.candidates.decode;
  if (prefill.empty() || decode.empty()) {
    return false;
  }
  const auto p = std::min_element(
      prefill.begin(), prefill.end(), [](const auto& a, const auto& b) {
        return a.prefill_time < b.prefill_time;
      });
  const auto d = std::min_element(
      decode.begin(), decode.end(), [](const auto& a, const auto& b) {
        return a.estimated_tpot < b.estimated_tpot;
      });
  const auto target =
      std::find_if(decode.begin(), decode.end(), [this](const auto& item) {
        return item.estimated_tpot <= target_tpot_;
      });
  decision->routing.decode_name = (target == decode.end() ? d : target)->name;
  int64_t total_prefill_time = 0;
  for (const auto& item : prefill) {
    total_prefill_time += item.prefill_time;
  }
  const int64_t avg_prefill_time = total_prefill_time / prefill.size();
  const float tpot_threshold =
      (static_cast<float>(decode.size()) - 1.0f) / decode.size();
  const bool use_decode_for_prefill =
      p->prefill_time > target_ttft_ && target != d &&
      d->estimated_tpot < target_tpot_ * tpot_threshold &&
      d->prefill_time < p->prefill_time;
  const auto& selected_prefill = use_decode_for_prefill ? *d : *p;
  decision->routing.prefill_name = selected_prefill.name;
  decision->estimated_ttft = selected_prefill.estimated_ttft;
  const float ttft_threshold =
      (static_cast<float>(prefill.size()) - 1.0f) / prefill.size();
  decision->flip_prefill_to_decode =
      target == decode.end() &&
      (avg_prefill_time < target_ttft_ * ttft_threshold ||
       decode.size() < prefill.size());
  return true;
}
}  // namespace xllm_service
