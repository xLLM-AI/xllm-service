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

#include "request/request.h"

namespace xllm_service {

// Called for successful output under RequestManager's request lock. Decode-only
// output proves Prefill has handed off, even without the legacy P-side flag.
inline bool record_generation(Request& request,
                              bool finished_on_prefill_instance) {
  const bool first_token =
      !request.prefill_stage_finished &&
      (finished_on_prefill_instance || !request.routing.decode_name.empty());
  request.prefill_stage_finished |= first_token;
  ++request.num_generated_tokens;
  return first_token;
}

inline void apply_prefill_request_metrics(const Request& request,
                                          RequestAction action,
                                          RequestMetrics& metrics) {
  const int64_t prompt_tokens = static_cast<int64_t>(request.token_ids.size());
  if (action == RequestAction::SCHEDULE) {
    ++metrics.prefill_request_num;
    metrics.prefill_token_num += prompt_tokens;
  } else if (action == RequestAction::FINISH_PREFILL ||
             ((action == RequestAction::FINISH_DECODE ||
               action == RequestAction::CANCEL) &&
              !request.prefill_stage_finished)) {
    --metrics.prefill_request_num;
    metrics.prefill_token_num -= prompt_tokens;
    metrics.estimated_prefill_time -= request.estimated_ttft;
  }
}

inline void apply_decode_request_metrics(const Request& request,
                                         RequestAction action,
                                         RequestMetrics& metrics) {
  const int64_t prompt_tokens = static_cast<int64_t>(request.token_ids.size());
  switch (action) {
    case RequestAction::SCHEDULE:
      ++metrics.decode_request_num;
      metrics.decode_token_num += prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
    case RequestAction::GENERATE:
      ++metrics.decode_token_num;
      break;
    case RequestAction::FINISH_DECODE:
    case RequestAction::CANCEL:
      --metrics.decode_request_num;
      metrics.decode_token_num -= prompt_tokens + request.num_generated_tokens;
      break;
  }
}

}  // namespace xllm_service
