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

#include <algorithm>

#include "request/request.h"

namespace xllm_service {

struct GenerationProgress {
  bool finished_prefill = false;
  int64_t generated_tokens = 0;
};

// Called for successful output under Scheduler's request lock. Streaming token
// IDs are deltas; non-streaming IDs and usage are cumulative, including P's
// first token. Empty parser/usage frames must not invent generated tokens.
inline GenerationProgress record_generation(Request& request,
                                            const llm::RequestOutput& output) {
  int64_t output_tokens = 0;
  for (const auto& sequence : output.outputs) {
    output_tokens += static_cast<int64_t>(sequence.token_ids.size());
  }
  int64_t total_tokens = request.stream
                             ? request.num_generated_tokens + output_tokens
                             : output_tokens;
  if (output.usage.has_value()) {
    total_tokens = static_cast<int64_t>(output.usage->num_generated_tokens);
  }
  total_tokens = std::max(total_tokens, request.num_generated_tokens);
  const int64_t generated_tokens = total_tokens - request.num_generated_tokens;
  const bool finished_prefill = !request.prefill_stage_finished &&
                                (output.finished_on_prefill_instance ||
                                 (!request.routing.decode_name.empty() &&
                                  (generated_tokens > 0 || output.finished)));
  request.prefill_stage_finished |= finished_prefill;
  request.num_generated_tokens = total_tokens;
  return {finished_prefill, generated_tokens};
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
                                         RequestMetrics& metrics,
                                         int64_t generated_tokens = 0) {
  const int64_t prompt_tokens = static_cast<int64_t>(request.token_ids.size());
  switch (action) {
    case RequestAction::SCHEDULE:
      ++metrics.decode_request_num;
      metrics.decode_token_num += prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
    case RequestAction::GENERATE:
      metrics.decode_token_num += generated_tokens;
      break;
    case RequestAction::FINISH_DECODE:
    case RequestAction::CANCEL:
      --metrics.decode_request_num;
      metrics.decode_token_num -= prompt_tokens + request.num_generated_tokens;
      break;
  }
}

}  // namespace xllm_service
