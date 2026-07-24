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

#include "http_service/usage_proto_adapter.h"

#include <cstdint>

namespace xllm_service {

xllm::proto::Usage to_openai_usage_proto(const llm::Usage& usage) {
  xllm::proto::Usage proto_usage;
  proto_usage.set_prompt_tokens(static_cast<int32_t>(usage.num_prompt_tokens));
  proto_usage.set_completion_tokens(
      static_cast<int32_t>(usage.num_generated_tokens));
  proto_usage.set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  proto_usage.mutable_prompt_tokens_details()->set_cached_tokens(
      static_cast<int32_t>(usage.num_prefix_cache_hit_tokens));
  return proto_usage;
}

xllm::proto::AnthropicUsage to_anthropic_usage_proto(const llm::Usage& usage) {
  xllm::proto::AnthropicUsage proto_usage;
  proto_usage.set_input_tokens(static_cast<int32_t>(
      usage.num_prompt_tokens - usage.num_prefix_cache_hit_tokens));
  proto_usage.set_output_tokens(
      static_cast<int32_t>(usage.num_generated_tokens));
  proto_usage.set_cache_read_input_tokens(
      static_cast<int32_t>(usage.num_prefix_cache_hit_tokens));
  return proto_usage;
}

}  // namespace xllm_service
