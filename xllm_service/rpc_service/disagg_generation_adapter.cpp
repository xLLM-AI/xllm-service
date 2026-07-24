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

#include "rpc_service/disagg_generation_adapter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace xllm_service {
namespace {

RequestOutputConversionResult invalid_usage(std::string message) {
  return {llm::Status(llm::StatusCode::INVALID_ARGUMENT, std::move(message)),
          std::nullopt};
}

RequestOutputConversionResult validate_usage(const proto::OutputUsage& usage) {
  if (usage.num_prompt_tokens() < 0 || usage.num_generated_tokens() < 0 ||
      usage.num_total_tokens() < 0 || usage.num_cached_tokens() < 0) {
    return invalid_usage("token counts must be non-negative");
  }
  if (usage.num_cached_tokens() > usage.num_prompt_tokens()) {
    return invalid_usage(
        "prefix cache hit tokens must not exceed prompt tokens");
  }
  const int64_t expected_total =
      static_cast<int64_t>(usage.num_prompt_tokens()) +
      static_cast<int64_t>(usage.num_generated_tokens());
  if (static_cast<int64_t>(usage.num_total_tokens()) != expected_total) {
    return invalid_usage(
        "total tokens must equal prompt tokens plus generated tokens");
  }
  return {};
}

}  // namespace

RequestOutputConversionResult request_output_from_disagg_generation(
    const proto::DisaggStreamGeneration& generation) {
  if (generation.has_usage()) {
    RequestOutputConversionResult validation =
        validate_usage(generation.usage());
    if (!validation.status.ok()) {
      return validation;
    }
  }

  llm::RequestOutput request_output;
  request_output.request_id = generation.req_id();
  request_output.service_request_id = generation.service_req_id();
  if (generation.has_gen_status()) {
    request_output.status = llm::Status(
        static_cast<llm::StatusCode>(generation.gen_status().status_code()),
        generation.gen_status().status_msg());
  }
  if (generation.has_usage()) {
    llm::Usage usage;
    usage.num_prompt_tokens =
        static_cast<size_t>(generation.usage().num_prompt_tokens());
    usage.num_generated_tokens =
        static_cast<size_t>(generation.usage().num_generated_tokens());
    usage.num_total_tokens =
        static_cast<size_t>(generation.usage().num_total_tokens());
    usage.num_cached_tokens =
        static_cast<size_t>(generation.usage().num_cached_tokens());
    request_output.usage = std::move(usage);
  }
  request_output.finished_on_prefill_instance =
      generation.finished_on_prefill_instance();
  request_output.finished = generation.finished();
  request_output.outputs.reserve(generation.outputs_size());
  for (const proto::SequenceOutput& output : generation.outputs()) {
    llm::SequenceOutput sequence_output;
    sequence_output.index = static_cast<size_t>(output.index());
    sequence_output.text = output.text();
    sequence_output.token_ids = std::vector<int32_t>(output.token_ids().begin(),
                                                     output.token_ids().end());
    if (!output.finish_reason().empty()) {
      sequence_output.finish_reason = output.finish_reason();
    }
    if (!output.logprobs().empty()) {
      std::vector<llm::LogProb> logprobs;
      logprobs.reserve(output.logprobs_size());
      for (const proto::LogProb& logprob : output.logprobs()) {
        llm::LogProb converted_logprob;
        converted_logprob.token = logprob.log_prob_data().token();
        converted_logprob.token_id = logprob.log_prob_data().token_id();
        converted_logprob.logprob = logprob.log_prob_data().logprob();
        converted_logprob.finished_token =
            logprob.log_prob_data().finished_token();
        if (!logprob.top_logprobs().empty()) {
          std::vector<llm::LogProbData> top_logprobs;
          top_logprobs.reserve(logprob.top_logprobs_size());
          for (const proto::LogProbData& top_logprob : logprob.top_logprobs()) {
            llm::LogProbData converted_top_logprob;
            converted_top_logprob.token = top_logprob.token();
            converted_top_logprob.token_id = top_logprob.token_id();
            converted_top_logprob.logprob = top_logprob.logprob();
            converted_top_logprob.finished_token = top_logprob.finished_token();
            top_logprobs.emplace_back(std::move(converted_top_logprob));
          }
          converted_logprob.top_logprobs = std::move(top_logprobs);
        }
        logprobs.emplace_back(std::move(converted_logprob));
      }
      sequence_output.logprobs = std::move(logprobs);
    }
    request_output.outputs.emplace_back(std::move(sequence_output));
  }
  return {llm::Status(), std::move(request_output)};
}

}  // namespace xllm_service
