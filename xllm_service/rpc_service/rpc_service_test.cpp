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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "disagg_pd.pb.h"
#include "rpc_service/disagg_generation_adapter.h"

namespace xllm_service {
namespace {

proto::DisaggStreamGeneration make_generation(int32_t num_prompt_tokens,
                                              int32_t num_generated_tokens,
                                              int32_t num_total_tokens,
                                              int32_t num_cache_hit_tokens) {
  proto::DisaggStreamGeneration generation;
  generation.set_req_id("request-123");
  generation.mutable_usage()->set_num_prompt_tokens(num_prompt_tokens);
  generation.mutable_usage()->set_num_generated_tokens(num_generated_tokens);
  generation.mutable_usage()->set_num_total_tokens(num_total_tokens);
  generation.mutable_usage()->set_num_cached_tokens(num_cache_hit_tokens);
  return generation;
}

TEST(DisaggGenerationAdapterTest,
     PrefixCacheHitTokensRemainWireCompatibleAtFieldFour) {
  const google::protobuf::FieldDescriptor* sender_field =
      xllm::proto::OutputUsage::descriptor()->FindFieldByName(
          "num_cached_tokens");
  const google::protobuf::FieldDescriptor* receiver_field =
      proto::OutputUsage::descriptor()->FindFieldByName("num_cached_tokens");

  ASSERT_NE(sender_field, nullptr);
  ASSERT_NE(receiver_field, nullptr);
  EXPECT_EQ(sender_field->number(), 4);
  EXPECT_EQ(receiver_field->number(), 4);
  EXPECT_EQ(sender_field->type(), receiver_field->type());

  xllm::proto::OutputUsage sender_usage;
  sender_usage.set_num_prompt_tokens(8);
  sender_usage.set_num_generated_tokens(2);
  sender_usage.set_num_total_tokens(10);
  sender_usage.set_num_cached_tokens(6);

  proto::OutputUsage receiver_usage;
  ASSERT_TRUE(receiver_usage.ParseFromString(sender_usage.SerializeAsString()));
  EXPECT_EQ(receiver_usage.num_cached_tokens(), 6);
}

TEST(DisaggGenerationAdapterTest, RejectsNegativeTokenCounts) {
  const proto::DisaggStreamGeneration generation =
      make_generation(/*num_prompt_tokens=*/-1,
                      /*num_generated_tokens=*/2,
                      /*num_total_tokens=*/1,
                      /*num_cache_hit_tokens=*/0);

  RequestOutputConversionResult result =
      request_output_from_disagg_generation(generation);

  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(result.status.code(), llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(result.status.message().find("non-negative"), std::string::npos);
  EXPECT_FALSE(result.output.has_value());
}

TEST(DisaggGenerationAdapterTest, RejectsCacheHitsGreaterThanPromptTokens) {
  const proto::DisaggStreamGeneration generation =
      make_generation(/*num_prompt_tokens=*/8,
                      /*num_generated_tokens=*/2,
                      /*num_total_tokens=*/10,
                      /*num_cache_hit_tokens=*/9);

  RequestOutputConversionResult result =
      request_output_from_disagg_generation(generation);

  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(result.status.code(), llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(result.status.message().find("prefix cache hit"),
            std::string::npos);
  EXPECT_FALSE(result.output.has_value());
}

TEST(DisaggGenerationAdapterTest, RejectsInconsistentTotalTokens) {
  const proto::DisaggStreamGeneration generation =
      make_generation(/*num_prompt_tokens=*/8,
                      /*num_generated_tokens=*/2,
                      /*num_total_tokens=*/11,
                      /*num_cache_hit_tokens=*/6);

  RequestOutputConversionResult result =
      request_output_from_disagg_generation(generation);

  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(result.status.code(), llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(result.status.message().find("total tokens"), std::string::npos);
  EXPECT_FALSE(result.output.has_value());
}

TEST(DisaggGenerationAdapterTest, ConvertsCompleteValidGeneration) {
  proto::DisaggStreamGeneration generation =
      make_generation(/*num_prompt_tokens=*/8,
                      /*num_generated_tokens=*/2,
                      /*num_total_tokens=*/10,
                      /*num_cache_hit_tokens=*/6);
  generation.set_service_req_id("service-request-456");
  generation.mutable_gen_status()->set_status_code(
      static_cast<int32_t>(llm::StatusCode::OK));
  generation.mutable_gen_status()->set_status_msg("complete");
  generation.set_finished(true);
  generation.set_finished_on_prefill_instance(true);

  proto::SequenceOutput* sequence = generation.add_outputs();
  sequence->set_index(3);
  sequence->set_text("answer");
  sequence->add_token_ids(101);
  sequence->add_token_ids(102);
  sequence->set_finish_reason("stop");
  proto::LogProb* logprob = sequence->add_logprobs();
  logprob->mutable_log_prob_data()->set_token("answer");
  logprob->mutable_log_prob_data()->set_token_id(102);
  logprob->mutable_log_prob_data()->set_logprob(-0.25f);
  logprob->mutable_log_prob_data()->set_finished_token(true);
  proto::LogProbData* top_logprob = logprob->add_top_logprobs();
  top_logprob->set_token("result");
  top_logprob->set_token_id(103);
  top_logprob->set_logprob(-0.5f);
  top_logprob->set_finished_token(false);

  RequestOutputConversionResult result =
      request_output_from_disagg_generation(generation);

  ASSERT_TRUE(result.status.ok()) << result.status.message();
  ASSERT_TRUE(result.output.has_value());
  const llm::RequestOutput& output = result.output.value();
  EXPECT_EQ(output.request_id, "request-123");
  EXPECT_EQ(output.service_request_id, "service-request-456");
  ASSERT_TRUE(output.status.has_value());
  EXPECT_TRUE(output.status->ok());
  EXPECT_EQ(output.status->message(), "complete");
  EXPECT_TRUE(output.finished);
  EXPECT_TRUE(output.finished_on_prefill_instance);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_prompt_tokens, 8u);
  EXPECT_EQ(output.usage->num_generated_tokens, 2u);
  EXPECT_EQ(output.usage->num_total_tokens, 10u);
  EXPECT_EQ(output.usage->num_cached_tokens, 6u);

  ASSERT_EQ(output.outputs.size(), 1u);
  const llm::SequenceOutput& converted_sequence = output.outputs.front();
  EXPECT_EQ(converted_sequence.index, 3u);
  EXPECT_EQ(converted_sequence.text, "answer");
  EXPECT_EQ(converted_sequence.token_ids, (std::vector<int32_t>{101, 102}));
  ASSERT_TRUE(converted_sequence.finish_reason.has_value());
  EXPECT_EQ(converted_sequence.finish_reason.value(), "stop");
  ASSERT_TRUE(converted_sequence.logprobs.has_value());
  ASSERT_EQ(converted_sequence.logprobs->size(), 1u);
  const llm::LogProb& converted_logprob = converted_sequence.logprobs->front();
  EXPECT_EQ(converted_logprob.token, "answer");
  EXPECT_EQ(converted_logprob.token_id, 102);
  EXPECT_FLOAT_EQ(converted_logprob.logprob, -0.25f);
  EXPECT_TRUE(converted_logprob.finished_token);
  ASSERT_TRUE(converted_logprob.top_logprobs.has_value());
  ASSERT_EQ(converted_logprob.top_logprobs->size(), 1u);
  EXPECT_EQ(converted_logprob.top_logprobs->front().token, "result");
  EXPECT_EQ(converted_logprob.top_logprobs->front().token_id, 103);
  EXPECT_FLOAT_EQ(converted_logprob.top_logprobs->front().logprob, -0.5f);
  EXPECT_FALSE(converted_logprob.top_logprobs->front().finished_token);
}

}  // namespace
}  // namespace xllm_service
