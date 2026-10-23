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

#include "scheduler/request_metrics.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

llm::RequestOutput make_output(size_t token_count,
                               bool finished_on_prefill = false) {
  llm::RequestOutput output;
  output.outputs.emplace_back();
  output.outputs.back().index = 0;
  output.outputs.back().token_ids.resize(token_count, 42);
  output.finished_on_prefill_instance = finished_on_prefill;
  return output;
}

class RequestMetricsTest : public testing::Test {
 protected:
  void SetUp() override {
    request_.routing.prefill_name = "prefill";
    request_.routing.decode_name = "decode";
    request_.stream = true;
    request_.token_ids = {1, 2, 3};
    request_.estimated_ttft = 20;
    prefill_.estimated_prefill_time = request_.estimated_ttft;
    apply(RequestAction::SCHEDULE);
  }

  void apply(RequestAction action, int64_t generated_tokens = 0) {
    apply_prefill_request_metrics(request_, action, prefill_);
    apply_decode_request_metrics(request_, action, decode_, generated_tokens);
  }

  bool receive_output(const llm::RequestOutput& output) {
    const GenerationProgress progress = record_generation(request_, output);
    apply(progress.finished_prefill ? RequestAction::FINISH_PREFILL
                                    : RequestAction::GENERATE,
          progress.generated_tokens);
    return progress.finished_prefill;
  }

  bool receive_token(bool legacy_prefill_flag = false) {
    return receive_output(make_output(1, legacy_prefill_flag));
  }

  void expect_released() const {
    EXPECT_EQ(prefill_.prefill_request_num, 0);
    EXPECT_EQ(prefill_.prefill_token_num, 0);
    EXPECT_EQ(prefill_.estimated_prefill_time, 0);
    EXPECT_EQ(decode_.decode_request_num, 0);
    EXPECT_EQ(decode_.decode_token_num, 0);
  }

  Request request_;
  RequestMetrics prefill_;
  RequestMetrics decode_;
};

TEST_F(RequestMetricsTest, DecodeFirstTokenFinishesPrefillExactlyOnce) {
  EXPECT_TRUE(receive_token());
  EXPECT_TRUE(request_.prefill_stage_finished);
  EXPECT_EQ(prefill_.prefill_request_num, 0);
  EXPECT_EQ(prefill_.prefill_token_num, 0);
  EXPECT_EQ(prefill_.estimated_prefill_time, 0);
  EXPECT_EQ(decode_.decode_token_num, 4);
  EXPECT_FALSE(receive_token());
  EXPECT_FALSE(receive_token());
  EXPECT_EQ(prefill_.prefill_request_num, 0);
  EXPECT_EQ(request_.num_generated_tokens, 3);
  apply(RequestAction::FINISH_DECODE);
  expect_released();
}

TEST_F(RequestMetricsTest, CancelBeforeFirstTokenReleasesBothReservations) {
  EXPECT_FALSE(request_.prefill_stage_finished);
  apply(RequestAction::CANCEL);
  expect_released();
}

TEST_F(RequestMetricsTest, CancelAfterHandoffDoesNotReleasePrefillTwice) {
  EXPECT_TRUE(receive_token());
  EXPECT_FALSE(receive_token());
  apply(RequestAction::CANCEL);
  expect_released();
}

TEST_F(RequestMetricsTest, LegacyPrefillFlagAndDecodeOutputShareOneTransition) {
  EXPECT_TRUE(receive_token(/*legacy_prefill_flag=*/true));
  EXPECT_FALSE(receive_token());
  EXPECT_FALSE(receive_token(/*legacy_prefill_flag=*/true));
  apply(RequestAction::FINISH_DECODE);
  expect_released();
}

TEST_F(RequestMetricsTest, StreamingMtpCountsTokensAndIgnoresEmptyFrames) {
  EXPECT_FALSE(receive_output(make_output(0)));
  EXPECT_FALSE(request_.prefill_stage_finished);
  EXPECT_EQ(prefill_.prefill_request_num, 1);
  EXPECT_TRUE(receive_output(make_output(4)));
  EXPECT_EQ(request_.num_generated_tokens, 4);
  EXPECT_EQ(decode_.decode_token_num, 7);
  EXPECT_FALSE(receive_output(make_output(0)));
  EXPECT_FALSE(receive_output(make_output(3)));
  EXPECT_EQ(request_.num_generated_tokens, 7);
  EXPECT_EQ(decode_.decode_token_num, 10);
  apply(RequestAction::CANCEL);
  expect_released();
}

TEST_F(RequestMetricsTest, CumulativeNonStreamIncludesPrefillFirstTokenOnce) {
  request_.stream = false;
  EXPECT_TRUE(receive_output(make_output(1, /*finished_on_prefill=*/true)));
  EXPECT_FALSE(receive_output(make_output(4)));
  EXPECT_FALSE(receive_output(make_output(4)));
  EXPECT_EQ(request_.num_generated_tokens, 4);
  EXPECT_EQ(decode_.decode_token_num, 7);
  apply(RequestAction::FINISH_DECODE);
  expect_released();
}

TEST_F(RequestMetricsTest, CumulativeUsageDoesNotDoubleCountStreamingTokens) {
  EXPECT_TRUE(receive_output(make_output(1, /*finished_on_prefill=*/true)));
  EXPECT_FALSE(receive_output(make_output(3)));
  auto output = make_output(0);
  output.usage.emplace();
  output.usage->num_generated_tokens = 4;
  EXPECT_FALSE(receive_output(output));
  EXPECT_FALSE(receive_output(output));
  EXPECT_EQ(request_.num_generated_tokens, 4);
  output.usage->num_generated_tokens = 6;
  EXPECT_FALSE(receive_output(output));
  EXPECT_EQ(request_.num_generated_tokens, 6);
  EXPECT_EQ(decode_.decode_token_num, 9);
  apply(RequestAction::FINISH_DECODE);
  expect_released();
}

TEST_F(RequestMetricsTest, HandoffWithoutTokensReleasesPrefillOnly) {
  EXPECT_TRUE(receive_output(make_output(0, /*finished_on_prefill=*/true)));
  EXPECT_EQ(request_.num_generated_tokens, 0);
  EXPECT_EQ(prefill_.estimated_prefill_time, 0);
  EXPECT_FALSE(receive_output(make_output(4)));
  EXPECT_EQ(decode_.decode_token_num, 7);
  apply(RequestAction::CANCEL);
  expect_released();
}

TEST_F(RequestMetricsTest, DefaultRouteFinishReleasesPendingPrefill) {
  request_.routing.decode_name.clear();
  EXPECT_FALSE(record_generation(request_, make_output(1)).finished_prefill);
  EXPECT_FALSE(record_generation(request_, make_output(1)).finished_prefill);
  ASSERT_FALSE(request_.prefill_stage_finished);
  ASSERT_EQ(prefill_.prefill_request_num, 1);

  apply_prefill_request_metrics(
      request_, RequestAction::FINISH_DECODE, prefill_);

  EXPECT_EQ(prefill_.prefill_request_num, 0);
  EXPECT_EQ(prefill_.prefill_token_num, 0);
  EXPECT_EQ(prefill_.estimated_prefill_time, 0);
}

TEST_F(RequestMetricsTest, DefaultRouteFinishKeepsReleasedPrefillUnchanged) {
  request_.routing.decode_name.clear();
  EXPECT_FALSE(record_generation(request_, make_output(1)).finished_prefill);
  EXPECT_FALSE(request_.prefill_stage_finished);
  ASSERT_TRUE(
      record_generation(request_, make_output(1, /*finished_on_prefill=*/true))
          .finished_prefill);
  apply_prefill_request_metrics(
      request_, RequestAction::FINISH_PREFILL, prefill_);

  apply_prefill_request_metrics(
      request_, RequestAction::FINISH_DECODE, prefill_);

  EXPECT_EQ(prefill_.prefill_request_num, 0);
  EXPECT_EQ(prefill_.prefill_token_num, 0);
  EXPECT_EQ(prefill_.estimated_prefill_time, 0);
}

}  // namespace
}  // namespace xllm_service
