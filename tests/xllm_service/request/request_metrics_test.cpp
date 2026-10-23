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

#include "request/request_metrics.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

class RequestMetricsTest : public testing::Test {
 protected:
  void SetUp() override {
    request_.routing.prefill_name = "prefill";
    request_.routing.decode_name = "decode";
    request_.token_ids = {1, 2, 3};
    request_.estimated_ttft = 20;
    prefill_.estimated_prefill_time = request_.estimated_ttft;
    apply(RequestAction::SCHEDULE);
  }

  void apply(RequestAction action) {
    apply_prefill_request_metrics(request_, action, prefill_);
    apply_decode_request_metrics(request_, action, decode_);
  }

  bool receive_token(bool legacy_prefill_flag = false) {
    const bool first_token = record_generation(request_, legacy_prefill_flag);
    apply(first_token ? RequestAction::FINISH_PREFILL
                      : RequestAction::GENERATE);
    return first_token;
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

TEST_F(RequestMetricsTest, DefaultRouteFinishReleasesPendingPrefill) {
  request_.routing.decode_name.clear();
  EXPECT_FALSE(
      record_generation(request_, /*finished_on_prefill_instance=*/false));
  EXPECT_FALSE(
      record_generation(request_, /*finished_on_prefill_instance=*/false));
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
  EXPECT_FALSE(
      record_generation(request_, /*finished_on_prefill_instance=*/false));
  EXPECT_FALSE(request_.prefill_stage_finished);
  ASSERT_TRUE(
      record_generation(request_, /*finished_on_prefill_instance=*/true));
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
