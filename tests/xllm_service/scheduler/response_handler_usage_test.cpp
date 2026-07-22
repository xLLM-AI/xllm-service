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

#include "scheduler/response_handler.h"

namespace xllm_service {
namespace {

TEST(ResponseHandlerUsageTest, IncludesCachedTokensInOpenAIUsage) {
  llm::Usage usage;
  usage.num_prompt_tokens = 8;
  usage.num_generated_tokens = 2;
  usage.num_total_tokens = 10;
  usage.num_cached_tokens = 6;
  xllm::proto::Usage proto_usage;

  set_openai_usage(&proto_usage, usage);

  EXPECT_EQ(proto_usage.prompt_tokens(), 8);
  EXPECT_EQ(proto_usage.completion_tokens(), 2);
  EXPECT_EQ(proto_usage.total_tokens(), 10);
  ASSERT_TRUE(proto_usage.has_prompt_tokens_details());
  EXPECT_TRUE(proto_usage.prompt_tokens_details().has_cached_tokens());
  EXPECT_EQ(proto_usage.prompt_tokens_details().cached_tokens(), 6);
}

TEST(ResponseHandlerUsageTest, IncludesZeroCachedTokensInOpenAIUsage) {
  llm::Usage usage;
  xllm::proto::Usage proto_usage;

  set_openai_usage(&proto_usage, usage);

  ASSERT_TRUE(proto_usage.has_prompt_tokens_details());
  EXPECT_TRUE(proto_usage.prompt_tokens_details().has_cached_tokens());
  EXPECT_EQ(proto_usage.prompt_tokens_details().cached_tokens(), 0);
}

}  // namespace
}  // namespace xllm_service
