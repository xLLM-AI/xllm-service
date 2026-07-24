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

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace xllm_service {
namespace {

struct UsageCase {
  size_t num_prompt_tokens;
  size_t num_generated_tokens;
  size_t num_total_tokens;
  size_t num_prefix_cache_hit_tokens;
};

const std::vector<UsageCase> kUsageCases = {
    {/*num_prompt_tokens=*/8,
     /*num_generated_tokens=*/0,
     /*num_total_tokens=*/8,
     /*num_prefix_cache_hit_tokens=*/0},
    {/*num_prompt_tokens=*/8,
     /*num_generated_tokens=*/2,
     /*num_total_tokens=*/10,
     /*num_prefix_cache_hit_tokens=*/3},
    {/*num_prompt_tokens=*/8,
     /*num_generated_tokens=*/2,
     /*num_total_tokens=*/10,
     /*num_prefix_cache_hit_tokens=*/8},
};

llm::Usage make_usage(const UsageCase& usage_case) {
  llm::Usage usage;
  usage.num_prompt_tokens = usage_case.num_prompt_tokens;
  usage.num_generated_tokens = usage_case.num_generated_tokens;
  usage.num_total_tokens = usage_case.num_total_tokens;
  usage.num_prefix_cache_hit_tokens = usage_case.num_prefix_cache_hit_tokens;
  return usage;
}

TEST(UsageProtoAdapterTest, ConvertsPrefixCacheHitsToOpenAIPromptTokenDetails) {
  for (const UsageCase& usage_case : kUsageCases) {
    xllm::proto::Usage proto_usage =
        to_openai_usage_proto(make_usage(usage_case));

    EXPECT_EQ(proto_usage.prompt_tokens(), usage_case.num_prompt_tokens);
    EXPECT_EQ(proto_usage.completion_tokens(), usage_case.num_generated_tokens);
    EXPECT_EQ(proto_usage.total_tokens(), usage_case.num_total_tokens);
    ASSERT_TRUE(proto_usage.has_prompt_tokens_details());
    EXPECT_TRUE(proto_usage.prompt_tokens_details().has_cached_tokens());
    EXPECT_EQ(proto_usage.prompt_tokens_details().cached_tokens(),
              usage_case.num_prefix_cache_hit_tokens);
  }
}

TEST(UsageProtoAdapterTest,
     SplitsAnthropicInputTokensIntoUncachedAndCacheReadTokens) {
  for (const UsageCase& usage_case : kUsageCases) {
    xllm::proto::AnthropicUsage proto_usage =
        to_anthropic_usage_proto(make_usage(usage_case));

    EXPECT_EQ(
        proto_usage.input_tokens(),
        usage_case.num_prompt_tokens - usage_case.num_prefix_cache_hit_tokens);
    EXPECT_EQ(proto_usage.cache_read_input_tokens(),
              usage_case.num_prefix_cache_hit_tokens);
    EXPECT_EQ(proto_usage.output_tokens(), usage_case.num_generated_tokens);
    EXPECT_EQ(
        proto_usage.input_tokens() + proto_usage.cache_read_input_tokens(),
        usage_case.num_prompt_tokens);
    EXPECT_FALSE(proto_usage.has_cache_creation_input_tokens());
  }
}

}  // namespace
}  // namespace xllm_service
