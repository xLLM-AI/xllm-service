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

#include "chat_template/message_projection.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(MessageProjectionTest, CopiesRoleAndStringContent) {
  Message msg("user", "hello");

  xllm::proto::ChatMessage proto;
  to_proto(msg, &proto);

  EXPECT_EQ(proto.role(), "user");
  EXPECT_EQ(proto.content(), "hello");
}

TEST(MessageProjectionTest, FlattensContentVecToTextBlocksOnly) {
  Message::MMContentVec blocks;
  Message::MMContent thinking("thinking");
  thinking.thinking = "hidden";
  blocks.push_back(thinking);
  Message::MMContent tool_use("tool_use");
  tool_use.id = "toolu_1";
  blocks.push_back(tool_use);
  blocks.emplace_back("text", "first");
  blocks.emplace_back("text", "second");

  Message msg("assistant", blocks);

  xllm::proto::ChatMessage proto;
  to_proto(msg, &proto);

  EXPECT_EQ(proto.role(), "assistant");
  EXPECT_EQ(proto.content(), "first\nsecond");
}

TEST(MessageProjectionTest, ProjectsToolCallsAndReasoningContent) {
  Message msg("assistant", "done");
  msg.reasoning_content = "hidden reasoning";
  Message::ToolCall call;
  call.id = "toolu_1";
  call.type = "function";
  call.function.name = "get_weather";
  call.function.arguments = R"({"city":"Beijing"})";
  msg.tool_calls = Message::ToolCallVec{call};

  xllm::proto::ChatMessage proto;
  to_proto(msg, &proto);

  ASSERT_TRUE(proto.has_reasoning_content());
  EXPECT_EQ(proto.reasoning_content(), "hidden reasoning");
  ASSERT_EQ(proto.tool_calls_size(), 1);
  EXPECT_EQ(proto.tool_calls(0).id(), "toolu_1");
  EXPECT_EQ(proto.tool_calls(0).type(), "function");
  EXPECT_EQ(proto.tool_calls(0).function().name(), "get_weather");
  EXPECT_EQ(proto.tool_calls(0).function().arguments(),
            R"({"city":"Beijing"})");
}

TEST(MessageProjectionTest, OmitsEmptyReasoningContentAndToolCallId) {
  Message msg("user", "hi");
  msg.reasoning_content = "";

  xllm::proto::ChatMessage proto;
  to_proto(msg, &proto);

  EXPECT_FALSE(proto.has_reasoning_content());
  EXPECT_FALSE(proto.has_tool_call_id());
  EXPECT_EQ(proto.tool_calls_size(), 0);
}

TEST(MessageProjectionTest, SetsToolCallIdForToolRole) {
  Message msg("tool", "sunny");
  msg.tool_call_id = "toolu_1";

  xllm::proto::ChatMessage proto;
  to_proto(msg, &proto);

  EXPECT_EQ(proto.role(), "tool");
  EXPECT_EQ(proto.content(), "sunny");
  ASSERT_TRUE(proto.has_tool_call_id());
  EXPECT_EQ(proto.tool_call_id(), "toolu_1");
}

TEST(MessageProjectionTest, NeedsContentVecRules) {
  EXPECT_FALSE(needs_content_vec({}));

  Message::MMContentVec single_text;
  single_text.emplace_back("text", "only");
  EXPECT_FALSE(needs_content_vec(single_text));

  Message::MMContentVec single_thinking;
  single_thinking.emplace_back("thinking");
  EXPECT_TRUE(needs_content_vec(single_thinking));

  Message::MMContentVec two_texts;
  two_texts.emplace_back("text", "a");
  two_texts.emplace_back("text", "b");
  EXPECT_TRUE(needs_content_vec(two_texts));
}

}  // namespace
}  // namespace xllm_service
