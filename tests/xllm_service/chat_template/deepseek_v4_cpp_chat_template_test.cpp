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

#include "chat_template/deepseek_v4_cpp_chat_template.h"

#include <gtest/gtest.h>

#include <memory>

#include "chat_template/chat_template.h"
#include "chat_template/jinja_chat_template.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {
namespace {

TokenizerArgs make_v4_args() {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  return args;
}

TEST(DeepseekV4CppChatTemplate, EncodeDoesNotAddSpecialTokens) {
  std::unique_ptr<ChatTemplate> tmpl =
      std::make_unique<DeepseekV4CppChatTemplate>(make_v4_args());
  EXPECT_FALSE(tmpl->encode_add_special_tokens());
}

TEST(DeepseekV4CppChatTemplate, PlainTextMatchesUpstreamGolden) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  ChatMessages messages;
  messages.emplace_back("system", "You are a helpful assistant.");
  messages.emplace_back("user", "Hello");

  auto prompt = tmpl.apply(messages, {}, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant."
            "<｜User｜>Hello<｜Assistant｜></think>");
}

TEST(DeepseekV4CppChatTemplate, MMContentVecIsFlattened) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  // Adjacent text blocks are flattened (joined by '\n') for the upstream
  // template.
  Message::MMContentVec blocks{Message::MMContent("text", "first"),
                               Message::MMContent("text", "second")};
  ChatMessages messages{Message("user", blocks)};

  auto prompt = tmpl.apply(messages, {}, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜User｜>first\nsecond"
            "<｜Assistant｜></think>");
}

TEST(DeepseekV4CppChatTemplate, ToolCallsBlockAndReasoningAreProjected) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  Message assistant("assistant", "");
  Message::ToolCall call;
  call.id = "call_001";
  call.type = "function";
  call.function.name = "get_weather";
  call.function.arguments = R"({"location":"Beijing"})";
  assistant.tool_calls = Message::ToolCallVec{call};
  assistant.reasoning_content = "Need weather data.";
  messages.push_back(assistant);

  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  kwargs["thinking"] = true;
  auto prompt = tmpl.apply(messages, {}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜DSML｜tool_calls>"), std::string::npos);
  EXPECT_NE(prompt->find("Need weather data.</think>"), std::string::npos);
}

TEST(DeepseekV4CppChatTemplate, ToolResultMergesViaToolCallId) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  Message assistant("assistant", "");
  Message::ToolCall call;
  call.id = "call_001";
  call.type = "function";
  call.function.name = "get_weather";
  call.function.arguments = R"({"location":"Beijing"})";
  assistant.tool_calls = Message::ToolCallVec{call};
  messages.push_back(assistant);

  // Non-empty tool_call_id drives tool-result merging.
  Message tool_msg("tool", R"({"temperature":22})");
  tool_msg.tool_call_id = "call_001";
  messages.push_back(tool_msg);

  auto prompt = tmpl.apply(messages, {}, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜User｜><tool_result>{\"temperature\":22}"
                         "</tool_result><｜Assistant｜></think>"),
            std::string::npos);
}

TEST(DeepseekV4CppChatTemplate, ToolsInjectionRendersV4Schema) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  ChatMessages messages;
  messages.emplace_back("user", "weather in beijing");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.description = "query weather";
  tool.function.parameters = nlohmann::json{
      {"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}};
  tools.push_back(tool);

  auto prompt = tmpl.apply(messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("### Available Tool Schemas"), std::string::npos);
  EXPECT_NE(prompt->find("get_weather"), std::string::npos);
}

TEST(DeepseekV4CppChatTemplate, ThinkingKwargTogglesThinkBlock) {
  DeepseekV4CppChatTemplate tmpl(make_v4_args());

  ChatMessages messages{Message("user", "Hello")};

  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  kwargs["thinking"] = true;
  auto thinking_on = tmpl.apply(messages, {}, kwargs);
  ASSERT_TRUE(thinking_on.has_value());
  EXPECT_EQ(*thinking_on,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>");

  // Default (no kwarg) closes thinking immediately.
  auto thinking_off =
      tmpl.apply(messages, {}, nlohmann::ordered_json::object());
  ASSERT_TRUE(thinking_off.has_value());
  EXPECT_EQ(*thinking_off,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>");
}

}  // namespace
}  // namespace xllm_service
