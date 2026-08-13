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

#include "scheduler/xllm_chat_parse_bridge.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "xllm/xllm/api_service/stream_output_parser.h"

namespace xllm_service {
namespace {

std::vector<JsonTool> make_tools() {
  nlohmann::json params = {{"type", "object"},
                           {"properties",
                            {{"content", {{"type", "string"}}},
                             {"file_path", {{"type", "string"}}}}},
                           {"required", {"content", "file_path"}}};
  return {JsonTool("function",
                   JsonFunction("Write", "Write a local file", params))};
}

std::vector<JsonTool> make_bash_tools() {
  nlohmann::json params = {{"type", "object"},
                           {"properties",
                            {{"command", {{"type", "string"}}},
                             {"description", {{"type", "string"}}}}},
                           {"required", {"command"}}};
  return {JsonTool("function",
                   JsonFunction("Bash", "Run a shell command", params))};
}

std::string make_deepseek_v4_tool_call() {
  return "<｜DSML｜tool_calls>"
         "<｜DSML｜invoke name=\"Write\">"
         "<｜DSML｜parameter name=\"content\" string=\"true\">"
         "hello</｜DSML｜parameter>"
         "<｜DSML｜parameter name=\"file_path\" string=\"true\">"
         "/tmp/two_sum.py</｜DSML｜parameter>"
         "</｜DSML｜invoke>"
         "</｜DSML｜tool_calls>";
}

void expect_write_tool_call(const std::string& arguments) {
  auto args = nlohmann::json::parse(arguments);
  EXPECT_EQ(args["content"], "hello");
  EXPECT_EQ(args["file_path"], "/tmp/two_sum.py");
}

TEST(XllmChatParseBridgeTest, ResolvesDeepSeekV4AutoParsersFromModelConfig) {
  for (const std::string model_type : {"deepseek_v4", "deepseek_v4_mtp"}) {
    auto formats =
        resolve_chat_parser_formats_with_xllm(model_type, "auto", "auto");
    EXPECT_EQ(formats.tool_call_parser, "deepseekv4");
    EXPECT_EQ(formats.reasoning_parser, "deepseek-v4");
  }
}

TEST(XllmChatParseBridgeTest, DefaultsDeepSeekV4ParsersWhenFlagsAreUnset) {
  for (const std::string model_type : {"deepseek_v4", "deepseek_v4_mtp"}) {
    auto formats = resolve_chat_parser_formats_with_xllm(model_type);
    EXPECT_EQ(formats.tool_call_parser, "deepseekv4");
    EXPECT_EQ(formats.reasoning_parser, "deepseek-v4");
  }
}

TEST(XllmChatParseBridgeTest, KeepsUnsetParsersDisabledForOtherModels) {
  auto formats = resolve_chat_parser_formats_with_xllm("qwen3");
  EXPECT_TRUE(formats.tool_call_parser.empty());
  EXPECT_TRUE(formats.reasoning_parser.empty());
}

TEST(XllmChatParseBridgeTest, ResolvesGlm5AutoParsersFromModelConfig) {
  for (const std::string model_type : {"glm_moe_dsa", "glm_moe_dsa_mtp"}) {
    auto formats =
        resolve_chat_parser_formats_with_xllm(model_type, "auto", "auto");
    EXPECT_EQ(formats.tool_call_parser, "glm5");
    EXPECT_EQ(formats.reasoning_parser, "glm5");
  }
}

TEST(XllmChatParseBridgeTest, KeepsExplicitDeepSeekV4ToolParser) {
  auto formats =
      resolve_chat_parser_formats_with_xllm("deepseek_v4", "deepseekv4", "");
  EXPECT_EQ(formats.tool_call_parser, "deepseekv4");

  auto result = parse_chat_output_with_xllm(make_deepseek_v4_tool_call(),
                                            make_tools(),
                                            "client-model-alias",
                                            "stop",
                                            formats.tool_call_parser);
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "Write");
  expect_write_tool_call(result.tool_calls->Get(0).function().arguments());
}

TEST(XllmChatParseBridgeTest,
     ParsesDeepSeekV4ToolCallUsingConfigFormatsWithClientAlias) {
  auto formats =
      resolve_chat_parser_formats_with_xllm("deepseek_v4", "auto", "auto");

  auto result = parse_chat_output_with_xllm(make_deepseek_v4_tool_call(),
                                            make_tools(),
                                            "client-model-alias",
                                            "stop",
                                            formats.tool_call_parser,
                                            formats.reasoning_parser,
                                            true);

  EXPECT_TRUE(result.text.empty());
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "Write");
  expect_write_tool_call(result.tool_calls->Get(0).function().arguments());
  EXPECT_EQ(result.finish_reason, "tool_calls");
}

TEST(XllmChatParseBridgeTest,
     ParsesDeepSeekV4OutputUsingDefaultFormatsWithClientAlias) {
  auto formats = resolve_chat_parser_formats_with_xllm("deepseek_v4");
  auto result = parse_chat_output_with_xllm(
      "need a file</think>" + make_deepseek_v4_tool_call(),
      make_tools(),
      "client-model-alias",
      "stop",
      formats.tool_call_parser,
      formats.reasoning_parser,
      true);

  EXPECT_TRUE(result.text.empty());
  ASSERT_TRUE(result.reasoning_content.has_value());
  EXPECT_EQ(result.reasoning_content.value(), "need a file");
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "Write");
  expect_write_tool_call(result.tool_calls->Get(0).function().arguments());
  EXPECT_EQ(result.finish_reason, "tool_calls");
}

TEST(XllmChatParseBridgeTest,
     StreamsDeepSeekV4ToolCallUsingConfigFormatsWithClientAlias) {
  auto formats =
      resolve_chat_parser_formats_with_xllm("deepseek_v4", "auto", "auto");
  auto stream_parser =
      create_stream_output_parser_with_xllm(make_tools(),
                                            "client-model-alias",
                                            formats.tool_call_parser,
                                            formats.reasoning_parser,
                                            true);
  auto* parser = stream_parser->get_tool_call_parser(0);
  ASSERT_NE(parser, nullptr);

  auto result = parser->parse_streaming_increment(make_deepseek_v4_tool_call());

  EXPECT_TRUE(result.normal_text.empty());
  ASSERT_FALSE(result.calls.empty());
  ASSERT_TRUE(result.calls.front().name.has_value());
  EXPECT_EQ(result.calls.front().name.value(), "Write");
  expect_write_tool_call(result.calls.back().parameters);
}

TEST(XllmChatParseBridgeTest,
     ParsesDeepSeekV4ReasoningUsingConfigFormatsWithClientAlias) {
  auto formats =
      resolve_chat_parser_formats_with_xllm("deepseek_v4", "auto", "auto");

  auto result = parse_chat_output_with_xllm("need data</think>final answer",
                                            {},
                                            "client-model-alias",
                                            "stop",
                                            formats.tool_call_parser,
                                            formats.reasoning_parser,
                                            true);

  EXPECT_EQ(result.text, "final answer");
  ASSERT_TRUE(result.reasoning_content.has_value());
  EXPECT_EQ(result.reasoning_content.value(), "need data");
}

TEST(XllmChatParseBridgeTest,
     StreamsDeepSeekV4ReasoningUsingConfigFormatsWithClientAlias) {
  auto formats =
      resolve_chat_parser_formats_with_xllm("deepseek_v4", "auto", "auto");
  auto stream_parser =
      create_stream_output_parser_with_xllm({},
                                            "client-model-alias",
                                            formats.tool_call_parser,
                                            formats.reasoning_parser,
                                            true);
  auto* parser = stream_parser->get_reasoning_parser(0);
  ASSERT_NE(parser, nullptr);

  auto reasoning = parser->parse_stream_chunk("need data");
  ASSERT_TRUE(reasoning.reasoning_text.has_value());
  EXPECT_EQ(reasoning.reasoning_text.value(), "need data");
  EXPECT_FALSE(reasoning.normal_text.has_value());

  auto answer = parser->parse_stream_chunk("</think>final answer");
  ASSERT_TRUE(answer.normal_text.has_value());
  EXPECT_EQ(answer.normal_text.value(), "final answer");
}

TEST(XllmChatParseBridgeTest, ReasoningEffortNoneDisablesForcedReasoning) {
  nlohmann::json kwargs = {{"reasoning_effort", "none"}, {"thinking", true}};
  EXPECT_FALSE(get_enable_thinking_from_request(kwargs, "deepseek-v4"));

  auto result = parse_chat_output_with_xllm("final answer",
                                            {},
                                            "client-model-alias",
                                            "stop",
                                            "",
                                            "deepseek-v4",
                                            false);
  EXPECT_EQ(result.text, "final answer");
  EXPECT_FALSE(result.reasoning_content.has_value());
}

TEST(XllmChatParseBridgeTest, StreamsNormalContentWhenReasoningEffortIsNone) {
  nlohmann::json kwargs = {{"reasoning_effort", "none"}};
  const bool force_reasoning =
      get_enable_thinking_from_request(kwargs, "deepseek-v4");
  auto stream_parser = create_stream_output_parser_with_xllm(
      {}, "client-model-alias", "", "deepseek-v4", force_reasoning);
  auto* parser = stream_parser->get_reasoning_parser(0);
  ASSERT_NE(parser, nullptr);

  auto result = parser->parse_stream_chunk("final answer");

  ASSERT_TRUE(result.normal_text.has_value());
  EXPECT_EQ(result.normal_text.value(), "final answer");
  EXPECT_FALSE(result.reasoning_text.has_value());
}

TEST(XllmChatParseBridgeTest, ParsesGlmToolCallWithReasoningParserConfigured) {
  std::string text =
      "<tool_call>Write<arg_key>content</arg_key><arg_value>hello</arg_value>"
      "<arg_key>file_path</arg_key><arg_value>/tmp/two_sum.py</arg_value>"
      "</tool_call>";

  auto result = parse_chat_output_with_xllm(
      text, make_tools(), "glm5", "stop", "glm47", "glm47", true);

  EXPECT_TRUE(result.text.empty());
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  const auto& tool_call = result.tool_calls->Get(0);
  EXPECT_EQ(tool_call.function().name(), "Write");
  auto args = nlohmann::json::parse(tool_call.function().arguments());
  EXPECT_EQ(args["content"], "hello");
  EXPECT_EQ(args["file_path"], "/tmp/two_sum.py");
  EXPECT_EQ(result.finish_reason, "tool_calls");
  EXPECT_FALSE(result.reasoning_content.has_value());
}

TEST(XllmChatParseBridgeTest, KeepsExplicitThinkingBeforeToolCall) {
  std::string text =
      "<think>need file</think>"
      "<tool_call>Write<arg_key>content</arg_key><arg_value>hello</arg_value>"
      "<arg_key>file_path</arg_key><arg_value>/tmp/two_sum.py</arg_value>"
      "</tool_call>";

  auto result = parse_chat_output_with_xllm(
      text, make_tools(), "glm5", "stop", "glm47", "glm47", true);

  ASSERT_TRUE(result.reasoning_content.has_value());
  EXPECT_EQ(result.reasoning_content.value(), "need file");
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "Write");
  EXPECT_TRUE(result.text.empty());
}

TEST(XllmChatParseBridgeTest, KeepsImplicitThinkingBeforeToolCall) {
  std::string text =
      "need file"
      "<tool_call>Write<arg_key>content</arg_key><arg_value>hello</arg_value>"
      "<arg_key>file_path</arg_key><arg_value>/tmp/two_sum.py</arg_value>"
      "</tool_call></think>";

  auto result = parse_chat_output_with_xllm(
      text, make_tools(), "glm5", "stop", "glm47", "glm47", true);

  ASSERT_TRUE(result.reasoning_content.has_value());
  EXPECT_EQ(result.reasoning_content.value(), "need file");
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "Write");
  EXPECT_TRUE(result.text.empty());
}

TEST(XllmChatParseBridgeTest, StreamsAdjacentGlmToolCallsSplitAcrossChunks) {
  auto stream_parser = create_stream_output_parser_with_xllm(
      make_bash_tools(), "GLM-5.1-W4A8", "glm47", "", false);
  auto* parser = stream_parser->get_tool_call_parser(0);
  ASSERT_NE(parser, nullptr);

  auto first = parser->parse_streaming_increment("<tool_call>Bash<arg_key>");
  EXPECT_TRUE(first.normal_text.empty());
  ASSERT_EQ(first.calls.size(), 1);
  ASSERT_TRUE(first.calls[0].name.has_value());
  EXPECT_EQ(first.calls[0].name.value(), "Bash");
  EXPECT_TRUE(first.calls[0].parameters.empty());

  auto second = parser->parse_streaming_increment(
      "command</arg_key><arg_value>rm hello_world.py</arg_value>"
      "<arg_key>description</arg_key><arg_value>Delete file</arg_value>"
      "</tool_call><tool_call>Bash<arg_key>command");
  EXPECT_TRUE(second.normal_text.empty());
  ASSERT_FALSE(second.calls.empty());
  for (const auto& call : second.calls) {
    EXPECT_FALSE(call.name.has_value());
  }

  auto third = parser->parse_streaming_increment(
      "</arg_key><arg_value>find .</arg_value>");
  EXPECT_TRUE(third.normal_text.empty());
  ASSERT_EQ(third.calls.size(), 1);
  ASSERT_TRUE(third.calls[0].name.has_value());
  EXPECT_EQ(third.calls[0].name.value(), "Bash");
}

TEST(XllmChatParseBridgeTest, StreamsGlmToolNameSplitBeforeArgKey) {
  auto stream_parser = create_stream_output_parser_with_xllm(
      make_bash_tools(), "GLM-5.1-W4A8", "glm47", "", false);
  auto* parser = stream_parser->get_tool_call_parser(0);
  ASSERT_NE(parser, nullptr);

  auto first = parser->parse_streaming_increment("<tool_call>Bash");
  EXPECT_TRUE(first.normal_text.empty());
  EXPECT_TRUE(first.calls.empty());

  auto second = parser->parse_streaming_increment(
      "<arg_key>command</arg_key><arg_value>"
      "rm /workspace/xllm-codex/pd-testspace/hello_world.py</arg_value>");
  EXPECT_TRUE(second.normal_text.empty());
  ASSERT_EQ(second.calls.size(), 1);
  ASSERT_TRUE(second.calls[0].name.has_value());
  EXPECT_EQ(second.calls[0].name.value(), "Bash");

  auto third = parser->parse_streaming_increment(
      "<arg_key>description</arg_key><arg_value>Delete hello_world.py"
      "</arg_value></tool_call>");
  EXPECT_TRUE(third.normal_text.empty());
  ASSERT_FALSE(third.calls.empty());
  for (const auto& call : third.calls) {
    EXPECT_FALSE(call.name.has_value());
  }
}

}  // namespace
}  // namespace xllm_service
