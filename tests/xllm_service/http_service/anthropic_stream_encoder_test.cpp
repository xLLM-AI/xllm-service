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

#include "http_service/anthropic_stream_encoder.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm_service {
namespace {

llm::RequestOutput make_output(const std::string& request_id) {
  llm::RequestOutput output;
  output.request_id = request_id;
  return output;
}

// Parses the `data:` payload of one SSE frame into JSON.
nlohmann::json sse_data_json(const std::string& sse) {
  const std::string prefix = "data: ";
  auto pos = sse.find(prefix);
  EXPECT_NE(pos, std::string::npos) << sse;
  pos += prefix.size();
  auto end = sse.find("\n\n", pos);
  EXPECT_NE(end, std::string::npos) << sse;
  return nlohmann::json::parse(sse.substr(pos, end - pos));
}

std::vector<nlohmann::json> to_jsons(const std::vector<std::string>& sse) {
  std::vector<nlohmann::json> jsons;
  jsons.reserve(sse.size());
  for (const auto& frame : sse) {
    jsons.push_back(sse_data_json(frame));
  }
  return jsons;
}

TEST(AnthropicStreamEncoderTest, PlainTextSingleChunkOpensMessageAndTextBlock) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> sse;
  auto result = encoder.on_text(output, "Hel", &sse);
  ASSERT_TRUE(result.ok) << result.error;

  auto events = to_jsons(sse);
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0]["type"], "message_start");
  EXPECT_EQ(events[0]["message"]["id"], "anthropiccmpl-test");
  EXPECT_EQ(events[0]["message"]["model"], "test-model");
  EXPECT_EQ(events[1]["type"], "content_block_start");
  EXPECT_EQ(events[1]["index"], 0);
  EXPECT_EQ(events[1]["content_block"]["type"], "text");
  EXPECT_EQ(events[2]["type"], "content_block_delta");
  EXPECT_EQ(events[2]["index"], 0);
  EXPECT_EQ(events[2]["delta"]["type"], "text_delta");
  EXPECT_EQ(events[2]["delta"]["text"], "Hel");
}

TEST(AnthropicStreamEncoderTest, MessageStartFrameNormalizesNullStopFields) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> sse;
  ASSERT_TRUE(encoder.on_text(output, "Hi", &sse).ok);
  EXPECT_NE(sse[0].find("event: message_start\n"), std::string::npos);
  auto message_start = sse_data_json(sse[0]);
  ASSERT_TRUE(message_start["message"].contains("stop_reason"));
  EXPECT_TRUE(message_start["message"]["stop_reason"].is_null());
  ASSERT_TRUE(message_start["message"].contains("stop_sequence"));
  EXPECT_TRUE(message_start["message"]["stop_sequence"].is_null());
  EXPECT_EQ(message_start["message"]["usage"]["input_tokens"], 0);
  EXPECT_EQ(message_start["message"]["usage"]["output_tokens"], 0);
}

TEST(AnthropicStreamEncoderTest, ReasoningThenFinishEmitsSignatureBeforeStop) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> reasoning;
  ASSERT_TRUE(encoder.on_reasoning(output, "reason", &reasoning).ok);

  llm::RequestOutput final = make_output("anthropiccmpl-test");
  final.finished = true;
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.finish_reason = "stop";
  final.outputs.push_back(std::move(seq));

  std::vector<std::string> done;
  ASSERT_TRUE(encoder.finish(final, &done).ok);
  auto events = to_jsons(done);
  // An unclosed thinking block is sealed with signature_delta then stop.
  ASSERT_GE(events.size(), 4);
  EXPECT_EQ(events[0]["type"], "content_block_delta");
  EXPECT_EQ(events[0]["index"], 0);
  EXPECT_EQ(events[0]["delta"]["type"], "signature_delta");
  EXPECT_FALSE(events[0]["delta"]["signature"].get<std::string>().empty());
  EXPECT_EQ(events[1]["type"], "content_block_stop");
  EXPECT_EQ(events[1]["index"], 0);
  EXPECT_EQ(events[2]["type"], "message_delta");
  EXPECT_EQ(events[3]["type"], "message_stop");
}

TEST(AnthropicStreamEncoderTest, SecondTextChunkEmitsOnlyDelta) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> first;
  ASSERT_TRUE(encoder.on_text(output, "Hel", &first).ok);

  std::vector<std::string> second;
  ASSERT_TRUE(encoder.on_text(output, "lo", &second).ok);

  auto events = to_jsons(second);
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0]["type"], "content_block_delta");
  EXPECT_EQ(events[0]["index"], 0);
  EXPECT_EQ(events[0]["delta"]["text"], "lo");
}

TEST(AnthropicStreamEncoderTest, ReasoningThenTextClosesThinkingWithSignature) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> reasoning;
  ASSERT_TRUE(encoder.on_reasoning(output, "reason", &reasoning).ok);
  auto reasoning_events = to_jsons(reasoning);
  ASSERT_EQ(reasoning_events.size(), 3);
  EXPECT_EQ(reasoning_events[0]["type"], "message_start");
  EXPECT_EQ(reasoning_events[1]["type"], "content_block_start");
  EXPECT_EQ(reasoning_events[1]["index"], 0);
  EXPECT_EQ(reasoning_events[1]["content_block"]["type"], "thinking");
  EXPECT_EQ(reasoning_events[2]["type"], "content_block_delta");
  EXPECT_EQ(reasoning_events[2]["delta"]["type"], "thinking_delta");
  EXPECT_EQ(reasoning_events[2]["delta"]["thinking"], "reason");

  // Switching to text must close the thinking block with a signature_delta
  // then a content_block_stop, and open a new text block at the next index.
  std::vector<std::string> text;
  ASSERT_TRUE(encoder.on_text(output, "answer", &text).ok);
  auto text_events = to_jsons(text);
  ASSERT_EQ(text_events.size(), 4);
  EXPECT_EQ(text_events[0]["type"], "content_block_delta");
  EXPECT_EQ(text_events[0]["index"], 0);
  EXPECT_EQ(text_events[0]["delta"]["type"], "signature_delta");
  EXPECT_FALSE(text_events[0]["delta"]["signature"].get<std::string>().empty());
  EXPECT_EQ(text_events[1]["type"], "content_block_stop");
  EXPECT_EQ(text_events[1]["index"], 0);
  EXPECT_EQ(text_events[2]["type"], "content_block_start");
  EXPECT_EQ(text_events[2]["index"], 1);
  EXPECT_EQ(text_events[2]["content_block"]["type"], "text");
  EXPECT_EQ(text_events[3]["type"], "content_block_delta");
  EXPECT_EQ(text_events[3]["index"], 1);
  EXPECT_EQ(text_events[3]["delta"]["text"], "answer");
}

TEST(AnthropicStreamEncoderTest, TextThenToolSwitchesBlockAndIncrementsIndex) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> text;
  ASSERT_TRUE(encoder.on_text(output, "Let me check ", &text).ok);

  std::vector<std::string> tool;
  auto result =
      encoder.on_tool(output, "call_1", "get_weather", R"({"city")", &tool);
  ASSERT_TRUE(result.ok) << result.error;
  auto events = to_jsons(tool);
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0]["type"], "content_block_stop");
  EXPECT_EQ(events[0]["index"], 0);
  EXPECT_EQ(events[1]["type"], "content_block_start");
  EXPECT_EQ(events[1]["index"], 1);
  EXPECT_EQ(events[1]["content_block"]["type"], "tool_use");
  EXPECT_EQ(events[1]["content_block"]["id"], "call_1");
  EXPECT_EQ(events[1]["content_block"]["name"], "get_weather");
  EXPECT_EQ(events[2]["type"], "content_block_delta");
  EXPECT_EQ(events[2]["index"], 1);
  EXPECT_EQ(events[2]["delta"]["type"], "input_json_delta");
  EXPECT_EQ(events[2]["delta"]["partial_json"], R"({"city")");
}

TEST(AnthropicStreamEncoderTest, ToolArgumentDeltaContinuesActiveBlock) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> open;
  ASSERT_TRUE(
      encoder.on_tool(output, "call_1", "get_weather", R"({"city")", &open).ok);

  // A follow-up delta with no id/name appends arguments to the open block.
  std::vector<std::string> more;
  auto result = encoder.on_tool(output, "", "", R"(: "Beijing"})", &more);
  ASSERT_TRUE(result.ok) << result.error;
  auto events = to_jsons(more);
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0]["type"], "content_block_delta");
  EXPECT_EQ(events[0]["index"], 0);
  EXPECT_EQ(events[0]["delta"]["partial_json"], R"(: "Beijing"})");
}

TEST(AnthropicStreamEncoderTest, ToolArgumentDeltaWithoutActiveBlockRejected) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> sse;
  auto result = encoder.on_tool(output, "", "", R"({"command":"ls"})", &sse);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find("Anthropic tool_use id and name are required"),
            std::string::npos);
  EXPECT_TRUE(sse.empty());
}

TEST(AnthropicStreamEncoderTest, FinishClosesBlockAndEmitsMessageDeltaStop) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");

  std::vector<std::string> text;
  ASSERT_TRUE(encoder.on_text(output, "answer", &text).ok);

  llm::RequestOutput final = make_output("anthropiccmpl-test");
  final.finished = true;
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.finish_reason = "stop";
  final.outputs.push_back(std::move(seq));
  llm::Usage usage;
  usage.num_prompt_tokens = 3;
  usage.num_generated_tokens = 5;
  final.usage = usage;

  std::vector<std::string> done;
  auto result = encoder.finish(final, &done);
  ASSERT_TRUE(result.ok) << result.error;
  auto events = to_jsons(done);
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0]["type"], "content_block_stop");
  EXPECT_EQ(events[0]["index"], 0);
  EXPECT_EQ(events[1]["type"], "message_delta");
  EXPECT_EQ(events[1]["delta"]["stop_reason"], "end_turn");
  EXPECT_EQ(events[1]["usage"]["input_tokens"], 3);
  EXPECT_EQ(events[1]["usage"]["output_tokens"], 5);
  EXPECT_EQ(events[2]["type"], "message_stop");
}

TEST(AnthropicStreamEncoderTest, FinishAfterToolUsesToolStopReason) {
  AnthropicStreamEncoder encoder("test-model");
  auto output = make_output("anthropiccmpl-test");
  std::vector<std::string> tool;
  ASSERT_TRUE(
      encoder.on_tool(output, "call_1", "get_weather", R"({"city":"X"})", &tool)
          .ok);

  llm::RequestOutput final = make_output("anthropiccmpl-test");
  final.finished = true;
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.finish_reason = "stop";
  final.outputs.push_back(std::move(seq));

  std::vector<std::string> done;
  ASSERT_TRUE(encoder.finish(final, &done).ok);
  auto events = to_jsons(done);
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0]["type"], "content_block_stop");
  EXPECT_EQ(events[1]["type"], "message_delta");
  EXPECT_EQ(events[1]["delta"]["stop_reason"], "tool_use");
  EXPECT_EQ(events[2]["type"], "message_stop");
}

}  // namespace
}  // namespace xllm_service
