/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "http_service/anthropic_adapter.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <variant>

namespace xllm_service {
namespace {

xllm::proto::AnthropicMessagesRequest parse_request(const std::string& json) {
  xllm::proto::AnthropicMessagesRequest request;
  auto result = parse_anthropic_json(json, &request);
  EXPECT_TRUE(result.ok) << result.error;
  return request;
}

const std::string& text_content(const Message& message) {
  return std::get<std::string>(message.content);
}

AnthropicAdaptResult adapt_request(
    const xllm::proto::AnthropicMessagesRequest& request,
    xllm::proto::ChatRequest* chat_request,
    ChatMessages* messages) {
  return fill_chat_req(request, chat_request, messages);
}

void expect_reject(const std::string& json, const std::string& expected_error) {
  auto request = parse_request(json);
  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find(expected_error), std::string::npos)
      << result.error;
  EXPECT_EQ(chat_request.messages_size(), 0);
  EXPECT_TRUE(messages.empty());
}

std::string chat_tool_choice(const std::string& tool_choice_json) {
  auto request = parse_request(tool_choice_json);
  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  EXPECT_TRUE(result.ok) << result.error;
  EXPECT_TRUE(chat_request.has_tool_choice());
  return chat_request.tool_choice();
}

TEST(AnthropicAdapterTest, MapsStringSystemMessagesAndParams) {
  auto request = parse_request(R"({
    "model": "test-model",
    "system": "You are concise.",
    "max_tokens": 32,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "stop_sequences": ["</stop>"],
    "ignore_eos": true,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = fill_chat_req(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;

  EXPECT_EQ(chat_request.model(), "test-model");
  ASSERT_TRUE(chat_request.has_max_tokens());
  EXPECT_EQ(chat_request.max_tokens(), 32);
  ASSERT_TRUE(chat_request.has_temperature());
  EXPECT_FLOAT_EQ(chat_request.temperature(), 0.2f);
  ASSERT_TRUE(chat_request.has_top_p());
  EXPECT_FLOAT_EQ(chat_request.top_p(), 0.9f);
  ASSERT_TRUE(chat_request.has_top_k());
  EXPECT_EQ(chat_request.top_k(), 20);
  ASSERT_TRUE(chat_request.has_ignore_eos());
  EXPECT_TRUE(chat_request.ignore_eos());
  ASSERT_EQ(chat_request.stop_size(), 1);
  EXPECT_EQ(chat_request.stop(0), "</stop>");

  ASSERT_EQ(chat_request.messages_size(), 2);
  EXPECT_EQ(chat_request.messages(0).role(), "system");
  EXPECT_EQ(chat_request.messages(0).content(), "You are concise.");
  EXPECT_EQ(chat_request.messages(1).role(), "user");
  EXPECT_EQ(chat_request.messages(1).content(), "hello");

  ASSERT_EQ(messages.size(), 2);
  EXPECT_EQ(messages[0].role, "system");
  EXPECT_EQ(text_content(messages[0]), "You are concise.");
  EXPECT_EQ(messages[1].role, "user");
  EXPECT_EQ(text_content(messages[1]), "hello");
}

TEST(AnthropicAdapterTest, MapsTextBlocksForSystemAndMessages) {
  auto request = parse_request(R"({
    "model": "test-model",
    "system": [
      {"type": "text", "text": "rule "},
      {"type": "text", "text": "style"}
    ],
    "max_tokens": 8,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "hello"},
          {"type": "text", "text": "world"}
        ]
      }
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = fill_chat_req(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;

  ASSERT_EQ(chat_request.messages_size(), 2);
  EXPECT_EQ(chat_request.messages(0).content(), "rule style");
  EXPECT_EQ(chat_request.messages(1).content(), "hello\nworld");

  ASSERT_EQ(messages.size(), 2);
  EXPECT_EQ(text_content(messages[0]), "rule style");
  ASSERT_TRUE(
      std::holds_alternative<Message::MMContentVec>(messages[1].content));
  const auto& content = std::get<Message::MMContentVec>(messages[1].content);
  ASSERT_EQ(content.size(), 2);
  EXPECT_EQ(content[0].type, "text");
  EXPECT_EQ(content[0].text, "hello");
  EXPECT_EQ(content[1].type, "text");
  EXPECT_EQ(content[1].text, "world");
}

TEST(AnthropicAdapterTest, PreservesAssistantRoleForTextBlocks) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "messages": [
      {"role": "user", "content": "question"},
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "partial"},
          {"type": "text", "text": "answer"}
        ]
      }
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;

  ASSERT_EQ(chat_request.messages_size(), 2);
  EXPECT_EQ(chat_request.messages(1).role(), "assistant");
  EXPECT_EQ(chat_request.messages(1).content(), "partial\nanswer");
  ASSERT_EQ(messages.size(), 2);
  EXPECT_EQ(messages[1].role, "assistant");
}

TEST(AnthropicAdapterTest, IgnoresUnknownJsonFields) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "metadata": {"trace": "x"},
    "unknown_top_level": true,
    "messages": [
      {
        "role": "user",
        "content": "hello",
        "unknown_message_field": {"ignored": true}
      }
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(chat_request.messages_size(), 1);
  EXPECT_EQ(chat_request.messages(0).content(), "hello");
}

TEST(AnthropicAdapterTest, RejectsEmptyMessages) {
  expect_reject(R"({
    "model": "test-model",
    "system": "system only",
    "max_tokens": 8,
    "messages": []
  })",
                "Messages is empty");
}

TEST(AnthropicAdapterTest, RejectsSystemImageBlocks) {
  expect_reject(R"({
    "model": "test-model",
    "system": [
      {"type": "image", "source": {"type": "base64", "data": "x"}}
    ],
    "max_tokens": 8,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  })",
                "Unsupported Anthropic content block type: image");
}

TEST(AnthropicAdapterTest, RejectsNonTextBlocks) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "source": {"type": "base64", "data": "x"}}
        ]
      }
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = fill_chat_req(request, &chat_request, &messages);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find("Unsupported Anthropic content block type"),
            std::string::npos);
}

TEST(AnthropicAdapterTest, RejectsUnknownMessageBlocks) {
  expect_reject(R"({
    "model": "test-model",
    "max_tokens": 8,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "audio", "audio": {"url": "x"}}
        ]
      }
    ]
  })",
                "Unsupported Anthropic content block type: audio");
}

TEST(AnthropicAdapterTest, AllowsTextStreamingRequest) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "stream": true,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(chat_request.has_stream());
  EXPECT_TRUE(chat_request.stream());
}

TEST(AnthropicAdapterTest, AllowsStreamingToolsRequest) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "stream": true,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(chat_request.has_stream());
  EXPECT_TRUE(chat_request.stream());
  ASSERT_EQ(chat_request.tools_size(), 1);
  EXPECT_EQ(chat_request.tool_choice(), "auto");
}

TEST(AnthropicAdapterTest, MapsToolsAndDefaultToolChoices) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    ],
    "messages": [
      {"role": "user", "content": "weather"}
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;

  ASSERT_EQ(chat_request.tools_size(), 1);
  const auto& tool = chat_request.tools(0);
  EXPECT_EQ(tool.type(), "function");
  EXPECT_EQ(tool.function().name(), "get_weather");
  EXPECT_EQ(tool.function().description(), "Get weather");
  ASSERT_TRUE(tool.function().has_parameters());
  EXPECT_EQ(tool.function()
                .parameters()
                .fields()
                .at("properties")
                .struct_value()
                .fields()
                .at("city")
                .struct_value()
                .fields()
                .at("type")
                .string_value(),
            "string");
  ASSERT_TRUE(chat_request.has_tool_choice());
  EXPECT_EQ(chat_request.tool_choice(), "auto");
}

TEST(AnthropicAdapterTest, MapsToolChoiceRules) {
  const std::string base = R"({
    "model": "test-model",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hello"}])";

  EXPECT_EQ(chat_tool_choice(base + "}"), "none");
  EXPECT_EQ(chat_tool_choice(base + R"(,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
  })"),
            "auto");
  EXPECT_EQ(chat_tool_choice(base + R"(,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "auto"}
  })"),
            "auto");
  EXPECT_EQ(chat_tool_choice(base + R"(,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "any"}
  })"),
            "required");
  auto specific = nlohmann::json::parse(chat_tool_choice(base + R"(,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "tool", "name": "lookup"}
  })"));
  EXPECT_EQ(specific["type"], "function");
  EXPECT_EQ(specific["function"]["name"], "lookup");
  EXPECT_EQ(chat_tool_choice(base + R"(,
    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "mystery"}
  })"),
            "auto");
}

TEST(AnthropicAdapterTest, MapsToolUseAndToolResultHistory) {
  auto request = parse_request(R"({
    "model": "test-model",
    "max_tokens": 8,
    "messages": [
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "I will call a tool."},
          {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Beijing"}
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "sunny"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "fallback"
          }
        ]
      }
    ]
  })");

  xllm::proto::ChatRequest chat_request;
  ChatMessages messages;
  auto result = adapt_request(request, &chat_request, &messages);
  ASSERT_TRUE(result.ok) << result.error;

  ASSERT_EQ(chat_request.messages_size(), 3);
  const auto& assistant = chat_request.messages(0);
  EXPECT_EQ(assistant.role(), "assistant");
  EXPECT_EQ(assistant.content(), "I will call a tool.");
  ASSERT_EQ(assistant.tool_calls_size(), 1);
  EXPECT_EQ(assistant.tool_calls(0).id(), "toolu_1");
  EXPECT_EQ(assistant.tool_calls(0).type(), "function");
  EXPECT_EQ(assistant.tool_calls(0).function().name(), "get_weather");
  EXPECT_EQ(assistant.tool_calls(0).function().arguments(),
            R"({"city":"Beijing"})");

  const auto& tool_result = chat_request.messages(1);
  EXPECT_EQ(tool_result.role(), "tool");
  EXPECT_EQ(tool_result.tool_call_id(), "toolu_1");
  EXPECT_EQ(tool_result.content(), "sunny");

  const auto& assistant_result = chat_request.messages(2);
  EXPECT_EQ(assistant_result.role(), "assistant");
  EXPECT_EQ(assistant_result.content(), "Tool result: fallback");

  ASSERT_EQ(messages.size(), 3);
  EXPECT_EQ(messages[0].role, "assistant");
  ASSERT_TRUE(messages[0].tool_calls.has_value());
  ASSERT_EQ(messages[0].tool_calls->size(), 1);
  EXPECT_EQ((*messages[0].tool_calls)[0].id, "toolu_1");
  EXPECT_EQ(messages[1].role, "tool");
  EXPECT_EQ(messages[1].tool_call_id, "toolu_1");
}

TEST(AnthropicAdapterTest, BuildsNonStreamAnthropicJson) {
  llm::RequestOutput output;
  output.request_id = "anthropiccmpl-test";
  output.finished = true;
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.text = "answer";
  seq.finish_reason = "stop";
  output.outputs.push_back(std::move(seq));
  llm::Usage usage;
  usage.num_prompt_tokens = 3;
  usage.num_generated_tokens = 4;
  usage.num_total_tokens = 7;
  output.usage = usage;

  xllm::proto::AnthropicMessagesResponse response;
  auto result = fill_anthropic_resp("test-model", output, &response);
  ASSERT_TRUE(result.ok) << result.error;

  std::string json_str;
  std::string error;
  ASSERT_TRUE(anthropic_json(response, &json_str, &error)) << error;
  auto json = nlohmann::json::parse(json_str);

  EXPECT_EQ(json["id"], "anthropiccmpl-test");
  EXPECT_EQ(json["type"], "message");
  EXPECT_EQ(json["role"], "assistant");
  EXPECT_EQ(json["model"], "test-model");
  EXPECT_EQ(json["stop_reason"], "end_turn");
  ASSERT_EQ(json["content"].size(), 1);
  EXPECT_EQ(json["content"][0]["type"], "text");
  EXPECT_EQ(json["content"][0]["text"], "answer");
  EXPECT_EQ(json["usage"]["input_tokens"], 3);
  EXPECT_EQ(json["usage"]["output_tokens"], 4);
  EXPECT_FALSE(json["usage"].contains("total_tokens"));
}

TEST(AnthropicAdapterTest, BuildsToolUseAnthropicJson) {
  llm::RequestOutput output;
  output.request_id = "anthropiccmpl-test";
  output.finished = true;
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.text = "";
  seq.finish_reason = "tool_calls";
  output.outputs.push_back(std::move(seq));

  google::protobuf::RepeatedPtrField<xllm::proto::ToolCall> tool_calls;
  auto* tool_call = tool_calls.Add();
  tool_call->set_id("call_1");
  tool_call->set_type("function");
  tool_call->mutable_function()->set_name("get_weather");
  tool_call->mutable_function()->set_arguments(R"({"city":"Beijing"})");

  xllm::proto::AnthropicMessagesResponse response;
  auto result =
      fill_anthropic_resp("test-model", output, &response, &tool_calls);
  ASSERT_TRUE(result.ok) << result.error;

  std::string json_str;
  std::string error;
  ASSERT_TRUE(anthropic_json(response, &json_str, &error)) << error;
  auto json = nlohmann::json::parse(json_str);

  EXPECT_EQ(json["stop_reason"], "tool_use");
  ASSERT_EQ(json["content"].size(), 1);
  EXPECT_EQ(json["content"][0]["type"], "tool_use");
  EXPECT_EQ(json["content"][0]["id"], "call_1");
  EXPECT_EQ(json["content"][0]["name"], "get_weather");
  EXPECT_EQ(json["content"][0]["input"]["city"], "Beijing");
}

TEST(AnthropicAdapterTest, BuildsTextStreamEvents) {
  AnthropicStreamState state;
  std::vector<xllm::proto::AnthropicStreamEvent> events;

  llm::RequestOutput first;
  first.request_id = "anthropiccmpl-test";
  llm::SequenceOutput first_seq;
  first_seq.index = 0;
  first_seq.text = "Hel";
  first.outputs.push_back(std::move(first_seq));

  auto result =
      fill_anthropic_stream_events("test-model", first, state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type(), "message_start");
  EXPECT_EQ(events[0].message().id(), "anthropiccmpl-test");
  EXPECT_EQ(events[0].message().usage().input_tokens(), 0);
  EXPECT_EQ(events[0].message().usage().output_tokens(), 0);
  EXPECT_EQ(events[1].type(), "content_block_start");
  EXPECT_EQ(events[1].index(), 0);
  EXPECT_EQ(events[1].content_block().type(), "text");
  EXPECT_EQ(events[1].content_block().text(), "");
  EXPECT_EQ(events[2].type(), "content_block_delta");
  EXPECT_EQ(events[2].index(), 0);
  EXPECT_EQ(events[2].delta().type(), "text_delta");
  EXPECT_EQ(events[2].delta().text(), "Hel");

  llm::RequestOutput second;
  second.request_id = "anthropiccmpl-test";
  llm::SequenceOutput second_seq;
  second_seq.index = 0;
  second_seq.text = "lo";
  second.outputs.push_back(std::move(second_seq));
  events.clear();

  result = fill_anthropic_stream_events("test-model", second, state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].type(), "content_block_delta");
  EXPECT_EQ(events[0].delta().text(), "lo");

  llm::RequestOutput final;
  final.request_id = "anthropiccmpl-test";
  final.finished = true;
  llm::SequenceOutput final_seq;
  final_seq.index = 0;
  final_seq.finish_reason = "stop";
  final.outputs.push_back(std::move(final_seq));
  llm::Usage usage;
  usage.num_prompt_tokens = 3;
  usage.num_generated_tokens = 5;
  final.usage = usage;
  events.clear();

  result = fill_anthropic_stream_events("test-model", final, state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type(), "content_block_stop");
  EXPECT_EQ(events[0].index(), 0);
  EXPECT_EQ(events[1].type(), "message_delta");
  EXPECT_EQ(events[1].delta().stop_reason(), "end_turn");
  EXPECT_EQ(events[1].usage().input_tokens(), 3);
  EXPECT_EQ(events[1].usage().output_tokens(), 5);
  EXPECT_EQ(events[2].type(), "message_stop");

  std::string sse;
  std::string error;
  ASSERT_TRUE(anthropic_event_sse(events[1], &sse, &error)) << error;
  EXPECT_NE(sse.find("event: message_delta\n"), std::string::npos);
  EXPECT_NE(sse.find("\ndata: {"), std::string::npos);
  EXPECT_NE(sse.find("\"stop_reason\":\"end_turn\""), std::string::npos);
  EXPECT_EQ(sse.substr(sse.size() - 2), "\n\n");
  EXPECT_EQ(anthropic_done_sse(), "data: [DONE]\n\n");
}

TEST(AnthropicAdapterTest, BuildsToolStreamEvents) {
  AnthropicStreamState state;
  std::vector<xllm::proto::AnthropicStreamEvent> events;

  llm::RequestOutput text;
  text.request_id = "anthropiccmpl-test";
  auto result = add_anthropic_text_delta(
      "test-model", text, "Let me check ", state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type(), "message_start");
  EXPECT_EQ(events[1].type(), "content_block_start");
  EXPECT_EQ(events[1].content_block().type(), "text");
  EXPECT_EQ(events[2].type(), "content_block_delta");
  EXPECT_EQ(events[2].delta().type(), "text_delta");

  events.clear();
  result = add_anthropic_tool_delta(
      "test-model", text, "call_1", "get_weather", R"({"city")", state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type(), "content_block_stop");
  EXPECT_EQ(events[0].index(), 0);
  EXPECT_EQ(events[1].type(), "content_block_start");
  EXPECT_EQ(events[1].index(), 1);
  EXPECT_EQ(events[1].content_block().type(), "tool_use");
  EXPECT_EQ(events[1].content_block().id(), "call_1");
  EXPECT_EQ(events[1].content_block().name(), "get_weather");
  EXPECT_TRUE(events[1].content_block().has_input());
  EXPECT_EQ(events[2].type(), "content_block_delta");
  EXPECT_EQ(events[2].index(), 1);
  EXPECT_EQ(events[2].delta().type(), "input_json_delta");
  EXPECT_EQ(events[2].delta().partial_json(), R"({"city")");

  events.clear();
  result = add_anthropic_tool_delta(
      "test-model", text, "", "", R"(: "Beijing"})", state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].type(), "content_block_delta");
  EXPECT_EQ(events[0].index(), 1);
  EXPECT_EQ(events[0].delta().partial_json(), R"(: "Beijing"})");

  llm::RequestOutput final;
  final.request_id = "anthropiccmpl-test";
  final.finished = true;
  llm::SequenceOutput final_seq;
  final_seq.index = 0;
  final_seq.finish_reason = "stop";
  final.outputs.push_back(std::move(final_seq));
  llm::Usage usage;
  usage.num_prompt_tokens = 6;
  usage.num_generated_tokens = 9;
  final.usage = usage;
  events.clear();

  result = finish_anthropic_stream("test-model", final, state, events);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type(), "content_block_stop");
  EXPECT_EQ(events[0].index(), 1);
  EXPECT_EQ(events[1].type(), "message_delta");
  EXPECT_EQ(events[1].delta().stop_reason(), "tool_use");
  EXPECT_EQ(events[1].usage().input_tokens(), 6);
  EXPECT_EQ(events[1].usage().output_tokens(), 9);
  EXPECT_EQ(events[2].type(), "message_stop");
}

TEST(AnthropicAdapterTest, MapsLengthStopReason) {
  llm::RequestOutput output;
  output.request_id = "anthropiccmpl-test";
  llm::SequenceOutput seq;
  seq.index = 0;
  seq.text = "";
  seq.finish_reason = "length";
  output.outputs.push_back(std::move(seq));

  xllm::proto::AnthropicMessagesResponse response;
  auto result = fill_anthropic_resp("test-model", output, &response);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(response.stop_reason(), "max_tokens");
}

}  // namespace
}  // namespace xllm_service
