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

#include "http_service/chat_json_parser.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

namespace xllm_service {
namespace {

nlohmann::json parse(const std::string& json_str) {
  return nlohmann::json::parse(json_str);
}

void expect_json(const std::string& input, const std::string& expected) {
  auto result = normalize_chat_json(input);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(parse(result.json), parse(expected));
}

void expect_error(const std::string& input, const std::string& error) {
  auto result = normalize_chat_json(input);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find(error), std::string::npos) << result.error;
}

TEST(ChatJsonParserTest, StringContentPassesThrough) {
  std::string input = R"({
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}]
  })";
  expect_json(input, input);
}

TEST(ChatJsonParserTest, SingleTextItemCombined) {
  std::string input = R"({
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "hello"}]
  })";
  expect_json(input, expected);
}

TEST(ChatJsonParserTest, MultipleTextItemsUseNewline) {
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"}
      ]
    }]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "hello\nworld"}]
  })";
  expect_json(input, expected);
}

TEST(ChatJsonParserTest, MultipleMessagesMixedContent) {
  std::string input = R"({
    "messages": [
      {"role": "system", "content": "plain"},
      {"role": "user", "content": [{"type": "text", "text": "array"}]}
    ]
  })";
  std::string expected = R"({
    "messages": [
      {"role": "system", "content": "plain"},
      {"role": "user", "content": "array"}
    ]
  })";
  expect_json(input, expected);
}

TEST(ChatJsonParserTest, EmptyArrayBecomesEmptyString) {
  std::string input = R"({
    "messages": [{"role": "user", "content": []}]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": ""}]
  })";
  expect_json(input, expected);
}

TEST(ChatJsonParserTest, PreservesOtherFields) {
  std::string input = R"({
    "model": "test",
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false},
    "tool_choice": "auto",
    "tools": [{"type": "function", "function": {"name": "f"}}],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
  })";
  auto result = normalize_chat_json(input);
  ASSERT_TRUE(result.ok) << result.error;
  auto json = parse(result.json);
  EXPECT_EQ(json["model"], "test");
  EXPECT_EQ(json["stream"], true);
  EXPECT_EQ(json["chat_template_kwargs"]["enable_thinking"], false);
  EXPECT_EQ(json["tool_choice"], "auto");
  EXPECT_EQ(json["tools"][0]["function"]["name"], "f");
  EXPECT_EQ(json["messages"][0]["content"], "hello");
}

TEST(ChatJsonParserTest, InvalidJsonReturnsError) {
  expect_error("{", "Invalid JSON format");
}

TEST(ChatJsonParserTest, NonObjectMessageReturnsError) {
  expect_error(R"({"messages": ["bad"]})", "Message in 'messages'");
}

TEST(ChatJsonParserTest, NonObjectItemReturnsError) {
  expect_error(R"({"messages": [{"content": ["bad"]}]})", "Content array item");
}

TEST(ChatJsonParserTest, MissingTypeReturnsError) {
  expect_error(R"({"messages": [{"content": [{"text": "hello"}]}]})",
               "Only text content arrays");
}

TEST(ChatJsonParserTest, NonTextTypeReturnsError) {
  expect_error(
      R"({"messages": [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})",
      "Only text content arrays");
}

TEST(ChatJsonParserTest, MissingTextReturnsError) {
  expect_error(R"({"messages": [{"content": [{"type": "text"}]}]})",
               "Missing or invalid 'text'");
}

TEST(ChatJsonParserTest, NonStringTextReturnsError) {
  expect_error(R"({"messages": [{"content": [{"type": "text", "text": 1}]}]})",
               "Missing or invalid 'text'");
}

}  // namespace
}  // namespace xllm_service
