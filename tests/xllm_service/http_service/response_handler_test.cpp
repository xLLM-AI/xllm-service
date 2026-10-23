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

#include "http_service/response_handler.h"

#include <brpc/callback.h>
#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace xllm_service {
namespace {

llm::RequestOutput chat_output() {
  llm::RequestOutput output;
  output.request_id = "request";
  output.finished = true;
  output.chat_frames.emplace();
  auto& frame = output.chat_frames->emplace_back();
  frame.set_id("request");
  frame.set_model("model");
  frame.set_object("chat.completion");
  auto* choice = frame.add_choices();
  choice->set_index(0);
  choice->set_finish_reason("tool_calls");
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content("answer");
  message->set_reasoning_content("reason");
  auto* tool = message->add_tool_calls();
  tool->set_id("call_engine");
  tool->set_type("function");
  tool->mutable_function()->set_name("weather");
  tool->mutable_function()->set_arguments(R"({"city":"Paris"})");
  output.usage = llm::Usage{8, 2, 10, 4};
  frame.mutable_usage()->set_prompt_tokens(8);
  frame.mutable_usage()->set_completion_tokens(2);
  frame.mutable_usage()->set_total_tokens(10);
  // Raw text must never reach the client or a model parser in the service.
  llm::SequenceOutput sequence;
  sequence.index = 0;
  sequence.text = "<think>RAW MUST NOT BE PARSED</think>";
  output.outputs.emplace_back(std::move(sequence));
  return output;
}

TEST(ResponseHandlerTest, ForwardsEngineChatFieldsWithoutReparsing) {
  ResponseHandler handler;
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  auto call = std::make_shared<ChatCallData>(
      &controller, false, brpc::DoNothing(), &request, &response);
  auto output = chat_output();
  ASSERT_TRUE(handler.send_result_to_client(call, output));
  ASSERT_FALSE(controller.Failed());
  auto json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  EXPECT_EQ(json["choices"][0]["message"]["content"], "answer");
  EXPECT_EQ(json["choices"][0]["message"]["reasoning_content"], "reason");
  EXPECT_EQ(json["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_engine");
  EXPECT_EQ(json["choices"][0]["finish_reason"], "tool_calls");
  EXPECT_EQ(json["usage"]["completion_tokens"], 2);
}

TEST(ResponseHandlerTest, RejectsLegacyRawChatInsteadOfParsingFallback) {
  ResponseHandler handler;
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  auto call = std::make_shared<ChatCallData>(
      &controller, false, brpc::DoNothing(), &request, &response);
  auto output = chat_output();
  output.chat_frames.reset();
  EXPECT_FALSE(handler.send_result_to_client(call, output));
  EXPECT_TRUE(controller.Failed());
  EXPECT_NE(controller.ErrorText().find("structured chat_output"),
            std::string::npos);
}

TEST(ResponseHandlerTest, EmptyParsedStreamWaitsWithoutProducingOutput) {
  ResponseHandler handler;
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  auto call = std::make_shared<ChatCallData>(
      &controller, false, brpc::DoNothing(), &request, &response);
  llm::RequestOutput output;
  output.chat_frames.emplace();
  EXPECT_TRUE(handler.send_delta_to_client(call, output));
  EXPECT_FALSE(controller.Failed());
  EXPECT_TRUE(controller.response_attachment().empty());
}

TEST(ResponseHandlerTest, RejectsDeltaFrameOnNonStreamChat) {
  ResponseHandler handler;
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  auto call = std::make_shared<ChatCallData>(
      &controller, false, brpc::DoNothing(), &request, &response);
  auto output = chat_output();
  output.chat_frames->front().set_object("chat.completion.chunk");
  EXPECT_FALSE(handler.send_result_to_client(call, output));
  EXPECT_TRUE(controller.Failed());
}

TEST(ResponseHandlerTest, EncodesStructuredReasoningAndToolsAsAnthropicBlocks) {
  ResponseHandler handler;
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::AnthropicMessagesResponse response;
  auto call = std::make_shared<AnthropicCallData>(
      &controller, false, brpc::DoNothing(), &request, &response);
  auto output = chat_output();
  ASSERT_TRUE(handler.send_result_to_client(call, "model", output));
  ASSERT_FALSE(controller.Failed());
  auto json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  ASSERT_EQ(json["content"].size(), 3);
  EXPECT_EQ(json["content"][0]["type"], "thinking");
  EXPECT_EQ(json["content"][0]["thinking"], "reason");
  EXPECT_EQ(json["content"][1]["text"], "answer");
  EXPECT_EQ(json["content"][2]["type"], "tool_use");
  EXPECT_EQ(json["content"][2]["id"], "call_engine");
  EXPECT_EQ(json["content"][2]["input"]["city"], "Paris");
  EXPECT_EQ(json["stop_reason"], "tool_use");
  EXPECT_EQ(json["usage"]["output_tokens"], 2);
}

}  // namespace
}  // namespace xllm_service
