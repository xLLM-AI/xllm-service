/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

namespace xllm_service {

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  JinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, ApplyChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}{{ message['content'] }}{% endfor %}";

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  auto result = template_.apply(
      messages, nlohmann::ordered_json::array(), chat_template_kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "hello");
}

TEST(JinjaChatTemplate, RendersAnthropicOrderedBlocks) {
  const std::string template_str =
      "{% for item in messages[0]['content'] %}"
      "{{ item['type'] }}:"
      "{% if item['type'] == 'thinking' %}{{ item['thinking'] }}:{{ "
      "item['signature'] }}{% endif %}"
      "{% if item['type'] == 'tool_use' %}{{ item['id'] }}:{{ item['name'] "
      "}}:{{ item['input']['path'] }}{% endif %}"
      "{% if item['type'] == 'redacted_thinking' %}{{ item['data'] }}{% endif "
      "%}"
      "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}|"
      "{% endfor %}"
      "reasoning={{ messages[0]['reasoning_content'] }}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  Message::MMContent thinking("thinking");
  thinking.thinking = "need tool";
  thinking.signature = "sig_1";

  Message::MMContent tool_use("tool_use");
  tool_use.id = "toolu_1";
  tool_use.name = "Read";
  tool_use.input = {{"path", "a.txt"}};

  Message::MMContent redacted("redacted_thinking");
  redacted.data = "opaque";

  Message::MMContent text("text", "done");

  Message message("assistant",
                  Message::MMContentVec{thinking, tool_use, redacted, text});
  message.reasoning_content = "need tool";
  ChatMessages messages{message};

  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(),
            "thinking:need tool:sig_1|tool_use:toolu_1:Read:a.txt|"
            "redacted_thinking:opaque|text:done|reasoning=need tool");
}

TEST(JinjaChatTemplate, RendersPreservedThinkingForStringTemplates) {
  const std::string template_str =
      "{% for message in messages %}"
      "{{ message['role'] }}:"
      "{% if message['content'] is string %}{{ message['content'] }}{% endif "
      "%}|"
      "{% if message.get('tool_calls') %}"
      "{% for tool_call in message['tool_calls'] %}"
      "call={{ tool_call['function']['arguments'] }}|"
      "{% endfor %}"
      "{% endif %}"
      "{% endfor %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  Message::MMContent thinking("thinking");
  thinking.thinking = "PTB_marker";
  thinking.signature = "sig_1";

  Message::MMContent tool_use("tool_use");
  tool_use.id = "toolu_1";
  tool_use.name = "preserve_probe";
  tool_use.input = {{"marker", "TIB_marker"}};

  Message::MMContent text("text", "done");

  Message assistant("assistant",
                    Message::MMContentVec{thinking, tool_use, text});
  assistant.reasoning_content = "PTB_marker";
  Message::ToolCall tool_call;
  tool_call.id = "toolu_1";
  tool_call.type = "function";
  tool_call.function.name = "preserve_probe";
  tool_call.function.arguments = R"({"marker":"TIB_marker"})";
  assistant.tool_calls = Message::ToolCallVec{tool_call};

  Message tool("tool", "TRB_marker");
  tool.tool_call_id = "toolu_1";

  auto result = template_.apply(ChatMessages{assistant, tool});
  ASSERT_TRUE(result.has_value());
  EXPECT_NE(result->find("<think>PTB_marker</think>"), std::string::npos);
  EXPECT_NE(result->find("TIB_marker"), std::string::npos);
  EXPECT_NE(result->find("TRB_marker"), std::string::npos);
}

}  // namespace xllm_service
