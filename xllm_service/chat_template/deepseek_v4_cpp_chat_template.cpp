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

#include <variant>

#include "chat_template/mm_content_text.h"  // pulls in xllm_service::Message
#include "core/common/message.h"
#include "core/common/types.h"
#include "framework/chat_template/deepseek_v4_cpp_template.h"
#include "framework/tokenizer/tokenizer_args.h"

// <torch/torch.h> (via message.h) defines c10's own LOG macro, but we don't
// link libtorch. Restore glog's LOG macro for our logging.
#include <glog/logging.h>
#undef LOG
#define LOG(severity) COMPACT_GOOGLE_LOG_##severity.stream()

namespace xllm_service {
namespace {

// Copies the service TokenizerArgs onto the upstream one (V4 only reads
// bos_token, but we copy all same-named string fields).
xllm::TokenizerArgs to_xllm_tokenizer_args(const TokenizerArgs& args) {
  xllm::TokenizerArgs out;
  out.chat_template(args.chat_template());
  out.add_bos_token(args.add_bos_token());
  out.add_eos_token(args.add_eos_token());
  out.bos_token(args.bos_token());
  out.eos_token(args.eos_token());
  out.pad_token(args.pad_token());
  out.tokenizer_class(args.tokenizer_class());
  return out;
}

xllm::Message to_xllm_message(const Message& msg) {
  // Flatten vector content to text; upstream get_text_content drops vectors.
  std::string content =
      std::holds_alternative<std::string>(msg.content)
          ? std::get<std::string>(msg.content)
          : flat_text(std::get<Message::MMContentVec>(msg.content));
  xllm::Message out(msg.role, content);

  if (msg.reasoning_content.has_value()) {
    out.reasoning_content = *msg.reasoning_content;
  }
  if (msg.tool_calls.has_value()) {
    xllm::Message::ToolCallVec calls;
    calls.reserve(msg.tool_calls->size());
    for (const auto& call : *msg.tool_calls) {
      xllm::Message::ToolCall out_call;
      out_call.id = call.id;
      out_call.type = call.type;
      out_call.function.name = call.function.name;
      out_call.function.arguments = call.function.arguments;
      calls.push_back(std::move(out_call));
    }
    out.tool_calls = std::move(calls);
  }
  if (!msg.tool_call_id.empty()) {
    out.tool_call_id = msg.tool_call_id;
  }
  return out;
}

std::vector<xllm::JsonTool> to_xllm_tools(const std::vector<JsonTool>& tools) {
  std::vector<xllm::JsonTool> out;
  out.reserve(tools.size());
  for (const auto& tool : tools) {
    xllm::JsonTool xllm_tool;
    xllm_tool.type = tool.type;
    xllm_tool.function.name = tool.function.name;
    xllm_tool.function.description = tool.function.description;
    xllm_tool.function.parameters = tool.function.parameters;
    out.push_back(std::move(xllm_tool));
  }
  return out;
}

}  // namespace

DeepseekV4CppChatTemplate::DeepseekV4CppChatTemplate(const TokenizerArgs& args)
    : impl_(std::make_unique<xllm::DeepseekV4CppTemplate>(
          to_xllm_tokenizer_args(args))) {
  LOG(INFO) << "DeepSeek V4 cpp chat template initialized.";
}

DeepseekV4CppChatTemplate::~DeepseekV4CppChatTemplate() = default;

std::optional<std::string> DeepseekV4CppChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  xllm::ChatMessages xllm_messages;
  xllm_messages.reserve(messages.size());
  for (const auto& msg : messages) {
    xllm_messages.push_back(to_xllm_message(msg));
  }
  return impl_->apply(
      xllm_messages, to_xllm_tools(json_tools), chat_template_kwargs);
}

}  // namespace xllm_service
