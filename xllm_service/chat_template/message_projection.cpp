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

#include "chat_template/mm_content_text.h"

namespace xllm_service {

void to_proto(const Message& msg, xllm::proto::ChatMessage* out) {
  out->set_role(msg.role);
  if (std::holds_alternative<std::string>(msg.content)) {
    out->set_content(std::get<std::string>(msg.content));
  } else {
    out->set_content(flat_text(std::get<Message::MMContentVec>(msg.content)));
  }
  if (msg.reasoning_content.has_value() && !msg.reasoning_content->empty()) {
    out->set_reasoning_content(*msg.reasoning_content);
  }
  if (msg.tool_calls.has_value()) {
    for (const auto& tool_call : *msg.tool_calls) {
      auto* proto_call = out->add_tool_calls();
      proto_call->set_id(tool_call.id);
      proto_call->set_type(tool_call.type);
      auto* proto_function = proto_call->mutable_function();
      proto_function->set_name(tool_call.function.name);
      proto_function->set_arguments(tool_call.function.arguments);
    }
  }
  if (!msg.tool_call_id.empty()) {
    out->set_tool_call_id(msg.tool_call_id);
  }
}

}  // namespace xllm_service
