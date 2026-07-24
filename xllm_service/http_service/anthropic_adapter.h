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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "chat_template/jinja_chat_template.h"
#include "common/xllm/output.h"

namespace xllm_service {

struct AnthropicAdaptResult {
  bool ok = true;
  std::string error;
};

std::string new_anthropic_id();

AnthropicAdaptResult parse_anthropic_json(
    std::string json,
    xllm::proto::AnthropicMessagesRequest* request);

AnthropicAdaptResult fill_chat_req(
    const xllm::proto::AnthropicMessagesRequest& anthropic_request,
    xllm::proto::ChatRequest* chat_request,
    ChatMessages* messages);

AnthropicAdaptResult fill_anthropic_resp(
    const std::string& model,
    const llm::RequestOutput& request_output,
    xllm::proto::AnthropicMessagesResponse* response,
    const google::protobuf::RepeatedPtrField<xllm::proto::ToolCall>*
        tool_calls = nullptr);

bool anthropic_json(const xllm::proto::AnthropicMessagesResponse& response,
                    std::string* json,
                    std::string* error);

bool anthropic_json(const xllm::proto::AnthropicMessagesResponse& response,
                    const std::optional<std::string>& thinking,
                    std::string* json,
                    std::string* error);

bool anthropic_event_sse(const xllm::proto::AnthropicStreamEvent& event,
                         std::string* sse,
                         std::string* error);

std::string anthropic_done_sse();

}  // namespace xllm_service
