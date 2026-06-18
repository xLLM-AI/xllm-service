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

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "chat.pb.h"
#include "common/types.h"

namespace xllm {
class StreamOutputParser;
}

namespace xllm_service {

struct ChatParseResult {
  std::optional<google::protobuf::RepeatedPtrField<::xllm::proto::ToolCall>>
      tool_calls;
  std::optional<std::string> reasoning_content;
  std::string text;
  std::string finish_reason;
};

struct ResolvedChatParserFormats {
  std::string tool_call_parser;
  std::string reasoning_parser;
};

ResolvedChatParserFormats resolve_chat_parser_formats_with_xllm(
    const std::string& model,
    const std::string& parser_preference = "",
    const std::string& reasoning_parser_preference = "");

bool get_enable_thinking_from_request(
    const nlohmann::json& chat_template_kwargs,
    const std::string& reasoning_parser_format);

std::shared_ptr<xllm::StreamOutputParser> create_stream_output_parser_with_xllm(
    const std::vector<JsonTool>& tools,
    const std::string& model,
    const std::string& parser_preference = "",
    const std::string& reasoning_parser_preference = "",
    bool force_reasoning = false);

ChatParseResult parse_chat_output_with_xllm(
    std::string text,
    const std::vector<JsonTool>& tools,
    const std::string& model,
    std::string finish_reason,
    const std::string& parser_preference = "",
    const std::string& reasoning_parser_preference = "",
    bool force_reasoning = false,
    google::protobuf::Arena* arena = nullptr);

}  // namespace xllm_service
