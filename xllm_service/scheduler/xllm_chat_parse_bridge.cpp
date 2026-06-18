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

#include "scheduler/xllm_chat_parse_bridge.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <exception>
#include <utility>

#include "xllm/xllm/api_service/stream_output_parser.h"
#include "xllm/xllm/function_call/function_call_parser.h"
#include "xllm/xllm/parser/reasoning_parser.h"

namespace xllm_service {

namespace {
// Keep xllm_service tool types decoupled from third_party xllm types.
// This bridge-local adapter isolates dependency and include path impact.
std::vector<xllm::JsonTool> to_xllm_tools(
    const std::vector<JsonTool>& service_tools) {
  std::vector<xllm::JsonTool> xllm_tools;
  xllm_tools.reserve(service_tools.size());

  for (const auto& tool : service_tools) {
    xllm::JsonTool xllm_tool;
    xllm_tool.type = tool.type;
    xllm_tool.function.name = tool.function.name;
    xllm_tool.function.description = tool.function.description;
    xllm_tool.function.parameters = tool.function.parameters;
    xllm_tools.emplace_back(std::move(xllm_tool));
  }
  return xllm_tools;
}

std::string infer_model_type_from_model_id(const std::string& model_id) {
  std::string lower = absl::AsciiStrToLower(model_id);
  if (lower.find("qwen3") != std::string::npos) {
    return "qwen3";
  }
  if (lower.find("qwen2") != std::string::npos) {
    return "qwen2";
  }
  if (lower.find("kimi_k2") != std::string::npos ||
      lower.find("kimi-k2") != std::string::npos) {
    return "kimi_k2";
  }
  if (lower.find("deepseek_v32") != std::string::npos ||
      lower.find("deepseek-v3.2") != std::string::npos ||
      lower.find("deepseekv32") != std::string::npos) {
    return "deepseek_v32";
  }
  if (lower.find("deepseek_v3") != std::string::npos ||
      lower.find("deepseek-v3") != std::string::npos ||
      lower.find("deepseekv3") != std::string::npos) {
    return "deepseek_v3";
  }
  if (lower.find("glm") != std::string::npos) {
    return "glm4_moe";
  }
  if (lower.find("step3") != std::string::npos) {
    return "step3";
  }
  return "";
}

std::string resolve_tool_call_parser(const std::string& parser_preference,
                                     const std::string& model_id) {
  if (parser_preference.empty()) {
    return "";
  }

  std::string model_type;
  if (parser_preference == "auto") {
    model_type = infer_model_type_from_model_id(model_id);
    if (model_type.empty()) {
      // Keep xllm_service behavior compatible: unknown model_id under auto
      // silently disables parsing instead of aborting the process.
      return "";
    }
  }

  return xllm::function_call::FunctionCallParser::get_parser_auto(
      parser_preference, model_type);
}

std::string resolve_reasoning_parser(
    const std::string& reasoning_parser_preference,
    const std::string& model_id) {
  if (reasoning_parser_preference.empty()) {
    return "";
  }

  std::string model_type;
  if (reasoning_parser_preference == "auto") {
    model_type = infer_model_type_from_model_id(model_id);
    if (model_type.empty()) {
      // Keep xllm_service behavior compatible: unknown model_id under auto
      // silently disables parsing instead of aborting the process.
      return "";
    }
  }

  return xllm::ReasoningParser::get_parser_auto(reasoning_parser_preference,
                                                model_type);
}

std::string strip_think_end(std::string text) {
  return absl::StrReplaceAll(text, {{"</think>", ""}});
}
}  // namespace

bool get_enable_thinking_from_request(
    const nlohmann::json& chat_template_kwargs,
    const std::string& reasoning_parser_format) {
  bool default_value = !reasoning_parser_format.empty();
  if (chat_template_kwargs.empty()) {
    return default_value;
  }

  if (chat_template_kwargs.contains("enable_thinking") ||
      chat_template_kwargs.contains("thinking")) {
    auto get_bool_val = [&](const char* key) {
      auto it = chat_template_kwargs.find(key);
      if (it != chat_template_kwargs.end() && it->is_boolean()) {
        return it->get<bool>();
      }
      return false;
    };
    return get_bool_val("enable_thinking") || get_bool_val("thinking");
  }

  return default_value;
}

ChatParseResult parse_chat_output_with_xllm(
    std::string text,
    const std::vector<JsonTool>& tools,
    const std::string& model,
    std::string finish_reason,
    const std::string& parser_preference,
    const std::string& reasoning_parser_preference,
    bool force_reasoning,
    google::protobuf::Arena* arena) {
  ChatParseResult result;
  result.text = std::move(text);
  result.finish_reason = std::move(finish_reason);

  const std::string reasoning_parser_format =
      resolve_reasoning_parser(reasoning_parser_preference, model);
  const std::string parser_format =
      resolve_tool_call_parser(parser_preference, model);

  if (!tools.empty() && !parser_format.empty() && !result.text.empty()) {
    auto xllm_tools = to_xllm_tools(tools);
    xllm::function_call::FunctionCallParser parser(xllm_tools, parser_format);

    if (parser.has_tool_call(result.text)) {
      if (result.finish_reason == "stop") {
        result.finish_reason = "tool_calls";
      }

      try {
        auto [parsed_text, call_info_list] =
            parser.parse_non_stream(result.text);
        result.text = strip_think_end(std::move(parsed_text));

        google::protobuf::RepeatedPtrField<::xllm::proto::ToolCall> tool_calls;
        for (const auto& call_info : call_info_list) {
          ::xllm::proto::ToolCall* tool_call =
              arena ? google::protobuf::Arena::CreateMessage<
                          ::xllm::proto::ToolCall>(arena)
                    : new ::xllm::proto::ToolCall();

          tool_call->set_id(
              xllm::function_call::utils::generate_tool_call_id());
          tool_call->set_type("function");

          auto* function = tool_call->mutable_function();
          if (call_info.name.has_value()) {
            function->set_name(*call_info.name);
          }
          function->set_arguments(call_info.parameters);

          tool_calls.AddAllocated(tool_call);
        }

        result.tool_calls = std::move(tool_calls);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Tool call parsing error: " << e.what();
      }
    }
  }

  if (!reasoning_parser_format.empty() && !result.text.empty()) {
    try {
      xllm::ReasoningParser reasoning_parser(reasoning_parser_format,
                                             /*stream_reasoning=*/false,
                                             force_reasoning);
      auto reasoning_result = reasoning_parser.parse_non_stream(result.text);
      if (reasoning_result.normal_text.has_value()) {
        result.text = reasoning_result.normal_text.value();
      } else {
        result.text.clear();
      }
      if (reasoning_result.reasoning_text.has_value()) {
        result.reasoning_content = reasoning_result.reasoning_text.value();
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Reasoning parsing error: " << e.what();
    }
  }

  return result;
}

ResolvedChatParserFormats resolve_chat_parser_formats_with_xllm(
    const std::string& model,
    const std::string& parser_preference,
    const std::string& reasoning_parser_preference) {
  ResolvedChatParserFormats formats;
  formats.tool_call_parser = resolve_tool_call_parser(parser_preference, model);
  formats.reasoning_parser =
      resolve_reasoning_parser(reasoning_parser_preference, model);
  return formats;
}

std::shared_ptr<xllm::StreamOutputParser> create_stream_output_parser_with_xllm(
    const std::vector<JsonTool>& tools,
    const std::string& model,
    const std::string& parser_preference,
    const std::string& reasoning_parser_preference,
    bool force_reasoning) {
  auto formats = resolve_chat_parser_formats_with_xllm(
      model, parser_preference, reasoning_parser_preference);
  if (formats.tool_call_parser.empty() && formats.reasoning_parser.empty()) {
    return nullptr;
  }

  auto xllm_tools = to_xllm_tools(tools);
  return std::make_shared<xllm::StreamOutputParser>(xllm_tools,
                                                    formats.tool_call_parser,
                                                    formats.reasoning_parser,
                                                    force_reasoning);
}

}  // namespace xllm_service
