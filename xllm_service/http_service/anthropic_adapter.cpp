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

#include "http_service/anthropic_adapter.h"

#include <google/protobuf/util/json_util.h>
#include <json2pb/pb_to_json.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "api_service/anthropic_json.h"
#include "api_service/anthropic_stream_utils.h"
#include "api_service/chat_json_parser.h"
#include "chat_template/message_projection.h"
#include "common/xllm/uuid.h"

namespace xllm_service {
namespace {

thread_local llm::ShortUUID short_uuid;

AnthropicAdaptResult ok_result() { return AnthropicAdaptResult{}; }

AnthropicAdaptResult error_result(std::string error) {
  return AnthropicAdaptResult{false, std::move(error)};
}

// Synthetic Anthropic-shaped compatibility value, not a Claude verification
// signature.
std::string new_thinking_sig() { return short_uuid.random(); }

std::string system_text(const xllm::proto::AnthropicContentBlockList& blocks) {
  std::string text;
  for (const auto& block : blocks.blocks()) {
    text += block.text();
  }
  return text;
}

AnthropicAdaptResult check_text_blocks(
    const xllm::proto::AnthropicContentBlockList& blocks) {
  for (const auto& block : blocks.blocks()) {
    if (block.type() != "text") {
      return error_result("Unsupported Anthropic content block type: " +
                          block.type());
    }
    if (!block.has_text()) {
      return error_result("Missing text in Anthropic text content block.");
    }
  }
  return ok_result();
}

std::string struct_json(const google::protobuf::Struct& proto_struct) {
  std::string json;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = false;
  google::protobuf::util::MessageToJsonString(proto_struct, &json, options);
  return json;
}

AnthropicAdaptResult tool_result_text(
    const xllm::proto::AnthropicContentBlock& block,
    std::string* text) {
  if (block.has_content_string()) {
    *text = block.content_string();
    return ok_result();
  }
  if (!block.has_content_list()) {
    text->clear();
    return ok_result();
  }

  bool first = true;
  text->clear();
  for (const auto& item : block.content_list().items()) {
    auto type_iter = item.fields().find("type");
    auto text_iter = item.fields().find("text");
    if (type_iter == item.fields().end() ||
        type_iter->second.string_value() != "text" ||
        text_iter == item.fields().end()) {
      return error_result("Unsupported Anthropic tool_result content block.");
    }
    if (!first) {
      *text += '\n';
    }
    *text += text_iter->second.string_value();
    first = false;
  }
  return ok_result();
}

AnthropicAdaptResult add_tool_use(
    const xllm::proto::AnthropicContentBlock& block,
    Message::ToolCallVec* tool_calls) {
  if (!block.has_id()) {
    return error_result("Anthropic tool_use id is required.");
  }
  if (!block.has_name()) {
    return error_result("Anthropic tool_use name is required.");
  }

  Message::ToolCall tool_call;
  tool_call.id = block.id();
  tool_call.type = "function";
  tool_call.function.name = block.name();
  tool_call.function.arguments =
      block.has_input() ? struct_json(block.input()) : "{}";
  tool_calls->emplace_back(std::move(tool_call));
  return ok_result();
}

bool is_empty_bash_tool_use(const xllm::proto::AnthropicContentBlock& block) {
  return block.has_name() && block.name() == "Bash" &&
         (!block.has_input() || block.input().fields().empty());
}

nlohmann::ordered_json struct_to_json(
    const google::protobuf::Struct& proto_struct) {
  return nlohmann::ordered_json::parse(struct_json(proto_struct));
}

Message::MMContent thinking_content(
    const xllm::proto::AnthropicContentBlock& block) {
  Message::MMContent content("thinking");
  if (block.has_thinking()) {
    content.thinking = block.thinking();
  }
  if (block.has_signature()) {
    content.signature = block.signature();
  }
  return content;
}

Message::MMContent redacted_content(
    const xllm::proto::AnthropicContentBlock& block) {
  Message::MMContent content("redacted_thinking");
  if (block.has_data()) {
    content.data = block.data();
  }
  return content;
}

Message::MMContent tool_use_content(
    const xllm::proto::AnthropicContentBlock& block) {
  Message::MMContent content("tool_use");
  if (block.has_id()) {
    content.id = block.id();
  }
  if (block.has_name()) {
    content.name = block.name();
  }
  if (block.has_input()) {
    content.input = struct_to_json(block.input());
  }
  return content;
}

AnthropicAdaptResult add_tool_result(
    const xllm::proto::AnthropicContentBlock& block,
    const std::string& role,
    const std::unordered_set<std::string>& skipped_tool_use_ids,
    ChatMessages* messages,
    Message::MMContentVec* content_parts) {
  if (!block.has_id()) {
    return error_result("Anthropic tool_result tool_use_id is required.");
  }
  if (skipped_tool_use_ids.find(block.id()) != skipped_tool_use_ids.end()) {
    return ok_result();
  }

  std::string result_text;
  auto text_result = tool_result_text(block, &result_text);
  if (!text_result.ok) {
    return text_result;
  }

  if (role == "user") {
    Message message("tool", result_text);
    message.tool_call_id = block.id();
    messages->emplace_back(std::move(message));
    return ok_result();
  }

  content_parts->emplace_back("text", "Tool result: " + result_text);
  return ok_result();
}

AnthropicAdaptResult add_tool_defs(
    const xllm::proto::AnthropicMessagesRequest& request,
    xllm::proto::ChatRequest* chat_request) {
  for (const auto& anthropic_tool : request.tools()) {
    auto* tool = chat_request->add_tools();
    tool->set_type("function");
    auto* function = tool->mutable_function();
    function->set_name(anthropic_tool.name());
    if (anthropic_tool.has_description()) {
      function->set_description(anthropic_tool.description());
    }
    if (anthropic_tool.has_input_schema()) {
      function->mutable_parameters()->CopyFrom(anthropic_tool.input_schema());
    } else {
      function->mutable_parameters();
    }
  }
  return ok_result();
}

std::string tool_choice(const xllm::proto::AnthropicMessagesRequest& request) {
  if (!request.has_tool_choice()) {
    return request.tools_size() > 0 ? "auto" : "none";
  }

  const auto& choice = request.tool_choice();
  if (choice.type() == "auto") {
    return "auto";
  }
  if (choice.type() == "any") {
    return "required";
  }
  if (choice.type() == "tool") {
    if (!choice.has_name()) {
      return "auto";
    }
    nlohmann::json choice_json = {{"type", "function"},
                                  {"function", {{"name", choice.name()}}}};
    return choice_json.dump();
  }
  return "auto";
}

AnthropicAdaptResult add_system_msg(
    const xllm::proto::AnthropicMessagesRequest& request,
    ChatMessages* messages) {
  if (request.has_system_string()) {
    messages->emplace_back("system", request.system_string());
    return ok_result();
  }

  if (!request.has_system_blocks()) {
    return ok_result();
  }

  auto checked = check_text_blocks(request.system_blocks());
  if (!checked.ok) {
    return checked;
  }
  std::string text = system_text(request.system_blocks());
  if (text.empty()) {
    return ok_result();
  }
  messages->emplace_back("system", std::move(text));
  return ok_result();
}

AnthropicAdaptResult add_content_msg(
    const xllm::proto::AnthropicMessage& src_message,
    ChatMessages* messages,
    std::unordered_set<std::string>* skipped_tool_use_ids) {
  switch (src_message.message_content_case()) {
    case xllm::proto::AnthropicMessage::kContentString: {
      messages->emplace_back(src_message.role(), src_message.content_string());
      return ok_result();
    }
    case xllm::proto::AnthropicMessage::kContentBlocks: {
      Message::MMContentVec ordered_content;
      Message::ToolCallVec tool_calls;
      std::string reasoning_content;
      for (const auto& block : src_message.content_blocks().blocks()) {
        if (block.type() == "text" && block.has_text()) {
          ordered_content.emplace_back("text", block.text());
        } else if (block.type() == "tool_use") {
          if (is_empty_bash_tool_use(block)) {
            if (block.has_id()) {
              skipped_tool_use_ids->insert(block.id());
            }
            continue;
          }
          auto result = add_tool_use(block, &tool_calls);
          if (!result.ok) {
            return result;
          }
          ordered_content.emplace_back(tool_use_content(block));
        } else if (block.type() == "tool_result") {
          auto result = add_tool_result(block,
                                        src_message.role(),
                                        *skipped_tool_use_ids,
                                        messages,
                                        &ordered_content);
          if (!result.ok) {
            return result;
          }
        } else if (block.type() == "thinking") {
          ordered_content.emplace_back(thinking_content(block));
          if (block.has_thinking()) {
            reasoning_content += block.thinking();
          }
        } else if (block.type() == "redacted_thinking") {
          ordered_content.emplace_back(redacted_content(block));
        } else {
          return error_result("Unsupported Anthropic content block type: " +
                              block.type());
        }
      }

      const std::string text = flat_text(ordered_content);
      if (ordered_content.empty() && tool_calls.empty()) {
        return ok_result();
      }

      Message dst_message(src_message.role(), text);
      if (!reasoning_content.empty()) {
        dst_message.reasoning_content = reasoning_content;
      }
      if (!tool_calls.empty()) {
        dst_message.tool_calls = std::move(tool_calls);
      }
      if (needs_content_vec(ordered_content)) {
        dst_message.content = std::move(ordered_content);
      }
      messages->emplace_back(std::move(dst_message));
      return ok_result();
    }
    case xllm::proto::AnthropicMessage::MESSAGE_CONTENT_NOT_SET:
    default:
      return error_result("Anthropic message content is required.");
  }
}

void fill_generation_params(
    const xllm::proto::AnthropicMessagesRequest& anthropic_request,
    xllm::proto::ChatRequest* chat_request) {
  chat_request->set_model(anthropic_request.model());
  chat_request->set_max_tokens(
      static_cast<uint32_t>(anthropic_request.max_tokens()));
  if (anthropic_request.has_stream()) {
    chat_request->set_stream(anthropic_request.stream());
  }
  if (anthropic_request.has_temperature()) {
    chat_request->set_temperature(anthropic_request.temperature());
  }
  if (anthropic_request.has_top_p()) {
    chat_request->set_top_p(anthropic_request.top_p());
  }
  if (anthropic_request.has_top_k()) {
    chat_request->set_top_k(anthropic_request.top_k());
  }
  if (anthropic_request.has_ignore_eos()) {
    chat_request->set_ignore_eos(anthropic_request.ignore_eos());
  }
  for (const auto& stop : anthropic_request.stop_sequences()) {
    chat_request->add_stop(stop);
  }
}

void fill_usage(const llm::RequestOutput& request_output,
                xllm::proto::AnthropicMessagesResponse* response) {
  if (!request_output.usage.has_value()) {
    return;
  }
  const auto& usage = request_output.usage.value();
  auto* proto_usage = response->mutable_usage();
  proto_usage->set_input_tokens(static_cast<int32_t>(usage.num_prompt_tokens));
  proto_usage->set_output_tokens(
      static_cast<int32_t>(usage.num_generated_tokens));
}

bool normalize_stream_event_json(const xllm::proto::AnthropicStreamEvent& event,
                                 std::string* json,
                                 std::string* error) {
  try {
    nlohmann::json parsed = nlohmann::json::parse(*json);

    if (event.type() == "message_start" && event.has_message()) {
      auto& message = parsed["message"];
      if (!event.message().has_stop_reason()) {
        message["stop_reason"] = nullptr;
      }
      if (!event.message().has_stop_sequence()) {
        message["stop_sequence"] = nullptr;
      }
      auto& usage = message["usage"];
      usage["input_tokens"] = event.message().usage().input_tokens();
      usage["output_tokens"] = event.message().usage().output_tokens();
    }

    if (event.type() == "message_delta" && event.has_delta()) {
      auto& delta = parsed["delta"];
      if (!event.delta().has_stop_sequence()) {
        delta["stop_sequence"] = nullptr;
      }
      if (event.has_usage()) {
        auto& usage = parsed["usage"];
        usage["input_tokens"] = event.usage().input_tokens();
        usage["output_tokens"] = event.usage().output_tokens();
      }
    }

    *json = parsed.dump();
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace

std::string new_anthropic_id() {
  std::stringstream ss;
  ss << "anthropiccmpl-" << short_uuid.random();
  return ss.str();
}

AnthropicAdaptResult parse_anthropic_json(
    std::string json,
    xllm::proto::AnthropicMessagesRequest* request) {
  auto [preprocess_status, processed_json] =
      xllm::ChatJsonParser::anthropic().preprocess(std::move(json));
  if (!preprocess_status.ok()) {
    return error_result(preprocess_status.message());
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      processed_json, request, options);
  if (!status.ok()) {
    return error_result(status.ToString());
  }
  return ok_result();
}

AnthropicAdaptResult fill_chat_req(
    const xllm::proto::AnthropicMessagesRequest& anthropic_request,
    xllm::proto::ChatRequest* chat_request,
    ChatMessages* messages) {
  if (anthropic_request.max_tokens() < 0) {
    return error_result("Anthropic max_tokens must be non-negative.");
  }
  if (anthropic_request.messages_size() == 0) {
    return error_result("Messages is empty!");
  }

  chat_request->Clear();
  messages->clear();
  fill_generation_params(anthropic_request, chat_request);
  auto tool_result = add_tool_defs(anthropic_request, chat_request);
  if (!tool_result.ok) {
    return tool_result;
  }
  chat_request->set_tool_choice(tool_choice(anthropic_request));

  auto system_result = add_system_msg(anthropic_request, messages);
  if (!system_result.ok) {
    return system_result;
  }
  std::unordered_set<std::string> skipped_tool_use_ids;
  for (const auto& message : anthropic_request.messages()) {
    auto content_result =
        add_content_msg(message, messages, &skipped_tool_use_ids);
    if (!content_result.ok) {
      return content_result;
    }
  }
  if (messages->empty()) {
    return error_result("Messages is empty!");
  }

  for (const auto& message : *messages) {
    to_proto(message, chat_request->add_messages());
  }
  return ok_result();
}

AnthropicAdaptResult fill_anthropic_resp(
    const std::string& model,
    const llm::RequestOutput& request_output,
    xllm::proto::AnthropicMessagesResponse* response,
    const google::protobuf::RepeatedPtrField<xllm::proto::ToolCall>*
        tool_calls) {
  response->Clear();
  response->set_id(request_output.request_id);
  response->set_type("message");
  response->set_role("assistant");
  response->set_model(model);
  fill_usage(request_output, response);

  if (request_output.outputs.empty()) {
    return ok_result();
  }

  const auto& output = request_output.outputs.front();
  if (output.finish_reason.has_value()) {
    response->set_stop_reason(
        xllm::api_service::convert_finish_reason_to_anthropic(
            output.finish_reason.value()));
  }

  if (!output.text.empty() || tool_calls == nullptr || tool_calls->empty()) {
    auto* text_block = response->add_content();
    text_block->set_type("text");
    text_block->set_text(output.text);
  }
  if (tool_calls == nullptr) {
    return ok_result();
  }
  for (const auto& tool_call : *tool_calls) {
    auto* tool_block = response->add_content();
    tool_block->set_type("tool_use");
    tool_block->set_id(tool_call.id());
    tool_block->set_name(tool_call.function().name());
    if (!tool_call.function().arguments().empty()) {
      auto status = google::protobuf::util::JsonStringToMessage(
          tool_call.function().arguments(), tool_block->mutable_input());
      if (!status.ok()) {
        return error_result("Invalid Anthropic tool call arguments JSON: " +
                            status.ToString());
      }
    }
  }
  return ok_result();
}

bool anthropic_json(const xllm::proto::AnthropicMessagesResponse& response,
                    std::string* json,
                    std::string* error) {
  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  return xllm::api_service::proto_to_anthropic_json(
      response, options, json, error);
}

bool anthropic_json(const xllm::proto::AnthropicMessagesResponse& response,
                    const std::optional<std::string>& thinking,
                    std::string* json,
                    std::string* error) {
  if (!anthropic_json(response, json, error)) {
    return false;
  }
  if (!thinking.has_value() || thinking->empty()) {
    return true;
  }

  try {
    nlohmann::json parsed = nlohmann::json::parse(*json);
    if (!parsed.contains("content") || !parsed["content"].is_array()) {
      parsed["content"] = nlohmann::json::array();
    }
    nlohmann::json thinking_block = {{"type", "thinking"},
                                     {"thinking", thinking.value()},
                                     {"signature", new_thinking_sig()}};
    parsed["content"].insert(parsed["content"].begin(),
                             std::move(thinking_block));
    *json = parsed.dump();
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool anthropic_event_sse(const xllm::proto::AnthropicStreamEvent& event,
                         std::string* sse,
                         std::string* error) {
  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;

  std::string json;
  if (!xllm::api_service::proto_to_anthropic_json(
          event, options, &json, error)) {
    return false;
  }
  if (!normalize_stream_event_json(event, &json, error)) {
    return false;
  }
  *sse = "event: " + event.type() + "\ndata: " + json + "\n\n";
  return true;
}

std::string anthropic_done_sse() { return "data: [DONE]\n\n"; }

}  // namespace xllm_service
