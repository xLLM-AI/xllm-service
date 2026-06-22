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

#include "http_service/anthropic_stream_encoder.h"

#include <nlohmann/json.hpp>
#include <utility>

#include "api_service/anthropic_stream_utils.h"
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

std::string json_sse(const std::string& event_type,
                     const nlohmann::json& event) {
  return "event: " + event_type + "\ndata: " + event.dump() + "\n\n";
}

bool append_proto_sse(
    const std::vector<xllm::proto::AnthropicStreamEvent>& events,
    std::vector<std::string>* sse_events,
    std::string* error) {
  for (const auto& event : events) {
    std::string sse;
    if (!anthropic_event_sse(event, &sse, error)) {
      return false;
    }
    sse_events->push_back(std::move(sse));
  }
  return true;
}

}  // namespace

AnthropicStreamEncoder::AnthropicStreamEncoder(std::string model)
    : model_(std::move(model)) {}

void AnthropicStreamEncoder::add_message_start(
    const llm::RequestOutput& request_output,
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  if (state_.message_started) {
    return;
  }

  xllm::proto::AnthropicStreamEvent event;
  event.set_type("message_start");
  auto* message = event.mutable_message();
  message->set_id(request_output.request_id);
  message->set_type("message");
  message->set_role("assistant");
  message->set_model(model_);
  auto* usage = message->mutable_usage();
  usage->set_input_tokens(0);
  usage->set_output_tokens(0);

  events->push_back(std::move(event));
  state_.message_started = true;
}

void AnthropicStreamEncoder::add_sig_delta(
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  xllm::proto::AnthropicStreamEvent event;
  event.set_type("content_block_delta");
  event.set_index(state_.content_block_index);
  auto* delta = event.mutable_delta();
  delta->set_type("signature_delta");
  delta->set_signature(state_.thinking_signature);
  events->push_back(std::move(event));
}

void AnthropicStreamEncoder::add_block_stop(
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  if (state_.content_block_index < 0) {
    return;
  }

  if (state_.last_content_block_type == "thinking") {
    if (state_.thinking_signature.empty()) {
      state_.thinking_signature = new_thinking_sig();
    }
    add_sig_delta(events);
  }

  xllm::proto::AnthropicStreamEvent event;
  event.set_type("content_block_stop");
  event.set_index(state_.content_block_index);
  events->push_back(std::move(event));
  state_.thinking_signature.clear();
}

void AnthropicStreamEncoder::start_text_block(
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  if (state_.last_content_block_type == "text") {
    return;
  }
  if (!state_.last_content_block_type.empty()) {
    add_block_stop(events);
  }

  state_.last_content_block_type = "text";
  ++state_.content_block_index;

  xllm::proto::AnthropicStreamEvent event;
  event.set_type("content_block_start");
  event.set_index(state_.content_block_index);
  auto* content_block = event.mutable_content_block();
  content_block->set_type("text");
  content_block->set_text("");
  events->push_back(std::move(event));
}

void AnthropicStreamEncoder::start_tool_block(
    const std::string& tool_call_id,
    const std::string& function_name,
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  if (!state_.last_content_block_type.empty()) {
    add_block_stop(events);
  }

  state_.last_content_block_type = "tool_use";
  ++state_.content_block_index;

  xllm::proto::AnthropicStreamEvent event;
  event.set_type("content_block_start");
  event.set_index(state_.content_block_index);
  auto* content_block = event.mutable_content_block();
  content_block->set_type("tool_use");
  if (!tool_call_id.empty()) {
    content_block->set_id(tool_call_id);
  }
  if (!function_name.empty()) {
    content_block->set_name(function_name);
  }
  content_block->mutable_input();
  events->push_back(std::move(event));
}

void AnthropicStreamEncoder::add_text_delta(
    const std::string& text,
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  xllm::proto::AnthropicStreamEvent event;
  event.set_type("content_block_delta");
  event.set_index(state_.content_block_index);
  auto* delta = event.mutable_delta();
  delta->set_type("text_delta");
  delta->set_text(text);
  events->push_back(std::move(event));
}

void AnthropicStreamEncoder::start_thinking_block(
    std::vector<std::string>* sse_events) {
  if (state_.last_content_block_type == "thinking") {
    return;
  }

  state_.last_content_block_type = "thinking";
  state_.thinking_signature = new_thinking_sig();
  ++state_.content_block_index;

  nlohmann::json event = {
      {"type", "content_block_start"},
      {"index", state_.content_block_index},
      {"content_block", {{"type", "thinking"}, {"thinking", ""}}}};
  sse_events->push_back(json_sse("content_block_start", event));
}

void AnthropicStreamEncoder::add_thinking_delta(
    const std::string& thinking,
    std::vector<std::string>* sse_events) {
  nlohmann::json event = {
      {"type", "content_block_delta"},
      {"index", state_.content_block_index},
      {"delta", {{"type", "thinking_delta"}, {"thinking", thinking}}}};
  sse_events->push_back(json_sse("content_block_delta", event));
}

void AnthropicStreamEncoder::add_message_delta(
    const llm::RequestOutput& request_output,
    const std::string& finish_reason,
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  xllm::proto::AnthropicStreamEvent event;
  event.set_type("message_delta");
  auto* delta = event.mutable_delta();
  delta->set_stop_reason(xllm::api_service::get_stream_stop_reason(
      true, state_.has_tool_call, finish_reason));

  auto* usage = event.mutable_usage();
  if (request_output.usage.has_value()) {
    const auto& source_usage = request_output.usage.value();
    usage->set_input_tokens(
        static_cast<int32_t>(source_usage.num_prompt_tokens));
    usage->set_output_tokens(
        static_cast<int32_t>(source_usage.num_generated_tokens));
  } else {
    usage->set_input_tokens(0);
    usage->set_output_tokens(0);
  }
  events->push_back(std::move(event));
}

void AnthropicStreamEncoder::add_message_stop(
    std::vector<xllm::proto::AnthropicStreamEvent>* events) {
  xllm::proto::AnthropicStreamEvent event;
  event.set_type("message_stop");
  events->push_back(std::move(event));
}

AnthropicAdaptResult AnthropicStreamEncoder::on_text(
    const llm::RequestOutput& output,
    const std::string& text,
    std::vector<std::string>* out) {
  if (text.empty()) {
    return ok_result();
  }

  std::vector<xllm::proto::AnthropicStreamEvent> events;
  add_message_start(output, &events);
  start_text_block(&events);
  add_text_delta(text, &events);

  std::string error;
  if (!append_proto_sse(events, out, &error)) {
    return error_result(error);
  }
  return ok_result();
}

AnthropicAdaptResult AnthropicStreamEncoder::on_reasoning(
    const llm::RequestOutput& output,
    const std::string& thinking,
    std::vector<std::string>* out) {
  if (thinking.empty()) {
    return ok_result();
  }

  std::vector<xllm::proto::AnthropicStreamEvent> prefix_events;
  add_message_start(output, &prefix_events);
  if (!state_.last_content_block_type.empty() &&
      state_.last_content_block_type != "thinking") {
    add_block_stop(&prefix_events);
  }

  std::string error;
  if (!append_proto_sse(prefix_events, out, &error)) {
    return error_result(error);
  }

  start_thinking_block(out);
  add_thinking_delta(thinking, out);
  return ok_result();
}

AnthropicAdaptResult AnthropicStreamEncoder::on_tool(
    const llm::RequestOutput& output,
    const std::string& tool_call_id,
    const std::string& function_name,
    const std::string& arguments,
    std::vector<std::string>* out) {
  const bool starts_new_call = !function_name.empty();
  if ((state_.last_content_block_type != "tool_use" || starts_new_call) &&
      (tool_call_id.empty() || function_name.empty())) {
    return error_result("Anthropic tool_use id and name are required.");
  }

  std::vector<xllm::proto::AnthropicStreamEvent> events;
  add_message_start(output, &events);
  state_.has_tool_call = true;
  if (state_.last_content_block_type != "tool_use" || starts_new_call) {
    start_tool_block(tool_call_id, function_name, &events);
  }

  auto event = xllm::api_service::make_input_json_delta_event(
      state_.content_block_index, arguments);
  if (event.has_value()) {
    events.push_back(std::move(event.value()));
  }

  std::string error;
  if (!append_proto_sse(events, out, &error)) {
    return error_result(error);
  }
  return ok_result();
}

AnthropicAdaptResult AnthropicStreamEncoder::finish(
    const llm::RequestOutput& output,
    std::vector<std::string>* out) {
  std::vector<xllm::proto::AnthropicStreamEvent> events;
  add_message_start(output, &events);
  if (state_.content_block_index >= 0) {
    add_block_stop(&events);
    state_.last_content_block_type.clear();
  }

  std::string finish_reason;
  for (const auto& seq_output : output.outputs) {
    if (seq_output.finish_reason.has_value()) {
      finish_reason = seq_output.finish_reason.value();
    }
  }
  add_message_delta(output, finish_reason, &events);
  add_message_stop(&events);

  std::string error;
  if (!append_proto_sse(events, out, &error)) {
    return error_result(error);
  }
  return ok_result();
}

}  // namespace xllm_service
