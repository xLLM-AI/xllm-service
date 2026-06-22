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
#include <functional>
#include <string>
#include <vector>

#include "common/xllm/output.h"
#include "http_service/anthropic_adapter.h"

namespace xllm_service {

// Translates a stream of parsed model deltas (reasoning / plain text / tool
// arguments) into Anthropic Messages SSE events. The encoder owns the whole
// content-block state machine (block index, block-type transitions, thinking
// signature, pending tool call); each `on_*` / `finish` call appends 0..N
// ready-to-write SSE strings to `out` and returns ok/err.
//
// Splitting a model chunk into reasoning vs tool vs plain text stays in the
// orchestration layer; the encoder only consumes already-split deltas and is
// therefore pure compute (no I/O, no stream_parser dependency).
class AnthropicStreamEncoder {
 public:
  explicit AnthropicStreamEncoder(std::string model);

  AnthropicAdaptResult on_reasoning(const llm::RequestOutput& output,
                                    const std::string& thinking,
                                    std::vector<std::string>* out);

  AnthropicAdaptResult on_text(const llm::RequestOutput& output,
                               const std::string& text,
                               std::vector<std::string>* out);

  AnthropicAdaptResult on_tool(const llm::RequestOutput& output,
                               const std::string& tool_call_id,
                               const std::string& function_name,
                               const std::string& arguments,
                               std::vector<std::string>* out);

  AnthropicAdaptResult finish(const llm::RequestOutput& output,
                              std::vector<std::string>* out);

  // State the orchestration layer reads/writes while deciding how to split an
  // incoming chunk (reasoning boundary vs tool boundary).
  bool pending_tool_call() const { return state_.pending_tool_call; }
  void set_pending_tool_call(bool pending) {
    state_.pending_tool_call = pending;
  }
  const std::string& last_content_block_type() const {
    return state_.last_content_block_type;
  }

 private:
  // Content-block state machine (formerly the public AnthropicStreamState,
  // now an internal implementation detail of the encoder).
  struct State {
    bool message_started = false;
    bool has_tool_call = false;
    bool pending_tool_call = false;
    int32_t content_block_index = -1;
    std::string last_content_block_type;
    std::string thinking_signature;
  };

  // Proto-event builders for the text / tool / finish paths.
  void add_message_start(
      const llm::RequestOutput& request_output,
      std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void add_sig_delta(std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void add_block_stop(std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void start_text_block(std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void start_tool_block(const std::string& tool_call_id,
                        const std::string& function_name,
                        std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void add_text_delta(const std::string& text,
                      std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void add_message_delta(
      const llm::RequestOutput& request_output,
      const std::string& finish_reason,
      std::vector<xllm::proto::AnthropicStreamEvent>* events);
  void add_message_stop(std::vector<xllm::proto::AnthropicStreamEvent>* events);

  // Thinking blocks are emitted as raw SSE strings (no proto representation).
  void start_thinking_block(std::vector<std::string>* sse_events);
  void add_thinking_delta(const std::string& thinking,
                          std::vector<std::string>* sse_events);

  std::string model_;
  State state_;
};

}  // namespace xllm_service
