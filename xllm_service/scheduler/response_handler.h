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

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"

namespace xllm {
class StreamOutputParser;
}

namespace xllm_service {

class AnthropicStreamEncoder;

struct ChatStreamParseState {
  std::unordered_set<size_t> first_message_sent;
  std::shared_ptr<xllm::StreamOutputParser> stream_parser;
};

class ResponseHandler final {
 public:
  ResponseHandler() = default;
  ~ResponseHandler() = default;

  std::shared_ptr<ChatStreamParseState> create_chat_stream_parse_state(
      const std::vector<JsonTool>& tools,
      const std::string& model,
      const std::string& tool_call_parser = "",
      const std::string& reasoning_parser = "",
      bool force_reasoning = false);

  bool send_delta_to_client(
      std::shared_ptr<ChatCallData> call_data,
      bool include_usage,
      int64_t created_time,
      const std::string& model,
      const llm::RequestOutput& output,
      std::shared_ptr<ChatStreamParseState> stream_state = nullptr);
  bool send_result_to_client(std::shared_ptr<ChatCallData> call_data,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output,
                             const std::vector<JsonTool>& tools = {},
                             const std::string& tool_call_parser = "",
                             const std::string& reasoning_parser = "",
                             bool force_reasoning = false);

  bool send_delta_to_client(std::shared_ptr<CompletionCallData> call_data,
                            bool include_usage,
                            int64_t created_time,
                            const std::string& model,
                            const llm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<CompletionCallData> call_data,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output);

  bool send_delta_to_client(
      std::shared_ptr<AnthropicCallData> call_data,
      const std::string& model,
      const llm::RequestOutput& output,
      AnthropicStreamEncoder& encoder,
      std::shared_ptr<xllm::StreamOutputParser> stream_parser = nullptr);
  bool send_result_to_client(std::shared_ptr<AnthropicCallData> call_data,
                             const std::string& model,
                             const llm::RequestOutput& req_output,
                             const std::vector<JsonTool>& tools = {},
                             const std::string& tool_call_parser = "",
                             const std::string& reasoning_parser = "",
                             bool force_reasoning = false);
};

}  // namespace xllm_service
