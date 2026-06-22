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

#include <functional>
#include <string>

namespace xllm_service {

// Centralizes anthropic request-trace formatting, the gflag guard and the LOG
// context that used to be duplicated across the http_service and scheduler
// layers. The sink is injected as an adapter: production callers wire it to the
// per-request trace stream (CallData::trace / Request::trace_callback), tests
// inject a recording sink.
class AnthropicTracer {
 public:
  // Receives the already-formatted trace string and forwards it to the client
  // trace stream.
  using Sink = std::function<void(const std::string& formatted)>;

  AnthropicTracer(Sink sink,
                  std::string request_id,
                  std::string service_request_id);

  // Guarded by FLAGS_enable_request_trace. Formats "[anthropic][label] data",
  // forwards it to the sink and emits a LOG line in one step. No-op when the
  // flag is disabled or no sink was provided.
  void trace(const std::string& label, const std::string& data) const;

 private:
  Sink sink_;
  std::string request_id_;
  std::string service_request_id_;
};

}  // namespace xllm_service
