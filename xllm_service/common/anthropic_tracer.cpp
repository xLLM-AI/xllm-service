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

#include "common/anthropic_tracer.h"

#include <glog/logging.h>

#include <utility>

#include "common/global_gflags.h"

namespace xllm_service {

AnthropicTracer::AnthropicTracer(Sink sink,
                                 std::string request_id,
                                 std::string service_request_id)
    : sink_(std::move(sink)),
      request_id_(std::move(request_id)),
      service_request_id_(std::move(service_request_id)) {}

void AnthropicTracer::trace(const std::string& label,
                            const std::string& data) const {
  if (!FLAGS_enable_request_trace) {
    return;
  }
  if (!sink_) {
    return;
  }
  std::string formatted = "[anthropic][" + label + "] " + data;
  sink_(formatted);
  LOG(INFO) << "anthropic trace request_id=" << request_id_
            << " service_request_id=" << service_request_id_
            << " label=" << label << " data=" << data;
}

}  // namespace xllm_service
