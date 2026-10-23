/* Copyright 2026 The xLLM Authors.

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

#include <optional>

#include "request/request.h"

namespace xllm_service {

// Runs on the request's output thread, including instance-failure cancellation.
// A value requests one final cleanup; true denotes an error termination.
inline std::optional<bool> deliver_request_output(
    Request& request,
    const llm::RequestOutput& output) {
  if (request.output_finished) {
    return std::nullopt;
  }
  const bool status_error = output.status.has_value() && !output.status->ok();
  const bool callback_ok = request.output_callback(output);
  if (!callback_ok || status_error || output.finished) {
    request.output_finished = true;
    return !callback_ok || status_error;
  }
  return std::nullopt;
}

}  // namespace xllm_service
