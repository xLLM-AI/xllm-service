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
#include <fstream>
#include <mutex>
#include <string>

namespace xllm_service {

class RequestTracer {
 public:
  RequestTracer(bool enable_request_trace);
  ~RequestTracer() = default;
  RequestTracer(const RequestTracer&) = delete;
  RequestTracer& operator=(const RequestTracer&) = delete;
  RequestTracer(RequestTracer&&) = delete;
  RequestTracer& operator=(RequestTracer&&) = delete;
  void log(const std::string& service_request_id,
           const std::string& input_or_output);

 private:
  std::ofstream log_stream_;
  std::mutex mutex_;
  bool enable_request_trace_ = false;
};
}  // namespace xllm_service