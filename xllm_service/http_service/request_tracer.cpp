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

#include "http_service/request_tracer.h"

#include <glog/logging.h>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>

namespace {
std::string get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}
}  // namespace

namespace xllm_service {

RequestTracer::RequestTracer(bool enable_request_trace)
    : enable_request_trace_(enable_request_trace) {
  if (!enable_request_trace_) return;
  std::filesystem::create_directories("trace");
  log_stream_.open("trace/trace.json", std::ios::app);
  if (!log_stream_.is_open()) {
    LOG(ERROR) << "Failed to open log file: trace/trace.json";
  }
}

void RequestTracer::log(const std::string& service_request_id,
                        const std::string& input_or_output) {
  if (!enable_request_trace_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  std::string timestamp = get_current_timestamp();

  nlohmann::json j;
  j["timestamp"] = timestamp;
  j["service_request_id"] = service_request_id;
  j["data"] = input_or_output;

  log_stream_ << j.dump() << "\n";
}
}  // namespace xllm_service