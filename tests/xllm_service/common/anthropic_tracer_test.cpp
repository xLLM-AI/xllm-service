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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/global_gflags.h"

namespace xllm_service {

TEST(AnthropicTracerTest, FormatsTraceWhenEnabled) {
  FLAGS_enable_request_trace = true;
  std::vector<std::string> received;
  AnthropicTracer tracer(
      [&](const std::string& formatted) { received.push_back(formatted); },
      "req-1",
      "svc-1");

  tracer.trace("stream_sse", "hello");

  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received[0], "[anthropic][stream_sse] hello");
}

TEST(AnthropicTracerTest, DoesNotCallSinkWhenDisabled) {
  FLAGS_enable_request_trace = false;
  std::vector<std::string> received;
  AnthropicTracer tracer(
      [&](const std::string& formatted) { received.push_back(formatted); },
      "req-1",
      "svc-1");

  tracer.trace("stream_sse", "hello");

  EXPECT_TRUE(received.empty());
}

TEST(AnthropicTracerTest, NoCrashWhenSinkIsEmpty) {
  FLAGS_enable_request_trace = true;
  AnthropicTracer tracer(/*sink=*/nullptr, "req-1", "svc-1");

  // Mirrors the service.cpp path where request == nullptr or the trace
  // callback is unset: the call must be a safe no-op.
  EXPECT_NO_FATAL_FAILURE(tracer.trace("raw_http_request", "payload"));
}

}  // namespace xllm_service
