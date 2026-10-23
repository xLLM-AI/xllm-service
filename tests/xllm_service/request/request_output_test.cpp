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

#include "request/request_output.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace xllm_service {
namespace {

llm::RequestOutput cancellation() {
  llm::RequestOutput output;
  output.status = llm::Status(llm::StatusCode::CANCELLED, "instance failed");
  return output;
}

TEST(RequestOutputTest, ErrorTerminatesExactlyOnce) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return true;
  };
  EXPECT_EQ(deliver_request_output(request, cancellation()), true);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_FALSE(
      deliver_request_output(request, llm::RequestOutput{}).has_value());
  EXPECT_EQ(callbacks, 1);
}

TEST(RequestOutputTest, SuccessfulTerminalOutputSuppressesLaterCancellation) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return true;
  };
  llm::RequestOutput output;
  output.finished = true;
  EXPECT_EQ(deliver_request_output(request, output), false);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_EQ(callbacks, 1);
}

TEST(RequestOutputTest, CallbackFailureSuppressesRemainingOutputs) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return false;
  };
  EXPECT_EQ(deliver_request_output(request, llm::RequestOutput{}), true);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_EQ(callbacks, 1);
}

}  // namespace
}  // namespace xllm_service
