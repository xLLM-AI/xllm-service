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

#include "common/call_data.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm_service {
namespace {

class TestClosure : public google::protobuf::Closure {
 public:
  void Run() override { ran = true; }
  bool ran = false;
};

TEST(CallDataTest, NonStreamWriteAndFinishTracesAttachment) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  std::vector<std::string> trace_chunks;

  {
    ChatCallData call_data(
        &controller,
        /*stream=*/false,
        &done,
        &request,
        &response,
        [&](const std::string& chunk) { trace_chunks.push_back(chunk); });

    ASSERT_FALSE(done.ran);
    ASSERT_TRUE(call_data.write_and_finish("{\"id\":\"chatcmpl-test\"}"));

    ASSERT_EQ(trace_chunks.size(), 1);
    EXPECT_EQ(trace_chunks[0], "{\"id\":\"chatcmpl-test\"}");
    EXPECT_EQ(controller.response_attachment().to_string(),
              "{\"id\":\"chatcmpl-test\"}");
  }

  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, NonStreamFinishWithErrorMarksControllerFailed) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);

    ASSERT_TRUE(call_data.finish_with_error("encode failed"));
    EXPECT_TRUE(controller.Failed());
    EXPECT_EQ(controller.ErrorText(), "encode failed");
  }

  EXPECT_TRUE(done.ran);
}

}  // namespace
}  // namespace xllm_service
