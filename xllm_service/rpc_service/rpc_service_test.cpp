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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "rpc_service/service.h"

namespace xllm_service::test {

class XllmRpcServiceTest : public ::testing::Test {
 protected:
  void SetUp() override { google::InitGoogleLogging("XllmRpcServiceTest"); }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

TEST_F(XllmRpcServiceTest, OutputUsageUsesFieldFourForCachedTokens) {
  const google::protobuf::FieldDescriptor* field =
      proto::OutputUsage::descriptor()->FindFieldByName("num_cached_tokens");

  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->number(), 4);
}

TEST_F(XllmRpcServiceTest, ConvertsCachedTokensIntoInternalUsage) {
  proto::DisaggStreamGeneration generation;
  generation.mutable_usage()->set_num_prompt_tokens(8);
  generation.mutable_usage()->set_num_generated_tokens(2);
  generation.mutable_usage()->set_num_total_tokens(10);
  generation.mutable_usage()->set_num_cached_tokens(6);

  llm::RequestOutput output = make_request_output(generation);

  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_cached_tokens, 6u);
}

// TODO
// TEST_F(XllmRpcServiceTest, RegisterInstance) {
//   RpcServiceConfig config;
//   HttpServiceConfig http_config;
//   ModelConfig model_config;
//   auto xllm_service =
//       std::make_shared<XllmRpcServiceImpl>(config, model_config,
//       http_config);
//   std::string inst_name = "127.0.0.1@nic0";
//   InstanceMetaInfo metainfo(inst_name, "127.0.0.1:7777",
//   InstanceType::PREFILL); EXPECT_EQ(ErrorCode::OK,
//             xllm_service->register_instance(inst_name, metainfo));

//   metainfo.type = InstanceType::DECODE;
//   EXPECT_EQ(ErrorCode::INSTANCE_EXISTED,
//             xllm_service->register_instance(inst_name, metainfo));
// }

// TEST_F(XllmRpcServiceTest, UpdateInstanceMetainfo) {
//   RpcServiceConfig config;
//   HttpServiceConfig http_config;
//   ModelConfig model_config;
//   auto xllm_service =
//       std::make_shared<XllmRpcServiceImpl>(config, model_config,
//       http_config);
//   std::string inst_name = "127.0.0.1@nic0";
//   InstanceMetaInfo metainfo(inst_name, "127.0.0.1:7777",
//   InstanceType::PREFILL); EXPECT_EQ(ErrorCode::OK,
//             xllm_service->register_instance(inst_name, metainfo));
//   metainfo.type = InstanceType::DECODE;
//   EXPECT_EQ(ErrorCode::OK,
//             xllm_service->update_instance_metainfo(inst_name, metainfo));

//   std::string inst_name2 = "127.0.0.1@nic2";
//   InstanceMetaInfo metainfo2(
//       inst_name2, "127.0.0.1:7778", InstanceType::PREFILL);
//   EXPECT_EQ(ErrorCode::INSTANCE_NOT_EXISTED,
//             xllm_service->update_instance_metainfo(inst_name2, metainfo));
// }

}  // namespace xllm_service::test
