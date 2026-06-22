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

#include "common/types.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm_service {
namespace {

// A fully populated instance, used to exercise the serialize -> parse path.
InstanceMetaInfo make_full_instance() {
  InstanceMetaInfo info;
  info.name = "instance-0";
  info.rpc_address = "10.0.0.1:9000";
  info.incarnation_id = "inc-1";
  info.register_ts_ms = 1234567890;
  info.type = InstanceType::PREFILL;
  info.cluster_ids = {1, 2, 3};
  info.addrs = {"10.0.0.1:1000", "10.0.0.1:1001"};
  info.dp_size = 4;
  info.kv_split_size = 2;
  info.ports = {1000, 1001};
  info.ttft_profiling_data = {{10, 1.5}};
  info.tpot_profiling_data = {{20, 30, 2.5}};
  info.backend_type = "vllm";
  return info;
}

TEST(InstanceMetaInfoTest, SerializeParseRoundtripPreservesFields) {
  InstanceMetaInfo src = make_full_instance();

  InstanceMetaInfo dst;
  ASSERT_TRUE(dst.parse_from_json(src.serialize_to_json().dump()));

  EXPECT_EQ(dst.name, src.name);
  EXPECT_EQ(dst.rpc_address, src.rpc_address);
  EXPECT_EQ(dst.incarnation_id, src.incarnation_id);
  EXPECT_EQ(dst.register_ts_ms, src.register_ts_ms);
  EXPECT_EQ(dst.type, src.type);
  EXPECT_EQ(dst.cluster_ids, src.cluster_ids);
  EXPECT_EQ(dst.addrs, src.addrs);
  EXPECT_EQ(dst.dp_size, src.dp_size);
  EXPECT_EQ(dst.kv_split_size, src.kv_split_size);
  EXPECT_EQ(dst.ports, src.ports);
  EXPECT_EQ(dst.ttft_profiling_data, src.ttft_profiling_data);
  EXPECT_EQ(dst.tpot_profiling_data, src.tpot_profiling_data);
  EXPECT_EQ(dst.backend_type, src.backend_type);
}

TEST(InstanceMetaInfoTest, SerializeIncludesBackendType) {
  InstanceMetaInfo info = make_full_instance();

  nlohmann::json json_val = info.serialize_to_json();
  ASSERT_TRUE(json_val.contains("backend_type"));
  EXPECT_EQ(json_val["backend_type"].get<std::string>(), "vllm");
}

TEST(InstanceMetaInfoTest, ParseDefaultsBackendTypeToXllmWhenAbsent) {
  const std::string json_str = R"({"name":"i1","rpc_address":"addr","type":0})";

  InstanceMetaInfo info;
  ASSERT_TRUE(info.parse_from_json(json_str));
  EXPECT_EQ(info.backend_type, "xllm");
}

TEST(InstanceMetaInfoTest, ParseAcceptsVllmBackendType) {
  const std::string json_str =
      R"({"name":"i1","rpc_address":"addr","type":0,"backend_type":"vllm"})";

  InstanceMetaInfo info;
  ASSERT_TRUE(info.parse_from_json(json_str));
  EXPECT_EQ(info.backend_type, "vllm");
}

TEST(InstanceMetaInfoTest, ParseDefaultsKvSplitSizeWhenAbsent) {
  const std::string json_str = R"({"name":"i1","rpc_address":"addr","type":0})";

  InstanceMetaInfo info;
  ASSERT_TRUE(info.parse_from_json(json_str));
  EXPECT_EQ(info.kv_split_size, 1);
}

// Minimal payloads (e.g. from a vLLM sidecar) carry neither profiling block;
// parsing must tolerate their absence instead of throwing.
TEST(InstanceMetaInfoTest, ParseToleratesMissingProfilingData) {
  const std::string json_str = R"({"name":"i1","rpc_address":"addr","type":0})";

  InstanceMetaInfo info;
  ASSERT_TRUE(info.parse_from_json(json_str));
  EXPECT_TRUE(info.ttft_profiling_data.empty());
  EXPECT_TRUE(info.tpot_profiling_data.empty());
}

TEST(InstanceMetaInfoTest, ParseDefaultsOptionalVectorsWhenAbsent) {
  const std::string json_str = R"({"name":"i1","rpc_address":"addr","type":0})";

  InstanceMetaInfo info;
  ASSERT_TRUE(info.parse_from_json(json_str));
  EXPECT_TRUE(info.cluster_ids.empty());
  EXPECT_TRUE(info.addrs.empty());
  EXPECT_TRUE(info.ports.empty());
  EXPECT_EQ(info.dp_size, 0);
}

TEST(InstanceMetaInfoTest, ParseReturnsFalseOnMalformedJson) {
  InstanceMetaInfo info;
  EXPECT_FALSE(info.parse_from_json("not-json"));
}

TEST(InstanceMetaInfoTest, ParseReturnsFalseWhenRequiredFieldMissing) {
  // "rpc_address" is required (parsed with .at); its absence must fail parsing.
  const std::string json_str = R"({"name":"i1","type":0})";

  InstanceMetaInfo info;
  EXPECT_FALSE(info.parse_from_json(json_str));
}

}  // namespace
}  // namespace xllm_service
