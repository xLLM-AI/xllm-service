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

#include <brpc/server.h>
#include <gtest/gtest.h>

#include <atomic>

#include "disagg_pd.pb.h"
#include "tests/xllm_service/service_test_peer.h"

namespace xllm_service {
namespace {
InstanceMetaInfo instance(std::string name,
                          std::string incarnation,
                          InstanceType type = InstanceType::DEFAULT) {
  InstanceMetaInfo info;
  info.name = std::move(name);
  info.rpc_address = info.name;
  info.incarnation_id = std::move(incarnation);
  info.type = type;
  info.dp_size = 1;
  info.kv_split_size = 1;
  return info;
}

TEST(InstanceStateTest, SnapshotsAreStableAndReplacementDoesNotInheritLoad) {
  InstanceState state;
  ASSERT_TRUE(state.add_instance(instance("node", "old")));
  const auto before = state.snapshot();
  ASSERT_TRUE(state.record_heartbeat(
      "node", "old", LoadMetrics(4, 0.5), LatencyMetrics(2, 3)));
  EXPECT_FALSE(state.record_heartbeat(
      "node", "stale", LoadMetrics(9, 0.9), LatencyMetrics(0, 0)));
  ASSERT_TRUE(state.snapshot()[0].load.has_value());
  EXPECT_EQ(state.snapshot()[0].load->waiting_requests_num, 4);
  EXPECT_FALSE(before[0].load.has_value());
  InstanceMetaInfo removed;
  EXPECT_FALSE(state.remove_instance("node", "stale", &removed));
  ASSERT_TRUE(state.remove_instance("node", "old", &removed));
  ASSERT_TRUE(state.add_instance(instance("node", "new")));
  EXPECT_FALSE(state.snapshot()[0].load.has_value());
  ASSERT_TRUE(state.record_heartbeat(
      "node", "new", LoadMetrics(1, 0.2), LatencyMetrics(0, 0)));
  const auto updates = state.take_load_updates();
  ASSERT_EQ(updates.changed.size(), 1);
  EXPECT_EQ(updates.changed.at("node").waiting_requests_num, 1);
  EXPECT_TRUE(updates.removed.empty());
  EXPECT_EQ(before[0].info->incarnation_id, "old");
  EXPECT_EQ(state.get_instance_info("node").incarnation_id, "new");
}

TEST(InstanceClientTest, StaleDisconnectCannotRemoveReplacementConnection) {
  ServiceConfig options;
  InstanceClient client(options);
  auto info = instance("127.0.0.1:1", "old");
  ASSERT_TRUE(client.connect_instance(info));
  auto retained = client.get_channel(info.name, "old");
  ASSERT_NE(retained, nullptr);
  EXPECT_EQ(client.get_channel(info.name, "new"), nullptr);
  client.disconnect_instance(info.name, "old");
  info.incarnation_id = "new";
  ASSERT_TRUE(client.connect_instance(info));
  client.disconnect_instance(info.name, "old");
  EXPECT_NE(client.get_channel(info.name, "new"), nullptr);
  EXPECT_NE(retained, nullptr);
}

class DecodePeer final : public xllm::proto::DisaggPDService {
 public:
  void LinkInstance(google::protobuf::RpcController* /*controller*/,
                    const xllm::proto::InstanceClusterInfo* request,
                    xllm::proto::Status* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    EXPECT_EQ(request->instance_name(), "127.0.0.1:1");
    ++links;
    response->set_ok(true);
  }
  void UnlinkInstance(google::protobuf::RpcController* /*controller*/,
                      const xllm::proto::InstanceClusterInfo* request,
                      xllm::proto::Status* response,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    EXPECT_EQ(request->instance_name(), "127.0.0.1:1");
    ++unlinks;
    response->set_ok(true);
  }
  std::atomic<int32_t> links{0};
  std::atomic<int32_t> unlinks{0};
};

TEST(InstanceControllerTest, LinksBeforePublishingAndUnlinksOnRetirement) {
  ServiceTestContext context;
  DecodePeer peer;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&peer, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const auto prefill = instance("127.0.0.1:1", "p", InstanceType::PREFILL);
  const std::string address =
      butil::endpoint2str(server.listen_address()).c_str();
  const auto decode = instance(address, "d", InstanceType::DECODE);
  context.controller->on_instance_event(
      InstanceEvent{prefill.name, prefill, false});
  EXPECT_FALSE(context.instances->has_available_instances());
  context.controller->on_instance_event(
      InstanceEvent{decode.name, decode, false});
  EXPECT_EQ(peer.links.load(), 1);
  EXPECT_TRUE(context.instances->has_available_instances());
  EXPECT_NE(context.client->get_channel(decode.name, "d"), nullptr);
  ServiceTestPeer::remove(context, prefill.name);
  EXPECT_EQ(peer.unlinks.load(), 1);
  EXPECT_FALSE(context.instances->find_instance(prefill.name).has_value());
  EXPECT_EQ(context.client->get_channel(prefill.name, "p"), nullptr);
  server.Stop(0);
  server.Join();
}
}  // namespace
}  // namespace xllm_service
