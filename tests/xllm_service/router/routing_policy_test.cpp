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

#include <gtest/gtest.h>

#include "request/request_metrics.h"
#include "router/policies/round_robin.h"
#include "router/policies/slo_aware.h"
#include "router/routing_state.h"
namespace xllm_service {
namespace {
RoutingCandidate candidate(std::string name, InstanceType type) {
  RoutingCandidate value;
  value.name = std::move(name);
  value.type = type;
  return value;
}
TEST(RoutingPolicyTest, RoundRobinKeepsDefaultAndPdRoutesSeparate) {
  RoundRobin policy;
  RoutingInput input;
  input.candidates.prefill = {candidate("P0", InstanceType::PREFILL)};
  RoutingDecision result;
  EXPECT_FALSE(policy.select(input, &result));
  input.candidates.decode = {candidate("D0", InstanceType::DECODE),
                             candidate("D1", InstanceType::DECODE)};
  ASSERT_TRUE(policy.select(input, &result));
  EXPECT_EQ(result.routing.decode_name, "D0");
  ASSERT_TRUE(policy.select(input, &result));
  EXPECT_EQ(result.routing.decode_name, "D1");
  input.candidates.decode.clear();
  input.candidates.prefill[0].type = InstanceType::DEFAULT;
  ASSERT_TRUE(policy.select(input, &result));
  EXPECT_EQ(result.routing.prefill_name, "P0");
  EXPECT_TRUE(result.routing.decode_name.empty());
}
TEST(RoutingPolicyTest, SloSelectionDoesNotMutateItsSnapshot) {
  SloAware policy(1000, 50);
  RoutingInput input;
  input.prompt_tokens = 32;
  input.candidates.prefill = {candidate("P0", InstanceType::PREFILL),
                              candidate("P1", InstanceType::PREFILL)};
  input.candidates.prefill[0].prefill_time = 100;
  input.candidates.prefill[1].prefill_time = 20;
  input.candidates.prefill[1].estimated_ttft = 7;
  input.candidates.decode = {candidate("D0", InstanceType::DECODE),
                             candidate("D1", InstanceType::DECODE)};
  input.candidates.decode[0].estimated_tpot = 60;
  input.candidates.decode[1].estimated_tpot = 40;
  RoutingDecision result;
  ASSERT_TRUE(policy.select(input, &result));
  EXPECT_EQ(result.routing.prefill_name, "P1");
  EXPECT_EQ(result.routing.decode_name, "D1");
  EXPECT_EQ(result.estimated_ttft, 7);
  EXPECT_FALSE(result.flip_prefill_to_decode);
  EXPECT_EQ(input.candidates.prefill[1].prefill_time, 20);
}
TEST(RoutingPolicyTest, SloRequestsRoleChangeOnlyWhenDecodeIsOverloaded) {
  SloAware policy(1000, 50);
  RoutingInput input;
  input.prompt_tokens = 32;
  input.candidates.prefill = {candidate("P0", InstanceType::MIX),
                              candidate("P1", InstanceType::MIX)};
  input.candidates.decode = {candidate("D0", InstanceType::MIX)};
  input.candidates.decode[0].estimated_tpot = 60;
  RoutingDecision result;
  ASSERT_TRUE(policy.select(input, &result));
  EXPECT_TRUE(result.flip_prefill_to_decode);
}
}  // namespace

namespace {
InstanceMetaInfo routing_instance(std::string name, InstanceType type) {
  InstanceMetaInfo result;
  result.name = std::move(name);
  result.incarnation_id = result.name + "-old";
  result.type = type;
  result.dp_size = 1;
  result.kv_split_size = 1;
  return result;
}

TEST(RoutingStateTest, ReservationsAreSeparateFromReportedLoadAndReleasedOnce) {
  auto instances = std::make_shared<InstanceState>();
  ASSERT_TRUE(
      instances->add_instance(routing_instance("P", InstanceType::PREFILL)));
  ASSERT_TRUE(
      instances->add_instance(routing_instance("D", InstanceType::DECODE)));
  ServiceConfig options;
  options.load_balance_policy("SLO_AWARE");
  RoutingState routing(options, instances);
  SloAware policy(1000, 50);
  auto request = std::make_shared<Request>();
  request->token_ids = {1, 2, 3};
  ASSERT_TRUE(routing.select_route(*request, policy, {}));
  EXPECT_EQ(routing.request_metrics("P").prefill_request_num, 1);
  EXPECT_EQ(routing.request_metrics("D").decode_token_num, 3);
  EXPECT_FALSE(instances->snapshot()[0].load.has_value());
  ASSERT_TRUE(record_generation(*request, false));
  routing.update_request_metrics(request, RequestAction::FINISH_PREFILL);
  EXPECT_EQ(routing.request_metrics("P").prefill_request_num, 0);
  routing.update_request_metrics(request, RequestAction::FINISH_DECODE);
  EXPECT_EQ(routing.request_metrics("D").decode_request_num, 0);
  EXPECT_EQ(routing.request_metrics("D").decode_token_num, 0);
}

class ReplacingPolicy final : public RoutingPolicy {
 public:
  explicit ReplacingPolicy(std::shared_ptr<InstanceState> instances)
      : instances_(std::move(instances)) {}
  bool select(const RoutingInput& input, RoutingDecision* decision) override {
    decision->routing.prefill_name = input.candidates.prefill.front().name;
    decision->routing.decode_name = input.candidates.decode.front().name;
    InstanceMetaInfo old;
    EXPECT_TRUE(instances_->remove_instance("P", "P-old", &old));
    old.incarnation_id = "P-new";
    EXPECT_TRUE(instances_->add_instance(std::move(old)));
    return true;
  }

 private:
  std::shared_ptr<InstanceState> instances_;
};

TEST(RoutingStateTest, SelectionCannotRebindToANewerIncarnation) {
  auto instances = std::make_shared<InstanceState>();
  ASSERT_TRUE(
      instances->add_instance(routing_instance("P", InstanceType::PREFILL)));
  ASSERT_TRUE(
      instances->add_instance(routing_instance("D", InstanceType::DECODE)));
  ServiceConfig options;
  options.load_balance_policy("SLO_AWARE");
  RoutingState routing(options, instances);
  ReplacingPolicy policy(instances);
  Request request;
  request.token_ids = {1, 2};
  EXPECT_FALSE(routing.select_route(request, policy, {}));
  EXPECT_TRUE(request.routing.prefill_name.empty());
  EXPECT_EQ(routing.request_metrics("D").decode_request_num, 0);
  EXPECT_EQ(instances->get_instance_info("P").incarnation_id, "P-new");
}
}  // namespace
}  // namespace xllm_service
