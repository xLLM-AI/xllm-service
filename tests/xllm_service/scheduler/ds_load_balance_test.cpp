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

#include "scheduler/loadbalance_policy/ds_load_balance.h"

#include <gtest/gtest.h>
#include <xxhash.h>

#include <future>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/request_metrics.h"
#include "scheduler/scheduler.h"

namespace xllm_service {

class SchedulerTestPeer final {
 public:
  static std::shared_ptr<InstanceMgr> create() {
    auto scheduler = std::shared_ptr<Scheduler>(new Scheduler());
    auto manager = std::shared_ptr<InstanceMgr>(new InstanceMgr());
    manager->options_.load_balance_policy("ds_load_balance");
    manager->scheduler_ = scheduler.get();
    scheduler->instance_mgr_ = manager;
    // Keep the scheduler (and therefore its manager) alive for the test.
    return std::shared_ptr<InstanceMgr>(std::move(scheduler), manager.get());
  }

  static void add(InstanceMgr& manager,
                  const std::string& name,
                  InstanceType type,
                  double scale = 1.0,
                  bool profiled = true) {
    InstanceMetaInfo info;
    info.name = name;
    info.incarnation_id = name + "-generation";
    info.type = type;
    info.dp_size = 1;
    info.kv_split_size = 1;
    if (profiled) {
      info.ttft_profiling_data = {
          {100, 30 * scale}, {200, 70 * scale}, {300, 130 * scale}};
    }
    manager.add_instance_to_index(name, info);
    manager.instances_.emplace(name, info);
    manager.request_metrics_.emplace(name, RequestMetrics{});
    manager.time_predictors_.emplace(
        name, TimePredictor(info.ttft_profiling_data, {}));
  }

  static RequestMetrics metrics(InstanceMgr& manager, const std::string& name) {
    return manager.request_metrics_.at(name);
  }

  static void set_decode_load(InstanceMgr& manager,
                              const std::string& name,
                              int64_t requests,
                              int64_t tokens) {
    auto& metrics = manager.request_metrics_.at(name);
    metrics.decode_request_num = requests;
    metrics.decode_token_num = tokens;
  }

  static void mark_suspect(InstanceMgr& manager, const std::string& name) {
    manager.instances_.at(name).runtime_state = InstanceRuntimeState::SUSPECT;
  }

  static bool publish(InstanceMgr& manager, std::shared_ptr<Request> request) {
    return manager.scheduler_->publish_request(std::move(request));
  }

  static void finish(InstanceMgr& manager, const std::string& request_id) {
    manager.scheduler_->finish_request(request_id, /*error=*/true);
  }
};

namespace {

std::shared_ptr<Request> make_request() {
  auto request = std::make_shared<Request>();
  request->token_ids.resize(100, 42);
  return request;
}

TEST(DsLoadBalanceTest, AccountsNewRequestAndQueuedWorkWithoutRoleFlips) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "fast", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "slow", InstanceType::PREFILL, 2.0);
  SchedulerTestPeer::add(*manager, "decode", InstanceType::DECODE);
  DsLoadBalance policy(manager, nullptr);
  auto first = make_request();
  auto second = make_request();
  auto third = make_request();
  ASSERT_TRUE(policy.select_instances_pair(first));
  ASSERT_TRUE(policy.select_instances_pair(second));
  ASSERT_TRUE(policy.select_instances_pair(third));
  EXPECT_EQ(first->routing.prefill_name, "fast");
  EXPECT_EQ(second->routing.prefill_name, "slow");
  EXPECT_EQ(third->routing.prefill_name, "fast");
  EXPECT_EQ(first->routing.decode_name, "decode");
  EXPECT_EQ(manager->get_instance_info("decode").type, InstanceType::DECODE);
  manager->update_request_metrics(first, RequestAction::CANCEL);
  manager->update_request_metrics(second, RequestAction::CANCEL);
  manager->update_request_metrics(third, RequestAction::CANCEL);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "fast").estimated_prefill_time,
            0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "slow").prefill_request_num,
            0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "decode").decode_request_num,
            0);
}

TEST(DsLoadBalanceTest, PrefillSelectsExistingLoadBeforeReservingNewWork) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "fast", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "slow", InstanceType::PREFILL, 10.0);
  SchedulerTestPeer::add(*manager, "d", InstanceType::DECODE);
  auto first = make_request();
  auto second = make_request();
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(first, {}));
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(second, {}));
  EXPECT_EQ(first->routing.prefill_name, "fast");
  EXPECT_EQ(second->routing.prefill_name, "slow");
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "slow").estimated_prefill_time,
            second->estimated_ttft);
  EXPECT_GT(second->estimated_ttft, first->estimated_ttft);
}

TEST(DsLoadBalanceTest, DecodeBalancesRequestsWithinTokenThreshold) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "p", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "d1", InstanceType::DECODE);
  SchedulerTestPeer::add(*manager, "d2", InstanceType::DECODE);
  SchedulerTestPeer::add(*manager, "offline", InstanceType::DECODE);
  SchedulerTestPeer::mark_suspect(*manager, "offline");
  SchedulerTestPeer::set_decode_load(*manager, "d1", 1, 50000);
  SchedulerTestPeer::set_decode_load(*manager, "d2", 100, 48000);
  auto short_request = make_request();
  ASSERT_TRUE(
      manager->select_instance_pair_on_ds_load_balance(short_request, {}));
  EXPECT_EQ(short_request->routing.decode_name, "d1");
  manager->update_request_metrics(short_request, RequestAction::CANCEL);

  auto long_request = make_request();
  long_request->token_ids.resize(20000, 42);
  ASSERT_TRUE(
      manager->select_instance_pair_on_ds_load_balance(long_request, {}));
  EXPECT_EQ(long_request->routing.decode_name, "d2");
  manager->update_request_metrics(long_request, RequestAction::CANCEL);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d1").decode_token_num, 50000);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d2").decode_token_num, 48000);

  // Few sequences do not bypass the token threshold on an overloaded D.
  SchedulerTestPeer::set_decode_load(*manager, "d1", 1, 64000);
  SchedulerTestPeer::set_decode_load(*manager, "d2", 100, 32000);
  auto request = make_request();
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(request, {}));
  EXPECT_EQ(request->routing.decode_name, "d2");
}

TEST(DsLoadBalanceTest, UsesCompatiblePrefixAndReleasesOnlyOnceAfterHandoff) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "p", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "d", InstanceType::DECODE);
  auto request = make_request();
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(
      request, {{"p", PrefillCacheHint{"p-generation", 50}}}));
  EXPECT_EQ(request->store_cached_tokens, 50);
  EXPECT_NEAR(request->estimated_ttft, 23, 1);
  manager->update_request_metrics(request, RequestAction::SCHEDULE);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p").prefill_request_num, 1);
  llm::RequestOutput output;
  output.outputs.emplace_back();
  output.outputs.back().token_ids = {1, 2, 3, 4};
  const GenerationProgress progress = record_generation(*request, output);
  ASSERT_TRUE(progress.finished_prefill);
  manager->update_request_metrics(
      request, RequestAction::FINISH_PREFILL, progress.generated_tokens);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d").decode_token_num, 104);
  manager->update_request_metrics(request, RequestAction::CANCEL);
  manager->update_request_metrics(request, RequestAction::CANCEL);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p").estimated_prefill_time,
            0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p").prefill_request_num, 0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d").decode_token_num, 0);
  auto stale = make_request();
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(
      stale, {{"p", PrefillCacheHint{"old-generation", 99}}}));
  EXPECT_EQ(stale->store_cached_tokens, 0);
}

TEST(DsLoadBalanceTest, RejectsMissingProfilesAndUnavailableDecode) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "p", InstanceType::PREFILL);
  auto request = make_request();
  EXPECT_FALSE(manager->select_instance_pair_on_ds_load_balance(request, {}));
  EXPECT_FALSE(request->ds_load_balance_reserved);
  SchedulerTestPeer::add(*manager, "default", InstanceType::DEFAULT, 1, false);
  EXPECT_FALSE(manager->select_instance_pair_on_ds_load_balance(request, {}));
  SchedulerTestPeer::add(*manager, "ready", InstanceType::DEFAULT);
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(request, {}));
  EXPECT_EQ(request->routing.prefill_name, "ready");
  EXPECT_TRUE(request->routing.decode_name.empty());
}

TEST(DsLoadBalanceTest, ConcurrentSelectionsReserveBeforeTheNextSelection) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "p1", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "p2", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "d", InstanceType::DECODE);
  SchedulerTestPeer::add(*manager, "d2", InstanceType::DECODE);
  std::vector<std::future<bool>> futures;
  futures.reserve(16);
  for (int32_t index = 0; index < 16; ++index) {
    futures.emplace_back(std::async(std::launch::async, [manager] {
      return manager->select_instance_pair_on_ds_load_balance(make_request(),
                                                              {});
    }));
  }
  for (auto& future : futures) {
    ASSERT_TRUE(future.get());
  }
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p1").prefill_request_num, 8);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p2").prefill_request_num, 8);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d").decode_request_num, 8);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d2").decode_request_num, 8);
}

TEST(DsLoadBalanceTest, DuplicateAdmissionPreservesActiveReservation) {
  auto manager = SchedulerTestPeer::create();
  SchedulerTestPeer::add(*manager, "p", InstanceType::PREFILL);
  SchedulerTestPeer::add(*manager, "d", InstanceType::DECODE);
  auto original = make_request();
  original->service_request_id = "duplicate";
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(original, {}));
  ASSERT_TRUE(SchedulerTestPeer::publish(*manager, original));
  EXPECT_FALSE(SchedulerTestPeer::publish(*manager, original));
  EXPECT_TRUE(original->ds_load_balance_reserved);
  auto duplicate = make_request();
  duplicate->service_request_id = original->service_request_id;
  ASSERT_TRUE(manager->select_instance_pair_on_ds_load_balance(duplicate, {}));
  EXPECT_FALSE(SchedulerTestPeer::publish(*manager, duplicate));
  EXPECT_FALSE(duplicate->ds_load_balance_reserved);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p").prefill_request_num, 1);
  SchedulerTestPeer::finish(*manager, original->service_request_id);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "p").estimated_prefill_time,
            0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*manager, "d").decode_request_num, 0);
}

TEST(PrefillTimePredictorTest, FitsPrefixWorkAndRejectsInvalidCosts) {
  TimePredictor predictor({{100, 30}, {200, 70}, {300, 130}}, {});
  ASSERT_TRUE(predictor.predict_prefill_time(100, 50).has_value());
  EXPECT_NEAR(*predictor.predict_prefill_time(100, 50), 22.5, 1e-6);
  EXPECT_FALSE(predictor.predict_prefill_time(100, 100).has_value());
  TimePredictor missing({}, {});
  EXPECT_FALSE(missing.predict_prefill_time(100, 0).has_value());
  TimePredictor underdetermined({{100, 30}, {200, 70}}, {});
  EXPECT_FALSE(underdetermined.predict_prefill_time(100, 0).has_value());
}

TEST(PrefillTimePredictorTest, NoisyFitStaysPositiveForShortAndCachedInputs) {
  TimePredictor predictor({{100, 10}, {200, 19}, {400, 21}}, {});
  for (int32_t length :
       {1, 16, 257, 65536, std::numeric_limits<int32_t>::max()}) {
    const auto full = predictor.predict_prefill_time(length, 0);
    const auto cached = predictor.predict_prefill_time(length, length - 1);
    ASSERT_TRUE(full.has_value());
    ASSERT_TRUE(cached.has_value());
    EXPECT_GT(*cached, 0);
    EXPECT_LE(*cached, *full);
  }
  TimePredictor decreasing({{100, 30}, {200, 20}, {400, 10}}, {});
  ASSERT_TRUE(decreasing.predict_prefill_time(16, 0).has_value());
  EXPECT_GT(*decreasing.predict_prefill_time(16, 0), 0);
}

class FakeStore final : public StoreKeyLookup {
 public:
  bool batch_exists(const std::vector<std::string>& keys,
                    std::vector<bool>* exists) override {
    queried = keys;
    *exists = replies;
    return ok;
  }

  bool ok = true;
  std::vector<bool> replies;
  std::vector<std::string> queried;
};

TEST(MooncakePrefixTest, RequiresAllShardsAndContiguousBlocks) {
  auto client = std::make_unique<FakeStore>();
  FakeStore* observed = client.get();
  observed->replies = {true, true, true, false, true, true};
  StorePrefixLayout layout{{"p"}, {"rank0:", "rank1:"}, 2, 1024, 0};
  MooncakePrefixLookup lookup({layout}, 64, std::move(client));
  InstanceMetaInfo instance;
  instance.name = "p";
  instance.incarnation_id = "generation";
  const auto match = lookup.match({1, 2, 3, 4, 5, 6, 7}, {instance});
  ASSERT_EQ(match.size(), 1);
  EXPECT_EQ(match.at("p").cached_tokens, 2);
  EXPECT_EQ(match.at("p").incarnation_id, "generation");
  ASSERT_EQ(observed->queried.size(), 6);
  const int32_t first_tokens[] = {1, 2};
  const XXH128_hash_t hash =
      XXH3_128bits_withSeed(first_tokens, sizeof(first_tokens), 1024);
  const std::string expected(reinterpret_cast<const char*>(&hash),
                             sizeof(hash));
  EXPECT_EQ(observed->queried[0], "rank0:" + expected);
  EXPECT_EQ(observed->queried[1], "rank1:" + expected);
}

TEST(MooncakePrefixTest, LeavesLogitTokenAndBoundsQueriesAndErrors) {
  auto client = std::make_unique<FakeStore>();
  FakeStore* observed = client.get();
  observed->replies = {true, true};
  StorePrefixLayout layout{{"p"}, {"rank0:", "rank1:"}, 2, 1024, 1};
  MooncakePrefixLookup lookup({layout}, 2, std::move(client));
  InstanceMetaInfo instance;
  instance.name = "p";
  auto match = lookup.match({1, 2, 3, 4}, {instance});
  ASSERT_EQ(match.at("p").cached_tokens, 2);
  EXPECT_EQ(observed->queried.size(), 2);
  const int32_t mtp_tokens[] = {1, 2, 3};
  const XXH128_hash_t hash =
      XXH3_128bits_withSeed(mtp_tokens, sizeof(mtp_tokens), 1024);
  EXPECT_EQ(observed->queried[0],
            "rank0:" + std::string(reinterpret_cast<const char*>(&hash), 16));
  observed->ok = false;
  EXPECT_TRUE(lookup.match({1, 2, 3, 4}, {instance}).empty());
  observed->ok = true;
  observed->replies.clear();
  EXPECT_TRUE(lookup.match({1, 2, 3, 4}, {instance}).empty());
}

}  // namespace
}  // namespace xllm_service
