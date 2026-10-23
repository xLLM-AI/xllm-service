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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>

#include "scheduler/scheduler.h"

namespace xllm_service {

// Build a scheduler without external discovery or model dependencies, while
// exercising its real admission, removal and output-worker paths.
class SchedulerTestPeer final {
 public:
  static std::unique_ptr<Scheduler> create() {
    auto scheduler = std::unique_ptr<Scheduler>(new Scheduler());
    auto manager = std::shared_ptr<InstanceMgr>(new InstanceMgr());
    manager->scheduler_ = scheduler.get();
    manager->options_.load_balance_policy("SLO_AWARE");
    scheduler->instance_mgr_ = std::move(manager);
    return scheduler;
  }

  static void add_instance(Scheduler& scheduler,
                           const std::string& name,
                           const std::string& incarnation_id,
                           InstanceType type) {
    auto& manager = *scheduler.instance_mgr_;
    std::unique_lock<std::shared_mutex> lock(manager.cluster_mutex_);
    InstanceMetaInfo info;
    info.name = name;
    info.incarnation_id = incarnation_id;
    info.type = type;
    info.dp_size = 1;
    info.kv_split_size = 1;
    manager.add_instance_to_index(name, info);
    manager.instances_.emplace(name, std::move(info));
    std::unique_lock<std::shared_mutex> metrics_lock(manager.metrics_mutex_);
    manager.request_metrics_.emplace(name, RequestMetrics{});
  }

  static bool bind(Scheduler& scheduler,
                   const std::shared_ptr<Request>& request) {
    auto& manager = *scheduler.instance_mgr_;
    if (!manager.bind_request_instance_incarnations(request)) {
      return false;
    }
    manager.update_request_metrics(request, RequestAction::SCHEDULE);
    return true;
  }

  static bool publish(Scheduler& scheduler, std::shared_ptr<Request> request) {
    return scheduler.publish_request(std::move(request));
  }

  static void remove(Scheduler& scheduler, const std::string& name) {
    scheduler.instance_mgr_->deregister_instance(name);
  }

  static bool contains(Scheduler& scheduler, const std::string& request_id) {
    std::lock_guard<std::mutex> lock(scheduler.request_mutex_);
    const bool recorded = scheduler.requests_.count(request_id) != 0;
    EXPECT_EQ(
        recorded,
        scheduler.remote_requests_output_thread_map_.count(request_id) != 0);
    return recorded;
  }

  static RequestMetrics metrics(Scheduler& scheduler, const std::string& name) {
    auto& manager = *scheduler.instance_mgr_;
    std::shared_lock<std::shared_mutex> lock(manager.metrics_mutex_);
    return manager.request_metrics_.at(name);
  }

  static void drain(Scheduler& scheduler) {
    // Markers follow all outputs enqueued by removal and wait for cleanup.
    for (auto& pool : scheduler.output_threadpools_) {
      std::promise<void> done;
      auto completed = done.get_future();
      pool.schedule([&done] { done.set_value(); });
      completed.wait();
    }
  }
};

namespace {

class ConnectedCall final : public CallData {
 public:
  bool proceed(bool /*rpc_ok*/) override { return true; }
  bool is_disconnected() const override { return false; }
};

class RequestAdmissionTest : public testing::Test {
 protected:
  void SetUp() override {
    scheduler_ = SchedulerTestPeer::create();
    SchedulerTestPeer::add_instance(
        *scheduler_, "prefill", "prefill-old", InstanceType::DEFAULT);
    SchedulerTestPeer::add_instance(
        *scheduler_, "decode", "decode-old", InstanceType::DECODE);
  }

  void TearDown() override { SchedulerTestPeer::drain(*scheduler_); }

  std::shared_ptr<Request> make_request(std::string request_id) {
    auto request = std::make_shared<Request>();
    request->service_request_id = std::move(request_id);
    request->routing.prefill_name = "prefill";
    request->routing.decode_name = "decode";
    request->token_ids = {1, 2, 3};
    request->call_data = std::make_shared<ConnectedCall>();
    return request;
  }

  void expect_decode_released() {
    const RequestMetrics metrics =
        SchedulerTestPeer::metrics(*scheduler_, "decode");
    EXPECT_EQ(metrics.decode_request_num, 0);
    EXPECT_EQ(metrics.decode_token_num, 0);
  }

  std::unique_ptr<Scheduler> scheduler_;
};

TEST_F(RequestAdmissionTest, RemovalBeforePublicationRejectsAndReleasesPeer) {
  auto request = make_request("not-yet-published");
  ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
  SchedulerTestPeer::remove(*scheduler_, "prefill");
  EXPECT_FALSE(SchedulerTestPeer::publish(*scheduler_, request));
  EXPECT_FALSE(
      SchedulerTestPeer::contains(*scheduler_, request->service_request_id));
  expect_decode_released();
}

TEST_F(RequestAdmissionTest, ReplacementDoesNotInheritOldRequestOrMetrics) {
  auto request = make_request("bound-to-old-incarnation");
  ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
  SchedulerTestPeer::remove(*scheduler_, "prefill");
  SchedulerTestPeer::add_instance(
      *scheduler_, "prefill", "prefill-new", InstanceType::DEFAULT);
  EXPECT_FALSE(SchedulerTestPeer::publish(*scheduler_, request));
  const RequestMetrics replacement =
      SchedulerTestPeer::metrics(*scheduler_, "prefill");
  EXPECT_EQ(replacement.prefill_request_num, 0);
  EXPECT_EQ(replacement.prefill_token_num, 0);
  expect_decode_released();
}

TEST_F(RequestAdmissionTest, DecodeRemovalBeforePublicationReleasesPrefill) {
  auto request = make_request("decode-removed-before-publication");
  ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
  SchedulerTestPeer::remove(*scheduler_, "decode");
  EXPECT_FALSE(SchedulerTestPeer::publish(*scheduler_, request));
  EXPECT_FALSE(
      SchedulerTestPeer::contains(*scheduler_, request->service_request_id));
  const RequestMetrics prefill =
      SchedulerTestPeer::metrics(*scheduler_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, DecodeReplacementKeepsItsNewMetricsUntouched) {
  auto request = make_request("decode-replaced-before-publication");
  ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
  SchedulerTestPeer::remove(*scheduler_, "decode");
  SchedulerTestPeer::add_instance(
      *scheduler_, "decode", "decode-new", InstanceType::DECODE);
  EXPECT_FALSE(SchedulerTestPeer::publish(*scheduler_, request));
  expect_decode_released();
  const RequestMetrics prefill =
      SchedulerTestPeer::metrics(*scheduler_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, QueuedGenerationPrecedesCancellation) {
  auto request = make_request("queued-generation");
  ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
  auto entered = std::make_shared<std::promise<void>>();
  auto started = entered->get_future();
  std::promise<void> release;
  auto released = release.get_future().share();
  auto phase = std::make_shared<std::atomic<int32_t>>(0);
  auto worker = std::make_shared<std::thread::id>();
  request->output_callback =
      [entered, released, phase, worker](const llm::RequestOutput& output) {
        if (output.status.has_value()) {
          EXPECT_EQ(output.status->code(), llm::StatusCode::CANCELLED);
          EXPECT_EQ(std::this_thread::get_id(), *worker);
          EXPECT_EQ(phase->fetch_add(1), 2);
          return true;
        }
        *worker = std::this_thread::get_id();
        EXPECT_EQ(phase->fetch_add(1), 0);
        entered->set_value();
        EXPECT_EQ(released.wait_for(std::chrono::seconds(5)),
                  std::future_status::ready);
        EXPECT_EQ(phase->fetch_add(1), 1);
        return true;
      };
  ASSERT_TRUE(SchedulerTestPeer::publish(*scheduler_, request));
  llm::RequestOutput output;
  output.service_request_id = request->service_request_id;
  output.outputs.emplace_back();
  output.outputs.back().index = 0;
  output.outputs.back().token_ids = {42, 43, 44, 45};
  ASSERT_TRUE(scheduler_->handle_generation(std::move(output)));
  const auto ready = started.wait_for(std::chrono::seconds(5));
  if (ready != std::future_status::ready) {
    release.set_value();
    FAIL() << "The request output worker did not start";
  }

  // A successful Decode output closes Prefill. Fail Decode while that
  // output's callback is blocked, exercising the real cancellation route.
  EXPECT_EQ(
      SchedulerTestPeer::metrics(*scheduler_, "prefill").prefill_request_num,
      0);
  EXPECT_EQ(SchedulerTestPeer::metrics(*scheduler_, "decode").decode_token_num,
            static_cast<int64_t>(request->token_ids.size()) + 4);
  SchedulerTestPeer::remove(*scheduler_, "decode");
  EXPECT_EQ(phase->load(), 1);
  EXPECT_TRUE(
      SchedulerTestPeer::contains(*scheduler_, request->service_request_id));
  release.set_value();
  SchedulerTestPeer::drain(*scheduler_);
  EXPECT_EQ(phase->load(), 3);
  EXPECT_FALSE(
      SchedulerTestPeer::contains(*scheduler_, request->service_request_id));
  const RequestMetrics prefill =
      SchedulerTestPeer::metrics(*scheduler_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, ConcurrentRemovalCannotOrphanRequest) {
  for (int32_t iteration = 0; iteration < 32; ++iteration) {
    if (iteration != 0) {
      SchedulerTestPeer::add_instance(
          *scheduler_, "prefill", "prefill-old", InstanceType::DEFAULT);
    }
    auto request = make_request("concurrent-" + std::to_string(iteration));
    ASSERT_TRUE(SchedulerTestPeer::bind(*scheduler_, request));
    std::atomic<int32_t> cancellations{0};
    request->output_callback = [&cancellations](const llm::RequestOutput&) {
      ++cancellations;
      return true;
    };
    std::promise<void> start;
    std::shared_future<void> started = start.get_future().share();
    auto publication = std::async(std::launch::async, [&] {
      started.wait();
      return SchedulerTestPeer::publish(*scheduler_, request);
    });
    auto removal = std::async(std::launch::async, [&] {
      started.wait();
      SchedulerTestPeer::remove(*scheduler_, "prefill");
    });
    start.set_value();
    const bool admitted = publication.get();
    removal.get();
    SchedulerTestPeer::drain(*scheduler_);
    EXPECT_EQ(cancellations.load(), admitted ? 1 : 0);
    EXPECT_FALSE(
        SchedulerTestPeer::contains(*scheduler_, request->service_request_id));
    expect_decode_released();
  }
}

}  // namespace
}  // namespace xllm_service
