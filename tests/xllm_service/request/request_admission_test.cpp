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

#include "tests/xllm_service/service_test_peer.h"

namespace xllm_service {

namespace {

class RequestAdmissionTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = ServiceTestPeer::create();
    ServiceTestPeer::add_instance(
        *context_, "prefill", "prefill-old", InstanceType::DEFAULT);
    ServiceTestPeer::add_instance(
        *context_, "decode", "decode-old", InstanceType::DECODE);
  }

  void TearDown() override { ServiceTestPeer::drain(*context_); }

  std::shared_ptr<Request> make_request(std::string request_id) {
    auto request = std::make_shared<Request>();
    request->service_request_id = std::move(request_id);
    request->routing.prefill_name = "prefill";
    request->routing.decode_name = "decode";
    request->token_ids = {1, 2, 3};
    request->is_disconnected = [] { return false; };
    return request;
  }

  void expect_decode_released() {
    const RequestMetrics metrics =
        ServiceTestPeer::metrics(*context_, "decode");
    EXPECT_EQ(metrics.decode_request_num, 0);
    EXPECT_EQ(metrics.decode_token_num, 0);
  }

  std::unique_ptr<ServiceTestContext> context_;
};

TEST_F(RequestAdmissionTest, RemovalBeforePublicationRejectsAndReleasesPeer) {
  auto request = make_request("not-yet-published");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
  ServiceTestPeer::remove(*context_, "prefill");
  EXPECT_FALSE(ServiceTestPeer::publish(*context_, request));
  EXPECT_FALSE(
      ServiceTestPeer::contains(*context_, request->service_request_id));
  expect_decode_released();
}

TEST_F(RequestAdmissionTest, ReplacementDoesNotInheritOldRequestOrMetrics) {
  auto request = make_request("bound-to-old-incarnation");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
  ServiceTestPeer::remove(*context_, "prefill");
  ServiceTestPeer::add_instance(
      *context_, "prefill", "prefill-new", InstanceType::DEFAULT);
  EXPECT_FALSE(ServiceTestPeer::publish(*context_, request));
  const RequestMetrics replacement =
      ServiceTestPeer::metrics(*context_, "prefill");
  EXPECT_EQ(replacement.prefill_request_num, 0);
  EXPECT_EQ(replacement.prefill_token_num, 0);
  expect_decode_released();
}

TEST_F(RequestAdmissionTest, DecodeRemovalBeforePublicationReleasesPrefill) {
  auto request = make_request("decode-removed-before-publication");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
  ServiceTestPeer::remove(*context_, "decode");
  EXPECT_FALSE(ServiceTestPeer::publish(*context_, request));
  EXPECT_FALSE(
      ServiceTestPeer::contains(*context_, request->service_request_id));
  const RequestMetrics prefill = ServiceTestPeer::metrics(*context_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, DecodeReplacementKeepsItsNewMetricsUntouched) {
  auto request = make_request("decode-replaced-before-publication");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
  ServiceTestPeer::remove(*context_, "decode");
  ServiceTestPeer::add_instance(
      *context_, "decode", "decode-new", InstanceType::DECODE);
  EXPECT_FALSE(ServiceTestPeer::publish(*context_, request));
  expect_decode_released();
  const RequestMetrics prefill = ServiceTestPeer::metrics(*context_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, QueuedGenerationPrecedesCancellation) {
  auto request = make_request("queued-generation");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
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
  ASSERT_TRUE(ServiceTestPeer::publish(*context_, request));
  llm::RequestOutput output;
  output.service_request_id = request->service_request_id;
  ASSERT_TRUE(context_->requests->handle_generation(std::move(output)));
  const auto ready = started.wait_for(std::chrono::seconds(5));
  if (ready != std::future_status::ready) {
    release.set_value();
    FAIL() << "The request output worker did not start";
  }

  // A successful Decode output closes Prefill. Fail Decode while that
  // output's callback is blocked, exercising the real cancellation route.
  ServiceTestPeer::remove(*context_, "decode");
  EXPECT_EQ(phase->load(), 1);
  EXPECT_TRUE(
      ServiceTestPeer::contains(*context_, request->service_request_id));
  release.set_value();
  ServiceTestPeer::drain(*context_);
  EXPECT_EQ(phase->load(), 3);
  EXPECT_FALSE(
      ServiceTestPeer::contains(*context_, request->service_request_id));
  const RequestMetrics prefill = ServiceTestPeer::metrics(*context_, "prefill");
  EXPECT_EQ(prefill.prefill_request_num, 0);
  EXPECT_EQ(prefill.prefill_token_num, 0);
}

TEST_F(RequestAdmissionTest, ConcurrentRemovalCannotOrphanRequest) {
  for (int32_t iteration = 0; iteration < 32; ++iteration) {
    if (iteration != 0) {
      ServiceTestPeer::add_instance(
          *context_, "prefill", "prefill-old", InstanceType::DEFAULT);
    }
    auto request = make_request("concurrent-" + std::to_string(iteration));
    ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
    std::atomic<int32_t> cancellations{0};
    request->output_callback = [&cancellations](const llm::RequestOutput&) {
      ++cancellations;
      return true;
    };
    std::promise<void> start;
    std::shared_future<void> started = start.get_future().share();
    auto publication = std::async(std::launch::async, [&] {
      started.wait();
      return ServiceTestPeer::publish(*context_, request);
    });
    auto removal = std::async(std::launch::async, [&] {
      started.wait();
      ServiceTestPeer::remove(*context_, "prefill");
    });
    start.set_value();
    const bool admitted = publication.get();
    removal.get();
    ServiceTestPeer::drain(*context_);
    EXPECT_EQ(cancellations.load(), admitted ? 1 : 0);
    EXPECT_FALSE(
        ServiceTestPeer::contains(*context_, request->service_request_id));
    expect_decode_released();
  }
}

}  // namespace
TEST_F(RequestAdmissionTest, ShutdownCancelsOnceAndRejectsFurtherAdmission) {
  auto request = make_request("shutdown");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, request));
  std::atomic<int32_t> callbacks{0};
  request->output_callback = [&callbacks](const llm::RequestOutput& output) {
    EXPECT_TRUE(output.status.has_value());
    ++callbacks;
    return true;
  };
  ASSERT_TRUE(ServiceTestPeer::publish(*context_, request));
  context_->requests->stop();
  context_->requests->stop();
  EXPECT_EQ(callbacks.load(), 1);
  EXPECT_FALSE(
      ServiceTestPeer::contains(*context_, request->service_request_id));
  expect_decode_released();
  auto later = make_request("after-shutdown");
  ASSERT_TRUE(ServiceTestPeer::bind(*context_, later));
  EXPECT_FALSE(ServiceTestPeer::publish(*context_, later));
  expect_decode_released();
}

}  // namespace xllm_service
