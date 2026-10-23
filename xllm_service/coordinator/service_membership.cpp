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

#include "coordinator/service_membership.h"

#include <chrono>
#include <future>
#include <utility>
namespace {
constexpr int32_t kHeartbeatInterval = 3;
constexpr int32_t kRegistrationMaxRetries = 5;
}  // namespace
namespace xllm_service {
ServiceMembership::ServiceMembership(const ServiceConfig& options,
                                     std::shared_ptr<EtcdClient> etcd,
                                     std::function<void()> on_master,
                                     std::function<void()> on_heartbeat)
    : options_(options),
      etcd_client_(std::move(etcd)),
      on_master_(std::move(on_master)),
      on_heartbeat_(std::move(on_heartbeat)) {}

ServiceMembership::~ServiceMembership() { stop(); }

bool ServiceMembership::start() {
  stopped_ = false;
  events_ = std::make_unique<ThreadPool>();
  if (!register_current_service()) {
    stop();
    return false;
  }
  etcd_client_->add_watch(
      ETCD_XSERVICE_KEY_PREFIX,
      [this](const etcd::Response& response, const uint64_t& prefix_len) {
        for (const auto& event : response.events()) {
          if (event.event_type() != etcd::Event::EventType::DELETE_ ||
              get_event_key_suffix(event, prefix_len) !=
                  options_.service_name()) {
            continue;
          }
          events_->schedule([this] {
            if (!stopped_) {
              register_current_service();
            }
          });
        }
      });
  etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY,
                          [this](const etcd::Response& /*response*/,
                                 const uint64_t& /*prefix_len*/) {
                            events_->schedule([this] { elect_master(); });
                          });
  std::promise<void> election_done;
  auto elected = election_done.get_future();
  events_->schedule([this, &election_done] {
    elect_master();
    election_done.set_value();
  });
  elected.wait();
  heartbeat_ = std::make_unique<std::thread>([this] { heartbeat_loop(); });
  return true;
}

void ServiceMembership::elect_master() {
  if (stopped_ || master_) {
    return;
  }
  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    if (!master_.exchange(true)) {
      on_master_();
    }
  }
}

void ServiceMembership::heartbeat_loop() {
  std::unique_lock lock(stop_mutex_);
  while (!stop_cv_.wait_for(lock,
                            std::chrono::seconds(kHeartbeatInterval),
                            [this] { return stopped_.load(); })) {
    lock.unlock();
    if (master_) {
      on_heartbeat_();
    }
    lock.lock();
  }
}

void ServiceMembership::stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  stop_cv_.notify_all();
  etcd_client_->remove_watch(ETCD_XSERVICE_KEY_PREFIX);
  etcd_client_->remove_watch(ETCD_MASTER_SERVICE_KEY);
  events_.reset();
  if (heartbeat_ && heartbeat_->joinable()) {
    heartbeat_->join();
  }
}

bool ServiceMembership::register_current_service() {
  const std::string service_key =
      ETCD_XSERVICE_KEY_PREFIX + options_.service_name();

  if (etcd_client_->set(
          service_key, options_.service_name(), kHeartbeatInterval)) {
    return true;
  }

  // Key already exists — likely a stale lease from a previous process.
  // Since the lease TTL is short (3 seconds), we can safely wait for it
  // to expire naturally and retry registration without deleting the key.
  LOG(WARNING) << "Service key already exists: " << service_key
               << ". Waiting for the stale lease to expire and retrying.";

  for (int32_t attempt = 1; attempt <= kRegistrationMaxRetries; ++attempt) {
    std::unique_lock lock(stop_mutex_);
    if (stop_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
          return stopped_.load();
        })) {
      return false;
    }
    lock.unlock();
    if (etcd_client_->set(
            service_key, options_.service_name(), kHeartbeatInterval)) {
      LOG(INFO) << "Service registered successfully on retry " << attempt;
      return true;
    }
    LOG(WARNING) << "Registration retry " << attempt << " failed, "
                 << "waiting for old lease to expire...";
  }

  LOG(ERROR) << "Registration failed after " << kRegistrationMaxRetries
             << " retries: " << service_key;
  return false;
}

}  // namespace xllm_service
