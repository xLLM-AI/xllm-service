/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "scheduler/loadbalance_policy/loadbalance_policy.h"

namespace xllm_service {

class SessionTable;

// A request's hold on a binding. Unconfirmed placement is discarded when its
// last hold ends. The shared state can outlive its scheduler during callbacks.
class SessionLease final {
 public:
  ~SessionLease();
  void commit();
  void finish(bool failed);

 private:
  friend class SessionTable;
  friend class SessionAwarePolicy;
  SessionLease(std::shared_ptr<SessionTable> table,
               std::string key,
               uint64_t version);

  std::shared_ptr<SessionTable> table_;
  std::string key_;
  uint64_t version_;
  std::atomic<bool> finished_{false};
  std::atomic<bool> committed_{false};
};

class SessionAwarePolicy final : public LoadBalancePolicy {
 public:
  using TimePoint = std::chrono::steady_clock::time_point;
  using Clock = std::function<TimePoint()>;

  SessionAwarePolicy(std::shared_ptr<InstanceMgr> instance_mgr,
                     std::unique_ptr<LoadBalancePolicy> fallback,
                     int32_t ttl_s,
                     int32_t max_entries,
                     Clock clock = std::chrono::steady_clock::now);

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  std::unique_ptr<LoadBalancePolicy> fallback_;
  std::shared_ptr<SessionTable> table_;
};

}  // namespace xllm_service
