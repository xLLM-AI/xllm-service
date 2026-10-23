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

#include "scheduler/loadbalance_policy/session_aware_policy.h"

#include <absl/strings/strip.h>
#include <glog/logging.h>

#include <algorithm>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace xllm_service {
namespace {

struct SessionSelection {
  Routing route;
  uint64_t version;
};

const RoutingCandidate* find_candidate(
    const std::vector<RoutingCandidate>& candidates,
    const std::string& name) {
  const auto found = std::find_if(
      candidates.begin(), candidates.end(), [&name](const auto& candidate) {
        return candidate.name == name;
      });
  return found == candidates.end() ? nullptr : &*found;
}

const RoutingCandidate* bound_candidate(
    const std::vector<RoutingCandidate>& candidates,
    const std::string& name,
    const std::string& incarnation_id) {
  const auto* found = find_candidate(candidates, name);
  return found != nullptr && !incarnation_id.empty() &&
                 found->incarnation_id == incarnation_id
             ? found
             : nullptr;
}

std::optional<Routing> select_route(const RoutingCandidates& candidates,
                                    const Routing* previous,
                                    const std::optional<Routing>& proposal) {
  const auto* prefill = previous == nullptr
                            ? nullptr
                            : bound_candidate(candidates.prefill,
                                              previous->prefill_name,
                                              previous->prefill_incarnation_id);
  const auto* decode = previous == nullptr
                           ? nullptr
                           : bound_candidate(candidates.decode,
                                             previous->decode_name,
                                             previous->decode_incarnation_id);
  const bool complete =
      prefill != nullptr &&
      (decode != nullptr || (previous->decode_name.empty() &&
                             prefill->type == InstanceType::DEFAULT));
  if (!complete) {
    if (!proposal) {
      return std::nullopt;
    }
    if (decode == nullptr) {
      decode = find_candidate(candidates.decode, proposal->decode_name);
    }
    if (prefill == nullptr ||
        (decode == nullptr && prefill->type != InstanceType::DEFAULT)) {
      prefill = find_candidate(candidates.prefill, proposal->prefill_name);
    }
  }
  if (prefill == nullptr ||
      (decode == nullptr && prefill->type != InstanceType::DEFAULT)) {
    return std::nullopt;
  }
  if (prefill->incarnation_id.empty() ||
      (decode != nullptr && decode->incarnation_id.empty())) {
    LOG(ERROR) << "Session affinity requires nonempty incarnation IDs for "
                  "all selected instances";
    return std::nullopt;
  }
  Routing route;
  route.prefill_name = prefill->name;
  route.prefill_incarnation_id = prefill->incarnation_id;
  if (decode != nullptr) {
    route.decode_name = decode->name;
    route.decode_incarnation_id = decode->incarnation_id;
  }
  return route;
}

bool same_route(const Routing& left, const Routing& right) {
  return left.prefill_name == right.prefill_name &&
         left.decode_name == right.decode_name &&
         left.prefill_incarnation_id == right.prefill_incarnation_id &&
         left.decode_incarnation_id == right.decode_incarnation_id;
}

}  // namespace

// Owns all binding transitions. Policies and leases never access its containers
// or mutex; selection consumes only local candidate snapshots.
class SessionTable final {
 public:
  SessionTable(int32_t ttl_s,
               int32_t max_entries,
               SessionAwarePolicy::Clock clock)
      : ttl_(std::chrono::seconds(ttl_s)),
        max_entries_(static_cast<size_t>(max_entries)),
        clock_(std::move(clock)) {
    CHECK_GT(ttl_s, 0);
    CHECK_GT(max_entries, 0);
    CHECK(clock_);
  }

  std::optional<SessionSelection> acquire(
      const std::string& key,
      const RoutingCandidates& candidates,
      const std::optional<Routing>& proposal = std::nullopt) {
    std::lock_guard<std::mutex> lock(mutex_);
    expire(clock_());
    auto found = bindings_.find(key);
    auto route =
        select_route(candidates,
                     found == bindings_.end() ? nullptr : &found->second.route,
                     proposal);
    if (!route) {
      return std::nullopt;
    }
    if (found == bindings_.end()) {
      if (bindings_.size() >= max_entries_) {
        if (idle_.empty()) {
          LOG(ERROR)
              << "Session affinity capacity is occupied by active requests";
          return std::nullopt;
        }
        erase(idle_.front());
      }
      found = bindings_.emplace(key, Binding{}).first;
    }
    auto& binding = found->second;
    if (binding.idle_position) {
      idle_.erase(*binding.idle_position);
      binding.idle_position.reset();
    }
    if (!same_route(binding.route, *route)) {
      binding.route = *route;
      binding.version = ++next_version_;
      binding.active = 0;
      binding.committed = false;
    }
    ++binding.active;
    return SessionSelection{std::move(*route), binding.version};
  }

  void commit(SessionLease& lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lease.finished_.load(std::memory_order_relaxed)) {
      return;
    }
    const auto found = bindings_.find(lease.key_);
    if (found == bindings_.end() || found->second.version != lease.version_) {
      // A newer binding superseded this request; future callbacks are no-ops.
      lease.finished_.store(true, std::memory_order_relaxed);
      return;
    }
    found->second.committed = true;
    lease.committed_.store(true, std::memory_order_relaxed);
  }

  void finish(SessionLease& lease, bool failed) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lease.finished_.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    const auto found = bindings_.find(lease.key_);
    if (found == bindings_.end() || found->second.version != lease.version_) {
      return;
    }
    auto& binding = found->second;
    if (failed) {
      erase(lease.key_);
      return;
    }
    CHECK_GT(binding.active, 0);
    if (--binding.active != 0) {
      return;
    }
    if (!binding.committed) {
      erase(lease.key_);
      return;
    }
    binding.idle_since = clock_();
    idle_.emplace_back(lease.key_);
    binding.idle_position = std::prev(idle_.end());
  }

 private:
  struct Binding {
    Routing route;
    uint64_t version = 0;
    uint64_t active = 0;
    bool committed = false;
    SessionAwarePolicy::TimePoint idle_since;
    std::optional<std::list<std::string>::iterator> idle_position;
  };

  void erase(const std::string& key) {
    const auto found = bindings_.find(key);
    if (found->second.idle_position) {
      idle_.erase(*found->second.idle_position);
    }
    bindings_.erase(found);
  }

  void expire(SessionAwarePolicy::TimePoint now) {
    while (!idle_.empty() &&
           now - bindings_.at(idle_.front()).idle_since >= ttl_) {
      erase(idle_.front());
    }
  }

  std::mutex mutex_;
  const std::chrono::seconds ttl_;
  const size_t max_entries_;
  SessionAwarePolicy::Clock clock_;
  std::unordered_map<std::string, Binding> bindings_;
  // Only idle, committed bindings can expire or be evicted.
  std::list<std::string> idle_;
  uint64_t next_version_ = 0;
};

SessionLease::SessionLease(std::shared_ptr<SessionTable> table,
                           std::string key,
                           uint64_t version)
    : table_(std::move(table)), key_(std::move(key)), version_(version) {}

SessionLease::~SessionLease() { finish(/*failed=*/false); }

void SessionLease::commit() {
  if (!committed_.load(std::memory_order_relaxed) &&
      !finished_.load(std::memory_order_relaxed)) {
    table_->commit(*this);
  }
}

void SessionLease::finish(bool failed) {
  if (!finished_.load(std::memory_order_relaxed)) {
    table_->finish(*this, failed);
  }
}

SessionAwarePolicy::SessionAwarePolicy(
    std::shared_ptr<InstanceMgr> instance_mgr,
    std::unique_ptr<LoadBalancePolicy> fallback,
    int32_t ttl_s,
    int32_t max_entries,
    Clock clock)
    : LoadBalancePolicy(std::move(instance_mgr)),
      fallback_(std::move(fallback)),
      table_(std::make_shared<SessionTable>(ttl_s,
                                            max_entries,
                                            std::move(clock))) {
  CHECK(fallback_ != nullptr);
}

bool SessionAwarePolicy::select_instances_pair(
    std::shared_ptr<Request> request) {
  const auto session_id = absl::StripAsciiWhitespace(request->session_id);
  if (session_id.empty()) {
    return fallback_->select_instances_pair(std::move(request));
  }
  // Length framing prevents model/session delimiter collisions.
  std::string key = std::to_string(request->model.size()) + ":";
  key.reserve(key.size() + request->model.size() + session_id.size());
  key.append(request->model);
  key.append(session_id.data(), session_id.size());

  auto selected = table_->acquire(
      key, instance_mgr_->get_routing_candidates(/*require_load=*/false));
  if (!selected) {
    // Fallback may perform an expensive CAR lookup. Reconcile afterward with
    // any provisional placement established by a concurrent first turn.
    std::optional<Routing> proposal;
    if (fallback_->select_instances_pair(request)) {
      proposal = request->routing;
    }
    selected = table_->acquire(
        key,
        instance_mgr_->get_routing_candidates(/*require_load=*/false),
        proposal);
  }
  if (!selected) {
    return false;
  }
  request->routing = std::move(selected->route);
  request->session_lease = std::shared_ptr<SessionLease>(
      new SessionLease(table_, std::move(key), selected->version));
  return true;
}

}  // namespace xllm_service
