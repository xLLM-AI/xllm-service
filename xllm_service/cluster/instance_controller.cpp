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

#include "cluster/instance_controller.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <utility>

#include "xllm_rpc_service.pb.h"

namespace {
uint64_t now_ms() {
  return static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
}
}  // namespace
namespace xllm_service {
InstanceController::InstanceController(const ServiceConfig& options,
                                       std::shared_ptr<InstanceState> instances,
                                       InstanceClient& client,
                                       RetirementCallback retire)
    : options_(options),
      instances_(std::move(instances)),
      client_(client),
      retire_(std::move(retire)) {
  CHECK(retire_);
}

InstanceController::~InstanceController() { stop(); }

void InstanceController::start() {
  exited_ = false;
  state_reconcile_thread_ =
      std::make_unique<std::thread>([this] { reconcile_instance_states(); });
}

void InstanceController::stop() {
  exited_ = true;
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }
}

void InstanceController::on_instance_event(InstanceEvent event) {
  if (exited_ || event.name.empty()) {
    return;
  }
  const auto current = instances_->find_instance(event.name);
  if (!event.deleted) {
    event.info.name = event.name;
    if (current.has_value() &&
        current->incarnation_id == event.info.incarnation_id) {
      instances_->refresh_instance(std::move(event.info));
      return;
    }
    if (current.has_value()) {
      deregister_instance(event.name, current->incarnation_id);
    }
    if (!register_instance(std::move(event.info))) {
      LOG(ERROR) << "Failed to register instance: " << event.name;
    }
    return;
  }
  if (!current.has_value() ||
      (!event.info.incarnation_id.empty() &&
       current->incarnation_id != event.info.incarnation_id)) {
    return;
  }
  // Probe outside the state lock. A heartbeat received during a failed probe
  // invalidates the old timestamp, so it cannot mark a recovered peer suspect.
  if (client_.probe_instance_health(event.name)) {
    instances_->mark_lease_lost(event.name, current->incarnation_id);
  } else {
    instances_->mark_suspect(
        event.name, current->incarnation_id, current->latest_timestamp);
  }
}

void InstanceController::on_load_event(LoadEvent event) {
  if (!exited_) {
    instances_->apply_load_event(event);
  }
}

bool InstanceController::handle_instance_heartbeat(
    const proto::HeartbeatRequest* request) {
  if (exited_) {
    return false;
  }
  return instances_->record_heartbeat(
      request->name(),
      request->incarnation_id(),
      LoadMetrics(request->load_metrics().waiting_requests_num(),
                  request->load_metrics().gpu_cache_usage_perc()),
      LatencyMetrics(request->latency_metrics().recent_max_ttft(),
                     request->latency_metrics().recent_max_tbt()));
}

std::vector<std::pair<std::string, InstanceMetaInfo>>
InstanceController::link_operations(const InstanceMetaInfo& info,
                                    bool removing) const {
  std::vector<std::pair<std::string, InstanceMetaInfo>> result;
  if (info.backend_type == "vllm" || info.type == InstanceType::DEFAULT) {
    return result;
  }
  const auto snapshot = instances_->snapshot();
  result.reserve(snapshot.size());
  for (const auto& peer : snapshot) {
    if (peer.info->name == info.name) {
      continue;
    }
    if (info.type == InstanceType::PREFILL &&
        peer.role == InstanceType::DECODE) {
      result.emplace_back(peer.info->rpc_address, info);
    } else if (info.type == InstanceType::DECODE &&
               peer.role == InstanceType::PREFILL) {
      result.emplace_back(info.rpc_address, *peer.info);
    } else if (info.type == InstanceType::MIX) {
      result.emplace_back(removing ? peer.info->rpc_address : info.rpc_address,
                          removing ? info : *peer.info);
    }
  }
  return result;
}

bool InstanceController::register_instance(InstanceMetaInfo info) {
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  if (instances_->find_instance(info.name).has_value()) {
    return false;
  }
  if (!client_.connect_instance(info)) {
    return false;
  }
  auto operations = link_operations(info, /*removing=*/false);
  size_t completed = 0;
  for (; completed < operations.size(); ++completed) {
    const auto& operation = operations[completed];
    if (!client_.call_link_instance(operation.first, operation.second)) {
      break;
    }
  }
  if (completed == operations.size()) {
    if (info.type == InstanceType::MIX) {
      const auto snapshot = instances_->snapshot();
      const bool has_decode = std::any_of(
          snapshot.begin(), snapshot.end(), [](const InstanceView& view) {
            return view.role == InstanceType::DECODE;
          });
      info.current_type =
          has_decode ? InstanceType::PREFILL : InstanceType::DECODE;
    }
    if (instances_->add_instance(info)) {
      return true;
    }
  }
  for (size_t i = 0; i < completed; ++i) {
    client_.call_unlink_instance(operations[i].first, operations[i].second);
  }
  client_.disconnect_instance(info.name, info.incarnation_id);
  return false;
}

void InstanceController::deregister_instance(
    const std::string& name,
    const std::string& incarnation_id) {
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  const auto current = instances_->find_instance(name);
  if (!current.has_value() ||
      (!incarnation_id.empty() && current->incarnation_id != incarnation_id)) {
    return;
  }
  const auto operations = link_operations(*current, /*removing=*/true);
  for (const auto& operation : operations) {
    client_.call_unlink_instance(operation.first, operation.second);
  }
  // The callback synchronously removes state and enqueues cancellations under
  // the request admission barrier. It is deliberately not an async
  // notification.
  if (retire_(name, current->incarnation_id)) {
    client_.disconnect_instance(name, current->incarnation_id);
  }
}

void InstanceController::reconcile_instance_states() {
  const uint64_t suspect_interval =
      static_cast<uint64_t>(
          std::max(1, options_.detect_disconnected_instance_interval())) *
      1000;
  const uint64_t heartbeat_timeout = static_cast<uint64_t>(
      std::max(1, options_.lease_lost_heartbeat_timeout_ms()));
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (exited_) {
      return;
    }
    const uint64_t now = now_ms();
    const auto snapshot = instances_->snapshot();
    for (const auto& view : snapshot) {
      if (view.runtime_state == InstanceRuntimeState::LEASE_LOST &&
          now - view.latest_timestamp >= heartbeat_timeout) {
        instances_->mark_suspect(
            view.info->name, view.info->incarnation_id, view.latest_timestamp);
      } else if (view.runtime_state == InstanceRuntimeState::SUSPECT &&
                 now - view.suspect_since_ms >= suspect_interval) {
        deregister_instance(view.info->name, view.info->incarnation_id);
      }
    }
  }
}
}  // namespace xllm_service
