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

#include "cluster/instance_state.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <utility>

namespace {
uint64_t now_ms() {
  return static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
}
}  // namespace
namespace xllm_service {
InstanceMetaInfo InstanceState::materialize(const Entry& entry) {
  InstanceMetaInfo result = *entry.view.info;
  result.runtime_state = entry.view.runtime_state;
  result.latest_timestamp = entry.view.latest_timestamp;
  result.current_type = entry.view.role;
  result.instance_index = entry.role_index;
  return result;
}

bool InstanceState::add_instance(InstanceMetaInfo info) {
  std::unique_lock lock(mutex_);
  if (instances_.count(info.name) != 0) {
    return false;
  }
  InstanceType role =
      info.type == InstanceType::DEFAULT ? InstanceType::PREFILL : info.type;
  if (info.type == InstanceType::MIX) {
    role = info.current_type;
  }
  auto& index = role == InstanceType::DECODE ? decode_ : prefill_;
  Entry entry;
  entry.view.role = role;
  entry.view.runtime_state = info.runtime_state;
  if (info.runtime_state == InstanceRuntimeState::SUSPECT) {
    entry.view.suspect_since_ms = now_ms();
  }
  entry.view.latest_timestamp = now_ms();
  entry.role_index = index.size();
  const std::string name = info.name;
  entry.view.info = std::make_shared<const InstanceMetaInfo>(std::move(info));
  instances_.emplace(name, std::move(entry));
  index.emplace_back(name);
  return true;
}

bool InstanceState::refresh_instance(InstanceMetaInfo info) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(info.name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != info.incarnation_id) {
    return false;
  }
  auto& view = found->second.view;
  view.info = std::make_shared<const InstanceMetaInfo>(std::move(info));
  view.runtime_state = InstanceRuntimeState::ACTIVE;
  view.latest_timestamp = now_ms();
  view.suspect_since_ms = 0;
  return true;
}

void InstanceState::remove_from_index(const std::string& name,
                                      const Entry& entry) {
  auto& index = entry.view.role == InstanceType::DECODE ? decode_ : prefill_;
  const size_t position = entry.role_index;
  CHECK_LT(position, index.size());
  CHECK_EQ(index[position], name);
  std::swap(index[position], index.back());
  instances_.at(index[position]).role_index = position;
  index.pop_back();
}

bool InstanceState::remove_instance(const std::string& name,
                                    const std::string& incarnation_id,
                                    InstanceMetaInfo* removed) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != incarnation_id) {
    return false;
  }
  *removed = materialize(found->second);
  remove_from_index(name, found->second);
  instances_.erase(found);
  removed_loads_.insert(name);
  return true;
}

std::optional<InstanceMetaInfo> InstanceState::find_instance(
    const std::string& name) const {
  std::shared_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end()) {
    return std::nullopt;
  }
  return materialize(found->second);
}

InstanceMetaInfo InstanceState::get_instance_info(
    const std::string& name) const {
  const auto found = find_instance(name);
  if (found.has_value()) {
    return *found;
  }
  LOG(ERROR) << "Instance is not registered: " << name;
  InstanceMetaInfo empty;
  empty.dp_size = 1;
  empty.kv_split_size = 1;
  return empty;
}

std::vector<InstanceView> InstanceState::snapshot() const {
  std::shared_lock lock(mutex_);
  std::vector<InstanceView> result;
  result.reserve(instances_.size());
  for (const auto& name : prefill_) {
    result.emplace_back(instances_.at(name).view);
  }
  for (const auto& name : decode_) {
    result.emplace_back(instances_.at(name).view);
  }
  return result;
}

bool InstanceState::matches(const Routing& routing,
                            const std::string& prefill_incarnation,
                            const std::string& decode_incarnation) const {
  std::shared_lock lock(mutex_);
  const auto matches_one = [this](const std::string& name,
                                  const std::string& incarnation) {
    if (name.empty()) {
      return true;
    }
    const auto found = instances_.find(name);
    return found != instances_.end() &&
           found->second.view.info->incarnation_id == incarnation &&
           found->second.view.runtime_state != InstanceRuntimeState::SUSPECT;
  };
  return matches_one(routing.prefill_name, prefill_incarnation) &&
         matches_one(routing.decode_name, decode_incarnation);
}

bool InstanceState::contains_instance(const std::string& name,
                                      const std::string& incarnation_id) const {
  std::shared_lock lock(mutex_);
  const auto found = instances_.find(name);
  return found != instances_.end() &&
         found->second.view.info->incarnation_id == incarnation_id;
}

bool InstanceState::has_available_instances() const {
  const auto views = snapshot();
  bool prefill = false;
  bool decode = false;
  bool mix_prefill = false;
  bool mix_decode = false;
  for (const auto& view : views) {
    if (view.runtime_state == InstanceRuntimeState::SUSPECT) {
      continue;
    }
    const InstanceType type = view.info->type;
    if (type == InstanceType::DEFAULT) {
      return true;
    }
    prefill |= type == InstanceType::PREFILL;
    decode |= type == InstanceType::DECODE;
    mix_prefill |=
        type == InstanceType::MIX && view.role == InstanceType::PREFILL;
    mix_decode |=
        type == InstanceType::MIX && view.role == InstanceType::DECODE;
  }
  return (prefill && decode) || (mix_prefill && mix_decode);
}

std::vector<std::string> InstanceState::get_static_decode_list(
    const std::string& /*name*/) const {
  const auto views = snapshot();
  std::vector<std::string> result;
  result.reserve(views.size());
  for (const auto& view : views) {
    if (view.info->type != InstanceType::DECODE ||
        view.runtime_state == InstanceRuntimeState::SUSPECT) {
      continue;
    }
    result.emplace_back(view.info->name);
  }
  return result;
}

std::vector<std::string> InstanceState::get_static_prefill_list(
    const std::string& /*name*/) const {
  const auto views = snapshot();
  std::vector<std::string> result;
  result.reserve(views.size());
  for (const auto& view : views) {
    if ((view.info->type != InstanceType::PREFILL &&
         view.info->type != InstanceType::DEFAULT) ||
        view.runtime_state == InstanceRuntimeState::SUSPECT) {
      continue;
    }
    result.emplace_back(view.info->name);
  }
  return result;
}

bool InstanceState::record_heartbeat(const std::string& name,
                                     const std::string& incarnation_id,
                                     LoadMetrics load,
                                     LatencyMetrics latency) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != incarnation_id) {
    return false;
  }
  auto& entry = found->second;
  entry.view.latest_timestamp = now_ms();
  entry.view.load = std::move(load);
  entry.latency = std::move(latency);
  entry.load_changed = true;
  if (entry.view.runtime_state == InstanceRuntimeState::SUSPECT) {
    entry.view.runtime_state = InstanceRuntimeState::LEASE_LOST;
    entry.view.suspect_since_ms = 0;
  }
  return true;
}

void InstanceState::apply_load_event(const LoadEvent& event) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(event.name);
  if (found == instances_.end()) {
    return;
  }
  if (event.deleted) {
    found->second.view.load.reset();
  } else {
    found->second.view.load = event.metrics;
  }
}

LoadUpdates InstanceState::take_load_updates() {
  std::unique_lock lock(mutex_);
  LoadUpdates result;
  result.changed.reserve(instances_.size());
  for (auto& [name, entry] : instances_) {
    if (!entry.load_changed || !entry.view.load.has_value()) {
      continue;
    }
    result.changed.emplace(name, *entry.view.load);
    entry.load_changed = false;
  }
  result.removed.swap(removed_loads_);
  // A new incarnation's report supersedes a pending removal at this address.
  for (const auto& [name, load] : result.changed) {
    result.removed.erase(name);
  }
  return result;
}

bool InstanceState::mark_lease_lost(const std::string& name,
                                    const std::string& incarnation_id) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != incarnation_id) {
    return false;
  }
  auto& view = found->second.view;
  view.runtime_state = InstanceRuntimeState::LEASE_LOST;
  view.latest_timestamp = now_ms();
  view.suspect_since_ms = 0;
  return true;
}

bool InstanceState::mark_suspect(const std::string& name,
                                 const std::string& incarnation_id,
                                 uint64_t expected_timestamp) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != incarnation_id ||
      found->second.view.latest_timestamp != expected_timestamp) {
    return false;
  }
  auto& view = found->second.view;
  if (view.runtime_state != InstanceRuntimeState::SUSPECT) {
    view.suspect_since_ms = now_ms();
  }
  view.runtime_state = InstanceRuntimeState::SUSPECT;
  return true;
}

bool InstanceState::change_role(const std::string& name,
                                const std::string& incarnation_id,
                                InstanceType role) {
  std::unique_lock lock(mutex_);
  const auto found = instances_.find(name);
  if (found == instances_.end() ||
      found->second.view.info->incarnation_id != incarnation_id ||
      found->second.view.info->type != InstanceType::MIX ||
      found->second.view.role == role ||
      (role != InstanceType::PREFILL && role != InstanceType::DECODE)) {
    return false;
  }
  auto& entry = found->second;
  const auto& previous =
      entry.view.role == InstanceType::DECODE ? decode_ : prefill_;
  const size_t remaining = std::count_if(
      previous.begin(), previous.end(), [this](const std::string& other) {
        return instances_.at(other).view.runtime_state !=
               InstanceRuntimeState::SUSPECT;
      });
  if (remaining <= 1) {
    return false;
  }
  remove_from_index(name, entry);
  auto& next = role == InstanceType::DECODE ? decode_ : prefill_;
  entry.role_index = next.size();
  entry.view.role = role;
  next.emplace_back(name);
  return true;
}
}  // namespace xllm_service
