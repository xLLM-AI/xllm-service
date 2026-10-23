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

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"

namespace xllm_service {
struct InstanceEvent {
  std::string name;
  InstanceMetaInfo info;
  bool deleted = false;
};
struct LoadEvent {
  std::string name;
  LoadMetrics metrics;
  bool deleted = false;
};
// Immutable registration data is shared by snapshots. Heartbeat/role updates
// do not copy model profiling vectors on the routing path.
struct InstanceView {
  std::shared_ptr<const InstanceMetaInfo> info;
  std::optional<LoadMetrics> load;
  InstanceType role = InstanceType::PREFILL;
  InstanceRuntimeState runtime_state = InstanceRuntimeState::ACTIVE;
  uint64_t latest_timestamp = 0;
  uint64_t suspect_since_ms = 0;
};
struct LoadUpdates {
  std::unordered_map<std::string, LoadMetrics> changed;
  std::unordered_set<std::string> removed;
};
}  // namespace xllm_service
