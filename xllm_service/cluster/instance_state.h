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

#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cluster/instance_types.h"

namespace xllm_service {
// Thread-safe facts about registered instances. No engine I/O or request state.
class InstanceState final {
 public:
  bool add_instance(InstanceMetaInfo info);
  bool refresh_instance(InstanceMetaInfo info);
  bool remove_instance(const std::string& name,
                       const std::string& incarnation_id,
                       InstanceMetaInfo* removed);
  std::optional<InstanceMetaInfo> find_instance(const std::string& name) const;
  InstanceMetaInfo get_instance_info(const std::string& name) const;
  std::vector<InstanceView> snapshot() const;
  bool matches(const Routing& routing,
               const std::string& prefill_incarnation,
               const std::string& decode_incarnation) const;
  bool contains_instance(const std::string& name,
                         const std::string& incarnation_id) const;
  bool has_available_instances() const;
  std::vector<std::string> get_static_decode_list(
      const std::string& name) const;
  std::vector<std::string> get_static_prefill_list(
      const std::string& name) const;
  bool record_heartbeat(const std::string& name,
                        const std::string& incarnation_id,
                        LoadMetrics load,
                        LatencyMetrics latency);
  void apply_load_event(const LoadEvent& event);
  LoadUpdates take_load_updates();
  bool mark_lease_lost(const std::string& name,
                       const std::string& incarnation_id);
  bool mark_suspect(const std::string& name,
                    const std::string& incarnation_id,
                    uint64_t expected_timestamp);
  bool change_role(const std::string& name,
                   const std::string& incarnation_id,
                   InstanceType role);

 private:
  struct Entry {
    InstanceView view;
    LatencyMetrics latency{0, 0};
    size_t role_index = 0;
    bool load_changed = false;
  };
  static InstanceMetaInfo materialize(const Entry& entry);
  void remove_from_index(const std::string& name, const Entry& entry);
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, Entry> instances_;
  std::vector<std::string> prefill_;
  std::vector<std::string> decode_;
  std::unordered_set<std::string> removed_loads_;
};
}  // namespace xllm_service
