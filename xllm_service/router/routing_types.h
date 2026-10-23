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

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"
#include "kv_conductor/kv_directory.h"

namespace xllm_service {
struct RoutingCandidate {
  std::string name;
  std::string incarnation_id;
  uint64_t register_ts_ms = 0;
  int32_t dp_size = 1;
  InstanceType type = InstanceType::DEFAULT;
  LoadMetrics load;
  bool has_load = false;
  int64_t prefill_time = 0;
  int64_t estimated_ttft = 0;
  int64_t estimated_tpot = 0;
};

struct RoutingCandidates {
  std::vector<RoutingCandidate> prefill;
  std::vector<RoutingCandidate> decode;
};

struct RoutingInput {
  RoutingCandidates candidates;
  std::vector<KVDirectoryMatch> prefix_matches;
  size_t prompt_tokens = 0;
};

struct RoutingDecision {
  Routing routing;
  int64_t estimated_ttft = 0;
  bool flip_prefill_to_decode = false;
};
}  // namespace xllm_service
