/* Copyright 2025-2026 The xLLM Authors.

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

#include "router/policies/cache_aware_routing.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace xllm_service {
namespace {

struct InstanceAffinity {
  int32_t matched_ranks = 0;
  double minimum = 1.0;
};

// The engine chooses the local DP: the current service routing protocol cannot
// pin it. Only credit benefit present on every DP of this exact incarnation.
std::unordered_map<std::string, double> prefix_benefits(
    const std::vector<RoutingCandidate>& candidates,
    const std::vector<KVDirectoryMatch>& matches,
    size_t prompt_tokens,
    double host_hit_weight) {
  std::unordered_map<std::string, const RoutingCandidate*> by_name;
  by_name.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    by_name.emplace(candidate.name, &candidate);
  }
  std::unordered_map<std::string, InstanceAffinity> affinities;
  affinities.reserve(candidates.size());
  for (const auto& match : matches) {
    const auto candidate_it = by_name.find(match.instance_name);
    if (candidate_it == by_name.end() || !match.prefix.synchronized) {
      continue;
    }
    const auto& candidate = *candidate_it->second;
    if (match.dp_rank < 0 || match.dp_rank >= candidate.dp_size ||
        match.incarnation_id != candidate.incarnation_id ||
        (candidate.incarnation_id.empty() &&
         match.register_ts_ms != candidate.register_ts_ms)) {
      continue;
    }
    const double benefit =
        std::clamp((static_cast<double>(match.prefix.device_hit_tokens) +
                    host_hit_weight * match.prefix.host_only_hit_tokens) /
                       static_cast<double>(prompt_tokens),
                   0.0,
                   1.0);
    auto& affinity = affinities[candidate.name];
    ++affinity.matched_ranks;
    affinity.minimum = std::min(affinity.minimum, benefit);
  }
  std::unordered_map<std::string, double> result;
  result.reserve(affinities.size());
  for (const auto& candidate : candidates) {
    const auto found = affinities.find(candidate.name);
    if (found != affinities.end() && candidate.dp_size > 0 &&
        found->second.matched_ranks == candidate.dp_size) {
      result.emplace(candidate.name, found->second.minimum);
    }
  }
  return result;
}

template <typename Cost>
std::string select_candidate(const std::vector<RoutingCandidate>& candidates,
                             uint64_t rotation,
                             Cost cost) {
  if (candidates.empty()) {
    return {};
  }
  uint64_t max_waiting = 0;
  for (const auto& candidate : candidates) {
    max_waiting = std::max(max_waiting, candidate.load.waiting_requests_num);
  }
  double best_score = std::numeric_limits<double>::infinity();
  std::string best;
  const size_t start = rotation % candidates.size();
  for (size_t offset = 0; offset < candidates.size(); ++offset) {
    const auto& candidate = candidates[(start + offset) % candidates.size()];
    const double queue_cost =
        max_waiting == 0
            ? 0.0
            : static_cast<double>(candidate.load.waiting_requests_num) /
                  static_cast<double>(max_waiting);
    const double score = cost(candidate, queue_cost);
    if (score < best_score) {
      best_score = score;
      best = candidate.name;
    }
  }
  return best;
}

}  // namespace

CacheAwareRouting::CacheAwareRouting(double host_hit_weight,
                                     double prefill_lambda)
    : host_hit_weight_(host_hit_weight), prefill_lambda_(prefill_lambda) {
  CHECK(std::isfinite(host_hit_weight_) && host_hit_weight_ >= 0.0 &&
        host_hit_weight_ <= 1.0)
      << "car_host_hit_weight must be in [0, 1]";
  CHECK(std::isfinite(prefill_lambda_) && prefill_lambda_ >= 0.0 &&
        prefill_lambda_ <= 1.0)
      << "car_prefill_lambda must be in [0, 1]";
}

bool CacheAwareRouting::select(const RoutingInput& input,
                               RoutingDecision* decision) {
  auto candidates = input.candidates;
  const auto invalid = [](const RoutingCandidate& candidate) {
    return !candidate.has_load ||
           !std::isfinite(candidate.load.gpu_cache_usage_perc) ||
           candidate.load.gpu_cache_usage_perc < 0 ||
           candidate.load.gpu_cache_usage_perc > 1;
  };
  candidates.prefill.erase(
      std::remove_if(
          candidates.prefill.begin(), candidates.prefill.end(), invalid),
      candidates.prefill.end());
  candidates.decode.erase(
      std::remove_if(
          candidates.decode.begin(), candidates.decode.end(), invalid),
      candidates.decode.end());
  decision->routing = {};
  if (candidates.decode.empty()) {
    // A lone PREFILL cannot generate a response. DEFAULT serves end to end.
    auto& prefill = candidates.prefill;
    prefill.erase(std::remove_if(prefill.begin(),
                                 prefill.end(),
                                 [](const RoutingCandidate& candidate) {
                                   return candidate.type !=
                                          InstanceType::DEFAULT;
                                 }),
                  prefill.end());
  }
  if (candidates.prefill.empty()) {
    return false;
  }
  std::unordered_map<std::string, double> benefits;
  if (input.prompt_tokens != 0) {
    benefits = prefix_benefits(candidates.prefill,
                               input.prefix_matches,
                               input.prompt_tokens,
                               host_hit_weight_);
  }
  const uint64_t rotation =
      next_selection_.fetch_add(1, std::memory_order_relaxed);
  decision->routing.prefill_name = select_candidate(
      candidates.prefill,
      rotation,
      [this, &benefits](const RoutingCandidate& candidate, double queue_cost) {
        const auto found = benefits.find(candidate.name);
        const double kv_hit_ratio =
            found == benefits.end() ? 0.0 : found->second;
        return prefill_lambda_ * (1.0 - kv_hit_ratio) +
               (1.0 - prefill_lambda_) * queue_cost;
      });
  // Decode-side reserved blocks are not a reusable prefill prefix. Minimizing
  // this cost preserves the original decode load ordering.
  decision->routing.decode_name = select_candidate(
      candidates.decode,
      rotation,
      [](const RoutingCandidate& candidate, double queue_cost) {
        return candidate.load.gpu_cache_usage_perc + queue_cost;
      });
  return true;
}

}  // namespace xllm_service
