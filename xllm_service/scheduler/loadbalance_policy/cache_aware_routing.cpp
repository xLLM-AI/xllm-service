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

#include "cache_aware_routing.h"

namespace xllm_service {

constexpr float MIN_SCORE = -2.0;

bool CacheAwareRouting::select_instances_pair(
    std::shared_ptr<Request> request) {
  LoadBalanceInfos lb_infos;
  if (!request->token_ids.empty()) {
    Slice<int32_t> token_ids(request->token_ids.data(),
                             request->token_ids.size());
    global_kvcache_mgr_->match(token_ids, &lb_infos.overlap_scores);
    DLOG(INFO) << lb_infos.debug_string();
  }

  instance_mgr_->get_load_metrics(&lb_infos);
  DLOG(INFO) << lb_infos.debug_string();

  if (lb_infos.prefill_load_metrics.size() == 0) {
    LOG(INFO) << "No node available!";
    return false;
  }

  // find preifll
  cost_function(lb_infos.overlap_scores.hbm_instance_score,
                lb_infos.overlap_scores.max_block_num,
                lb_infos.prefill_load_metrics,
                lb_infos.prefill_max_waiting_requests_num,
                &request->routing.prefill_name);

  // find decode
  if (lb_infos.decode_load_metrics.size()) {
    cost_function(lb_infos.overlap_scores.hbm_instance_score,
                  lb_infos.overlap_scores.max_block_num,
                  lb_infos.decode_load_metrics,
                  lb_infos.decode_max_waiting_requests_num,
                  &request->routing.decode_name);
  }

  return true;
}

void CacheAwareRouting::cost_function(
    const std::unordered_map<std::string, uint32_t>& overlap_scores,
    const uint32_t& max_block_num,
    const std::unordered_map<std::string, LoadMetrics>& load_metrics,
    const int64_t& max_waiting_requests_num,
    std::string* best_choice) {
  float best_score = MIN_SCORE;
  for (const auto& it : load_metrics) {
    const auto matched_blocks_it = overlap_scores.find(it.first);
    uint32_t matched_blocks = 0;
    if (matched_blocks_it != overlap_scores.end()) {
      matched_blocks = matched_blocks_it->second;
    }

    auto score =
        (max_block_num == 0 ? 0 : matched_blocks / max_block_num) -
        it.second.gpu_cache_usage_perc -
        (max_waiting_requests_num == 0
             ? 0
             : it.second.waiting_requests_num / max_waiting_requests_num);

    if (score > best_score) {
      best_score = score;
      *best_choice = it.first;
    }
  }
}

}  // namespace xllm_service
