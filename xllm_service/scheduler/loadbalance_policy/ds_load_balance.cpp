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

#include "scheduler/loadbalance_policy/ds_load_balance.h"

#include <utility>

namespace xllm_service {

DsLoadBalance::DsLoadBalance(
    std::shared_ptr<InstanceMgr> instance_mgr,
    std::unique_ptr<MooncakePrefixLookup> prefix_lookup)
    : LoadBalancePolicy(std::move(instance_mgr)),
      prefix_lookup_(std::move(prefix_lookup)) {}

bool DsLoadBalance::select_instances_pair(std::shared_ptr<Request> request) {
  // /v1/models uses the same scheduler with no generation payload.
  if (request->token_ids.empty()) {
    return instance_mgr_->get_next_instance_pair(&request->routing);
  }
  // Network IO occurs before taking the manager's topology/metrics locks.
  // Selection discards hints if an engine has since changed incarnation.
  const auto hints =
      prefix_lookup_
          ? prefix_lookup_->match(request->token_ids,
                                  instance_mgr_->get_instance_snapshot())
          : std::unordered_map<std::string, PrefillCacheHint>{};
  return instance_mgr_->select_instance_pair_on_ds_load_balance(request, hints);
}

}  // namespace xllm_service
