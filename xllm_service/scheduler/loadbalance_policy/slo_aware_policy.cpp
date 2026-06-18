/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "slo_aware_policy.h"

#include "common/global_gflags.h"

namespace xllm_service {

SloAwarePolicy::SloAwarePolicy(const Options& options,
                               std::shared_ptr<InstanceMgr> instance_mgr)
    : options_(options), LoadBalancePolicy(instance_mgr) {}

bool SloAwarePolicy::select_instances_pair(std::shared_ptr<Request> request) {
  if (request->token_ids.empty()) {
    return instance_mgr_->get_next_instance_pair(&request->routing);
  }

  // select instances pair based on slo
  if (!instance_mgr_->select_instance_pair_on_slo(request)) {
    LOG(ERROR) << "Select instances based on the SLO failed!";
    return false;
  }

  return true;
}

}  // namespace xllm_service