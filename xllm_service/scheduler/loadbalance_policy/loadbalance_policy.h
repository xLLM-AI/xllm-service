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

#pragma once

#include "common/types.h"
#include "request/request.h"
#include "scheduler/managers/instance_mgr.h"

namespace xllm_service {

class LoadBalancePolicy {
 public:
  LoadBalancePolicy(std::shared_ptr<InstanceMgr> instance_mgr)
      : instance_mgr_(instance_mgr) {}

  virtual ~LoadBalancePolicy() = default;

  virtual bool select_instances_pair(std::shared_ptr<Request> request) = 0;

 protected:
  std::shared_ptr<InstanceMgr> instance_mgr_;
};

}  // namespace xllm_service
