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
#include <mutex>
#include <unordered_map>

#include "request/output_dispatcher.h"
#include "request/request.h"
namespace xllm_service {
class InstanceState;
class RoutingState;
class RequestManager final {
 public:
  RequestManager(std::shared_ptr<InstanceState> instances,
                 std::shared_ptr<RoutingState> routing);
  ~RequestManager();
  bool publish_request(std::shared_ptr<Request> request);
  void finish_request(const std::string& request_id, bool error = false);
  bool handle_generation(llm::RequestOutput output);
  void clear_requests_on_failed_instance(const std::string& name,
                                         const std::string& incarnation_id,
                                         InstanceType type);
  bool retire_instance(const std::string& name,
                       const std::string& incarnation_id);
  void drain();
  void stop();

 private:
  friend class ServiceTestPeer;
  bool update_request_metrics(const std::shared_ptr<Request>& request,
                              bool finished_on_prefill_instance);
  void update_token_latency_metrics(const std::shared_ptr<Request>& request,
                                    bool first_token);
  std::shared_ptr<InstanceState> instances_;
  std::shared_ptr<RoutingState> routing_;
  std::mutex admission_mutex_;
  std::unordered_map<std::string, std::shared_ptr<Request>> requests_;
  std::mutex request_mutex_;
  bool stopped_ = false;
  // Last member: output tasks finish while the request table and routing state
  // live.
  OutputDispatcher outputs_;
};
}  // namespace xllm_service
