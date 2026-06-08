/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "chat_template/jinja_chat_template.h"
#include "common/call_data.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/xllm/output.h"
#include "etcd_client/etcd_client.h"
#include "loadbalance_policy/loadbalance_policy.h"
#include "managers/global_kvcache_mgr.h"
#include "managers/instance_mgr.h"
#include "request/request.h"
#include "response_handler.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

// A scheduler for scheduling requests and instances
class Scheduler final {
 public:
  Scheduler(const Options& options);
  ~Scheduler();

  bool schedule(std::shared_ptr<Request> request);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& target_name);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  bool handle_instance_heartbeat(const proto::HeartbeatRequest* req);

  void exited() { exited_ = true; }

  // Returns true if at least one valid instance group is available.
  bool has_available_instances() const;

  // register new requests from http service
  // keep http callback util request finished.
  // `handle_generation` will handle response with these callbacks.
  bool record_new_request(std::shared_ptr<ChatCallData> call_data,
                          std::shared_ptr<Request> request);
  bool record_new_request(std::shared_ptr<AnthropicCallData> call_data,
                          std::shared_ptr<Request> request);
  bool record_new_request(std::shared_ptr<CompletionCallData> call_data,
                          std::shared_ptr<Request> request);
  void finish_request(const std::string& service_request_id,
                      bool error = false);

  void clear_requests_on_failed_instance(const std::string& instance_name,
                                         const std::string& incarnation_id,
                                         InstanceType type);

  // handle generations from prefill/decode instance
  bool handle_generation(const llm::RequestOutput& request_output);

  // update request metrics for prefill finished request
  void update_request_metrics(std::shared_ptr<Request> request,
                              bool finished_on_prefill_instance);

  // update token latency metrics
  void update_token_latency_metrics(std::shared_ptr<Request> request,
                                    bool finished_on_prefill_instance);

 private:
  DISALLOW_COPY_AND_ASSIGN(Scheduler);

  void update_master_service_heartbeat();

  bool register_current_service();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);

  void handle_xservice_watch(const etcd::Response& response,
                             const uint64_t& prefix_len);

  Tokenizer* get_tls_tokenizer();

 private:
  Options options_;

  bool exited_ = false;

  bool is_master_service_ = false;

  TokenizerArgs tokenizer_args_;

  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::unique_ptr<Tokenizer> tokenizer_;

  std::shared_ptr<InstanceMgr> instance_mgr_;

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;

  std::unique_ptr<LoadBalancePolicy> lb_policy_;

  std::unique_ptr<std::thread> heartbeat_thread_;

  // `service request id` -> `request` map
  std::unordered_map<std::string, std::shared_ptr<Request>> requests_;
  std::mutex request_mutex_;

  // use threadpool to handle all RequestOuputs queue
  static constexpr size_t kOutputTheadNum_ = 128;  // magic num
  ThreadPool output_threadpools_[kOutputTheadNum_];
  // A request will be handled in the same thread to guarantee the token's
  // order.
  std::unordered_map<std::string, size_t> remote_requests_output_thread_map_;
  size_t next_thread_idx = 0;
  std::mutex thread_map_mutex_;

  // used when receive token from decode instance.
  ResponseHandler response_handler_;
};

}  // namespace xllm_service
