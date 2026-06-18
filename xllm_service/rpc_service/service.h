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

#include <mutex>
#include <unordered_map>

#include "chat.pb.h"
#include "common/options.h"
#include "common/types.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "completion.pb.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class Scheduler;
class InstanceMgr;

class XllmRpcServiceImpl final {
 public:
  XllmRpcServiceImpl(const Options& options, Scheduler* scheduler);
  ~XllmRpcServiceImpl();

  bool heartbeat(const proto::HeartbeatRequest* req);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  std::vector<std::string> get_static_decode_list(
      const std::string& prefill_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& decode_name);

 public:
  // handle generations from prefill/decode instance
  bool handle_generation(const llm::RequestOutput& request_output);

 private:
  Options options_;

  // not own
  Scheduler* scheduler_;

  // In disagg pd mode, we support receive generated token from
  // prefill or from decode directly.
  // 1.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp--- [prefill] ---first resp---> [decode]
  // [service] <---resp--- [prefill] <---resp--- [decode]
  //
  // 2.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp-- [prefill] --first resp---> [decode]
  // [service] <---resp-- [decode]
  //
  bool enable_decode_response_to_service_ = false;
};

// parse proto data and call XllmRpcService
class XllmRpcService : public proto::XllmRpcService {
 public:
  explicit XllmRpcService(const Options& options, Scheduler* scheduler);
  virtual ~XllmRpcService();

  virtual void Hello(google::protobuf::RpcController* cntl_base,
                     const proto::Empty* req,
                     proto::Status* resp,
                     google::protobuf::Closure* done) override;

  virtual void Heartbeat(google::protobuf::RpcController* cntl_base,
                         const proto::HeartbeatRequest* req,
                         proto::Status* resp,
                         google::protobuf::Closure* done) override;

  virtual void GetInstanceInfo(google::protobuf::RpcController* cntl_base,
                               const proto::InstanceID* req,
                               proto::InstanceMetaInfo* resp,
                               google::protobuf::Closure* done) override;

  virtual void GetStaticDecodeList(google::protobuf::RpcController* cntl_base,
                                   const proto::InstanceID* req,
                                   proto::InstanceIDs* resp,
                                   google::protobuf::Closure* done) override;

  virtual void GetStaticPrefillList(google::protobuf::RpcController* cntl_base,
                                    const proto::InstanceID* req,
                                    proto::InstanceIDs* resp,
                                    google::protobuf::Closure* done) override;

  // xllm service receive response from decode instance directly in disagg pd
  // mode. This can eliminate the cost brought by forwarding through prefill.
  virtual void Generations(google::protobuf::RpcController* cntl_base,
                           const proto::DisaggStreamGenerations* req,
                           proto::StatusSet* resp,
                           google::protobuf::Closure* done) override;

 private:
  std::unique_ptr<XllmRpcServiceImpl> xllm_rpc_service_impl_;
};

}  // namespace xllm_service
