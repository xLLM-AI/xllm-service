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

#include "rpc_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/closure_guard.h>

#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "rpc_service/disagg_generation_adapter.h"
#include "scheduler/scheduler.h"

namespace xllm_service {

XllmRpcServiceImpl::XllmRpcServiceImpl(const Options& options,
                                       Scheduler* scheduler)
    : options_(options), scheduler_(scheduler) {}

XllmRpcServiceImpl::~XllmRpcServiceImpl() { scheduler_->exited(); }

bool XllmRpcServiceImpl::heartbeat(const proto::HeartbeatRequest* req) {
  return scheduler_->handle_instance_heartbeat(req);
}

InstanceMetaInfo XllmRpcServiceImpl::get_instance_info(
    const std::string& instance_name) {
  return scheduler_->get_instance_info(instance_name);
}

std::vector<std::string> XllmRpcServiceImpl::get_static_decode_list(
    const std::string& instance_name) {
  return scheduler_->get_static_decode_list(instance_name);
}

std::vector<std::string> XllmRpcServiceImpl::get_static_prefill_list(
    const std::string& instance_name) {
  return scheduler_->get_static_prefill_list(instance_name);
}

bool XllmRpcServiceImpl::handle_generation(
    const llm::RequestOutput& request_output) {
  return scheduler_->handle_generation(request_output);
}

XllmRpcService::XllmRpcService(const Options& options, Scheduler* scheduler) {
  xllm_rpc_service_impl_ =
      std::make_unique<XllmRpcServiceImpl>(options, scheduler);
}

XllmRpcService::~XllmRpcService() {}

void XllmRpcService::Hello(google::protobuf::RpcController* cntl_base,
                           const proto::Empty* req,
                           proto::Status* resp,
                           google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  resp->set_ok(true);
}

void XllmRpcService::GetInstanceInfo(google::protobuf::RpcController* cntl_base,
                                     const proto::InstanceID* req,
                                     proto::InstanceMetaInfo* resp,
                                     google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  InstanceMetaInfo metainfo =
      xllm_rpc_service_impl_->get_instance_info(req->name());
  resp->set_name(metainfo.name);
  resp->set_rpc_address(metainfo.rpc_address);
  resp->set_incarnation_id(metainfo.incarnation_id);
  resp->set_register_ts_ms(metainfo.register_ts_ms);
  if (metainfo.type == InstanceType::PREFILL) {
    resp->set_type(proto::InstanceType::PREFILL);
  } else if (metainfo.type == InstanceType::DECODE) {
    resp->set_type(proto::InstanceType::DECODE);
  } else if (metainfo.type == InstanceType::MIX) {
    resp->set_type(proto::InstanceType::MIX);
  } else {
    resp->set_type(proto::InstanceType::DEFAULT);
  }
  for (auto& cluster_id : metainfo.cluster_ids) {
    *(resp->mutable_cluster_ids()->Add()) = cluster_id;
  }
  for (auto& addr : metainfo.addrs) {
    *(resp->mutable_addrs()->Add()) = addr;
  }
  resp->set_dp_size(metainfo.dp_size);
  resp->set_kv_split_size(metainfo.kv_split_size);
  for (auto& port : metainfo.ports) {
    resp->add_ports(port);
  }
}

void XllmRpcService::Heartbeat(google::protobuf::RpcController* cntl_base,
                               const proto::HeartbeatRequest* req,
                               proto::Status* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  resp->set_ok(xllm_rpc_service_impl_->heartbeat(req));
}

void XllmRpcService::GetStaticDecodeList(
    google::protobuf::RpcController* cntl_base,
    const proto::InstanceID* req,
    proto::InstanceIDs* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<std::string> decode_list =
      xllm_rpc_service_impl_->get_static_decode_list(req->name());
  for (auto& d : decode_list) {
    *(resp->mutable_names()->Add()) = std::move(d);
  }
}

void XllmRpcService::GetStaticPrefillList(
    google::protobuf::RpcController* cntl_base,
    const proto::InstanceID* req,
    proto::InstanceIDs* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<std::string> prefill_list =
      xllm_rpc_service_impl_->get_static_prefill_list(req->name());
  for (auto& p : prefill_list) {
    *(resp->mutable_names()->Add()) = std::move(p);
  }
}

void XllmRpcService::Generations(google::protobuf::RpcController* cntl_base,
                                 const proto::DisaggStreamGenerations* req,
                                 proto::StatusSet* resp,
                                 google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  // TODO: use threadpool here
  for (const proto::DisaggStreamGeneration& generation : req->gens()) {
    RequestOutputConversionResult conversion =
        request_output_from_disagg_generation(generation);
    proto::Status* status = resp->mutable_all_status()->Add();
    if (!conversion.status.ok()) {
      LOG(ERROR) << "Rejecting invalid generation for request "
                 << generation.req_id() << ": " << conversion.status.message();
      status->set_ok(false);
      continue;
    }
    status->set_ok(
        xllm_rpc_service_impl_->handle_generation(conversion.output.value()));
  }
}

}  // namespace xllm_service
