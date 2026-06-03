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

#include "rpc_service/client.h"

#include <glog/logging.h>

namespace {
// magic number, TODO: move to config file or env var
constexpr int32_t kHeartbeatInterval = 3;  // in seconds
}  // namespace

namespace xllm_service {

XllmRpcClient::XllmRpcClient(const std::string& instace_name,
                             const std::string& master_addr,
                             const ChannelOptions& options)
    : instance_name_(instace_name), master_addr_(master_addr) {
  brpc::ChannelOptions chan_options;
  chan_options.protocol = options.protocol;
  chan_options.connection_type = options.connection_type;
  chan_options.timeout_ms = options.timeout_ms /*milliseconds*/;
  chan_options.max_retry = options.max_retry;
  if (master_channel_.Init(master_addr_.c_str(),
                           options.load_balancer.c_str(),
                           &chan_options) != 0) {
    LOG(ERROR) << "Fail to initialize brpc channel to server " << master_addr_;
    return;
  }
  master_stub_ = std::make_unique<proto::XllmRpcService_Stub>(&master_channel_);

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&XllmRpcClient::heartbeat, this);
}

XllmRpcClient::~XllmRpcClient() {
  exited_ = true;
  if (heartbeat_thread_) {
    heartbeat_thread_->join();
  }
}

// TODO: send metainfo/metrics to master ?
void XllmRpcClient::heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));
    if (!register_inst_done_) continue;

    brpc::Controller cntl;
    proto::HeartbeatRequest req;
    req.set_name(instance_name_);
    // TODO: set req.cache_event and req.load_metrics
    proto::Status res;
    master_stub_->Heartbeat(&cntl, &req, &res, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << instance_name_
                 << " failed to send heartbeat to master: " << cntl.ErrorText();
      ;
    } else if (!res.ok()) {
      LOG(ERROR) << instance_name_
                 << " failed to send heartbeat to master, status: " << res.ok();
    }
  }
}

ErrorCode XllmRpcClient::register_instance() {
  InstanceMetaInfo metainfo;
  metainfo.name = instance_name_;
  return register_instance(metainfo);
}

ErrorCode XllmRpcClient::register_instance(const InstanceMetaInfo& metainfo) {
  brpc::Controller cntl;
  proto::InstanceMetaInfo req;
  req.set_name(metainfo.name);
  if (metainfo.type == InstanceType::PREFILL) {
    req.set_type(proto::InstanceType::PREFILL);
  } else if (metainfo.type == InstanceType::DECODE) {
    req.set_type(proto::InstanceType::DECODE);
  } else if (metainfo.type == InstanceType::MIX) {
    req.set_type(proto::InstanceType::MIX);
  } else {
    req.set_type(proto::InstanceType::DEFAULT);
  }
  for (auto& cluster_id : metainfo.cluster_ids) {
    *(req.mutable_cluster_ids()->Add()) = cluster_id;
  }
  for (auto& addr : metainfo.addrs) {
    *(req.mutable_addrs()->Add()) = addr;
  }
  req.set_dp_size(metainfo.dp_size);
  req.set_kv_split_size(metainfo.kv_split_size);
  for (auto& port : metainfo.ports) {
    req.add_ports(port);
  }
  proto::StatusCode res;
  master_stub_->RegisterInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << instance_name_
               << " failed to send register_instance to master: "
               << cntl.ErrorText();
    ;
  } else if (res.status_code() != ConvertErrorCode::to_int(ErrorCode::OK)) {
    LOG(ERROR) << instance_name_
               << " failed to send register_instance to master: "
               << "res = " << res.status_code();
  } else {
    // register instance success
    register_inst_done_ = true;
  }
  return ConvertErrorCode::from_int(res.status_code());
}

}  // namespace xllm_service
