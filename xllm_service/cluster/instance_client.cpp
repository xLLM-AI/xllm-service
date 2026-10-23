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

#include "cluster/instance_client.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <thread>

#include "disagg_pd.pb.h"
#include "xllm_service.pb.h"

namespace {
constexpr char kHealthPath[] = "/health";
constexpr int64_t kDeleteProbeRetryBackoffMs = 100;
}  // namespace
namespace xllm_service {
InstanceClient::InstanceClient(const ServiceConfig& options)
    : options_(options) {}

bool InstanceClient::connect_instance(const InstanceMetaInfo& info) {
  std::shared_ptr<brpc::Channel> channel;
  if (!init_brpc_channel(info.name, info, &channel)) {
    return false;
  }
  std::lock_guard lock(mutex_);
  return channels_
      .emplace(info.name, Connection{info.incarnation_id, std::move(channel)})
      .second;
}

void InstanceClient::disconnect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  std::shared_ptr<brpc::Channel> retired;
  {
    std::lock_guard lock(mutex_);
    const auto found = channels_.find(name);
    if (found == channels_.end() ||
        found->second.incarnation_id != incarnation_id) {
      return;
    }
    retired = std::move(found->second.channel);
    channels_.erase(found);
  }
  // Release the transport after the cache lock has been dropped.
}

std::shared_ptr<brpc::Channel> InstanceClient::get_channel(
    const std::string& name,
    const std::string& incarnation_id) const {
  std::lock_guard lock(mutex_);
  const auto found = channels_.find(name);
  if (found == channels_.end() ||
      found->second.incarnation_id != incarnation_id) {
    return nullptr;
  }
  return found->second.channel;
}

bool InstanceClient::init_brpc_channel(
    const std::string& instance_name,
    const InstanceMetaInfo& info,
    std::shared_ptr<brpc::Channel>* out_channel) {
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  if (info.backend_type == "vllm") {
    // Pass the bare host:port to Init: brpc treats any "scheme://" prefix as a
    // naming service, and "http" is not one, so a "http://" target fails with
    // "Unknown naming service". options.protocol selects HTTP instead.
    options.protocol = "http";
    options.timeout_ms = options_.vllm_http_timeout_ms();
    options.max_retry = 0;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  } else {
    options.timeout_ms = options_.timeout_ms();
    options.max_retry = 3;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  }
  std::string load_balancer = "";
  if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize channel for " << instance_name
               << " (backend=" << info.backend_type << ")";
    return false;
  }
  *out_channel = std::move(channel);
  return true;
}

bool InstanceClient::probe_instance_health(const std::string& instance_name) {
  const int32_t attempts =
      std::max(1, options_.instance_delete_probe_attempts());
  const int32_t timeout_ms =
      std::max(1, options_.instance_delete_probe_timeout_ms());
  const std::string url = "http://" + instance_name + kHealthPath;

  for (int32_t attempt = 1; attempt <= attempts; ++attempt) {
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = timeout_ms;
    options.connect_timeout_ms = timeout_ms;
    options.max_retry = 0;
    if (channel.Init(url.c_str(), "", &options) != 0) {
      LOG(WARNING) << "Failed to initialize health probe channel, instance: "
                   << instance_name << ", attempt: " << attempt << "/"
                   << attempts;
    } else {
      brpc::Controller cntl;
      cntl.http_request().uri() = url;
      cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
      channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
      if (!cntl.Failed() && cntl.http_response().status_code() == 200) {
        return true;
      }
      LOG(WARNING) << "Health probe failed, instance: " << instance_name
                   << ", attempt: " << attempt << "/" << attempts << ", error: "
                   << (cntl.Failed() ? cntl.ErrorText()
                                     : std::to_string(
                                           cntl.http_response().status_code()));
    }

    if (attempt < attempts) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDeleteProbeRetryBackoffMs));
    }
  }

  return false;
}

bool InstanceClient::call_link_instance(const std::string& target_rpc_addr,
                                        const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceClient::call_unlink_instance(const std::string& target_rpc_addr,
                                          const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

void InstanceClient::generate(const std::shared_ptr<brpc::Channel>& channel,
                              brpc::Controller* controller,
                              const xllm::proto::ChatRequest& request,
                              google::protobuf::Closure* done) {
  xllm::proto::XllmAPIService_Stub stub(channel.get());
  stub.ChatCompletions(controller, &request, nullptr, done);
}

void InstanceClient::generate(const std::shared_ptr<brpc::Channel>& channel,
                              brpc::Controller* controller,
                              const xllm::proto::CompletionRequest& request,
                              google::protobuf::Closure* done) {
  xllm::proto::XllmAPIService_Stub stub(channel.get());
  stub.Completions(controller, &request, nullptr, done);
}
}  // namespace xllm_service
