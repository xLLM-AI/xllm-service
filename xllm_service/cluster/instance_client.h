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

#include <brpc/channel.h>
#include <brpc/controller.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "chat.pb.h"
#include "common/types.h"
#include "completion.pb.h"
#include "config/service_config.h"

namespace xllm_service {
class InstanceClient final {
 public:
  explicit InstanceClient(const ServiceConfig& options);
  bool connect_instance(const InstanceMetaInfo& info);
  void disconnect_instance(const std::string& name,
                           const std::string& incarnation_id);
  std::shared_ptr<brpc::Channel> get_channel(
      const std::string& name,
      const std::string& incarnation_id) const;
  bool probe_instance_health(const std::string& name);
  bool call_link_instance(const std::string& target,
                          const InstanceMetaInfo& peer);
  bool call_unlink_instance(const std::string& target,
                            const InstanceMetaInfo& peer);
  void generate(const std::shared_ptr<brpc::Channel>& channel,
                brpc::Controller* controller,
                const xllm::proto::ChatRequest& request,
                google::protobuf::Closure* done);
  void generate(const std::shared_ptr<brpc::Channel>& channel,
                brpc::Controller* controller,
                const xllm::proto::CompletionRequest& request,
                google::protobuf::Closure* done);

 private:
  bool init_brpc_channel(const std::string& name,
                         const InstanceMetaInfo& info,
                         std::shared_ptr<brpc::Channel>* channel);
  struct Connection {
    std::string incarnation_id;
    std::shared_ptr<brpc::Channel> channel;
  };
  ServiceConfig options_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Connection> channels_;
};
}  // namespace xllm_service
