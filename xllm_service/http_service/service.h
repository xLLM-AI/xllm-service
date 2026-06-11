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

#include <brpc/channel.h>

#include <iostream>
#include <mutex>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "completion.pb.h"
#include "request/request.h"
#include "request_tracer.h"
#include "xllm_http_service.pb.h"

namespace xllm_service {

class Scheduler;
class InstanceMgr;
class ClosureGuard;

class XllmHttpServiceImpl : public proto::XllmHttpService {
 public:
  XllmHttpServiceImpl(const Options& options, Scheduler* scheduler);
  ~XllmHttpServiceImpl();

  void Hello(::google::protobuf::RpcController* controller,
             const proto::HttpHelloRequest* request,
             proto::HttpHelloResponse* response,
             ::google::protobuf::Closure* done) override;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void AnthropicMessages(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::HttpRequest* request,
              proto::HttpResponse* response,
              ::google::protobuf::Closure* done) override;

  void Metrics(::google::protobuf::RpcController* controller,
               const proto::HttpRequest* request,
               proto::HttpResponse* response,
               ::google::protobuf::Closure* done) override;

  // Internal heartbeat from non-brpc backends (vLLM sidecar): JSON body of
  // proto::HeartbeatRequest carrying LoadMetrics/LatencyMetrics.
  void Heartbeat(::google::protobuf::RpcController* controller,
                 const proto::HttpRequest* request,
                 proto::HttpResponse* response,
                 ::google::protobuf::Closure* done) override;

 private:
  template <typename T>
  std::shared_ptr<Request> generate_request(T* req_pb,
                                            const std::string& method);

  template <typename T>
  void handle(std::shared_ptr<T> call_data,
              const std::string& req_attachment,
              std::shared_ptr<Request> request,
              const std::string& method);

  template <typename T>
  void handle(std::shared_ptr<T> call_data, std::shared_ptr<Request> request);

  void get_serving_models(::google::protobuf::RpcController* controller,
                          const proto::HttpRequest* request,
                          proto::HttpResponse* response,
                          ::google::protobuf::Closure* done);

 private:
  Options options_;

  // not own
  Scheduler* scheduler_;

  bool initialized_ = false;

  std::unique_ptr<RequestTracer> request_tracer_;

  std::unique_ptr<ThreadPool> thread_pool_;
};

}  // namespace xllm_service
