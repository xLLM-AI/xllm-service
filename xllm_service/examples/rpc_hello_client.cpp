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

#include <brpc/channel.h>
#include <butil/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <string>

#include "xllm_rpc_service.pb.h"

DEFINE_string(server_address, "localhost:9999", "Grpc server address.");
DEFINE_string(protocol,
              "baidu_std",
              "Protocol type. Defined in src/brpc/options.proto");
DEFINE_string(connection_type,
              "",
              "Connection type. Available values: single, pooled, short");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_int32(timeout_ms, 100, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");
DEFINE_int32(interval_ms, 1000, "Milliseconds between consecutive requests");

namespace xllm_service {
namespace test {

struct ChannelOptions {
  std::string protocol = "baidu_std";
  std::string connection_type = "";
  std::string load_balancer = "";
  int timeout_ms = 100;
  int max_retry = 3;
  int interval_ms = 1000;
};

class HelloClient final {
 public:
  HelloClient(const std::string& addr, ChannelOptions options) {
    brpc::ChannelOptions chan_options;
    chan_options.protocol = options.protocol;
    chan_options.connection_type = options.connection_type;
    chan_options.timeout_ms = options.timeout_ms /*milliseconds*/;
    chan_options.max_retry = options.max_retry;
    if (master_channel_.Init(
            addr.c_str(), options.load_balancer.c_str(), &chan_options) != 0) {
      LOG(ERROR) << "Fail to initialize brpc channel to server " << addr;
      return;
    }
    master_stub_ =
        std::make_unique<proto::XllmRpcService_Stub>(&master_channel_);
  }

  void hello() {
    // Create a message to send to the server
    brpc::Controller cntl;
    proto::Empty request;
    proto::Status response;
    master_stub_->Hello(&cntl, &request, &response, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Send to server faild, err msg:" << cntl.ErrorText();
      return;
    }

    std::cout << "Get server response: " << response.ok() << "\n";
  }

 private:
  brpc::Channel master_channel_;
  std::unique_ptr<proto::XllmRpcService_Stub> master_stub_;
};

}  // namespace test
}  // namespace xllm_service

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Define the server address and port
  std::string server_address(FLAGS_server_address);

  xllm_service::test::ChannelOptions opt;
  opt.protocol = FLAGS_protocol;
  opt.connection_type = FLAGS_connection_type;
  opt.load_balancer = FLAGS_load_balancer;
  opt.timeout_ms = FLAGS_timeout_ms;
  opt.max_retry = FLAGS_max_retry;
  opt.interval_ms = FLAGS_interval_ms;

  // Create a chat client
  xllm_service::test::HelloClient client(server_address, opt);

  client.hello();

  return 0;
}
