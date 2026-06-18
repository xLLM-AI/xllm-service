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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "rpc_service/client.h"

DEFINE_string(server_address, "localhost:9999", "Grpc server address.");
DEFINE_string(client_name, "127.0.0.1@9999", "client name.");
DEFINE_string(protocol,
              "baidu_std",
              "Protocol type. Defined in src/brpc/options.proto");
DEFINE_string(connection_type,
              "",
              "Connection type. Available values: single, pooled, short");
DEFINE_string(server, "0.0.0.0:8000", "IP Address of server");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_int32(timeout_ms, 100, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");
DEFINE_int32(interval_ms, 1000, "Milliseconds between consecutive requests");

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Define the server address and port
  std::string server_address(FLAGS_server_address);

  xllm_service::ChannelOptions options;

  // Create a client instance
  xllm_service::XllmRpcClient client(
      FLAGS_client_name, server_address, options);

  // Register the instance
  auto ret = client.register_instance();
  if (ret != xllm_service::ErrorCode::OK) {
    LOG(ERROR) << "Register instance failed.";
    return -1;
  }

  // Keep the client running
  while (true) {
    sleep(1);
  }

  return 0;
}
