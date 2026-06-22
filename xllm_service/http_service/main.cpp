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

#include <brpc/server.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "common/global_gflags.h"
#include "common/options.h"
#include "http_service/service.h"

int main(int argc, char** argv) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting xllm http service, port: " << FLAGS_port;

  xllm_service::Options service_options;
  xllm_service::XllmHttpServiceImpl service_impl(service_options, nullptr);

  // register http methods here
  brpc::Server server;
  if (server.AddService(&service_impl,
                        brpc::SERVER_DOESNT_OWN_SERVICE,
                        "/hello => Hello,"
                        "/v1/completions => Completions,") != 0) {
    LOG(ERROR) << "Fail to add brpc http service";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  if (server.Start(FLAGS_port, &options) != 0) {
    LOG(ERROR) << "Failed to start brpc http server on port " << FLAGS_port;
    return false;
  }

  LOG(INFO) << "Xllm http server started on port " << FLAGS_port
            << ", idle_timeout_sec: " << FLAGS_idle_timeout_s
            << ", num_threads: " << FLAGS_num_threads
            << ", max_concurrency: " << FLAGS_max_concurrency;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();

  return 0;
}
