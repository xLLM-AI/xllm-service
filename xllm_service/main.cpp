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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <csignal>
#include <thread>

#include "common/utils.h"
#include "config/config_loader.h"
#include "service_app.h"
namespace {
volatile std::sig_atomic_t signal_received = 0;
void shutdown_handler(int signal) { signal_received = signal; }
}  // namespace
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  const xllm_service::ServiceConfig config =
      xllm_service::load_service_config();
  if (!xllm_service::utils::is_port_available(config.http_port()) ||
      !xllm_service::utils::is_port_available(config.rpc_port())) {
    LOG(ERROR) << "A configured service port is already in use";
    return 1;
  }
  std::signal(SIGINT, shutdown_handler);
  std::signal(SIGTERM, shutdown_handler);
  xllm_service::ServiceApp app(config);
  if (!app.start()) {
    return 1;
  }
  while (signal_received == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  LOG(INFO) << "Stopping service after signal " << signal_received;
  app.stop();
  return 0;
}
