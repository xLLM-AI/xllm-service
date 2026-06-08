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

#include "master.h"

#include <chrono>
#include <csignal>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace xllm_service {

Master::Master(const Options& options) : options_(options) {
  scheduler_ = std::make_unique<Scheduler>(options);

  rpc_service_ =
      std::make_unique<xllm_service::XllmRpcService>(options, scheduler_.get());

  http_service_ = std::make_unique<xllm_service::XllmHttpServiceImpl>(
      options, scheduler_.get());
}

Master::~Master() { stop(); }

bool Master::start() {
  if (!setup_http_server()) {
    return false;
  }

  // 1. start readiness thread to manage http server lifecycle
  readiness_thread_ = std::make_unique<std::thread>(
      [this]() { manage_http_server_lifecycle(); });

  // 2. start rpc server
  rpc_server_thread_ =
      std::make_unique<std::thread>([this]() { start_rpc_server(); });

  return true;
}

void Master::stop() {
  stopped_.store(true);

  if (readiness_thread_ && readiness_thread_->joinable()) {
    readiness_thread_->join();
  }

  if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
    rpc_server_thread_->join();
  }
}

bool Master::setup_http_server() {
  if (http_server_.AddService(http_service_.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              // for testing
                              "/hello => Hello,"
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/messages => AnthropicMessages,"
                              "/v1/embeddings => Embeddings,"
                              "/v1/models => Models,"
                              "/metrics => Metrics,") != 0) {
    LOG(FATAL) << "Fail to add http service";
    return false;
  }

  http_options_.idle_timeout_sec = options_.http_idle_timeout_s();
  http_options_.num_threads = options_.http_num_threads();
  http_options_.max_concurrency = options_.http_max_concurrency();

  if (!options_.server_host().empty()) {
    http_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.http_port());
    if (butil::str2endpoint(http_server_address_.c_str(), &http_endpoint_) <
        0) {
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << http_server_address_;
      return false;
    }
  } else {
    http_endpoint_ = butil::EndPoint(butil::IP_ANY, options_.http_port());
  }

  return true;
}

void Master::manage_http_server_lifecycle() {
  bool http_running = false;
  while (!stopped_.load()) {
    bool has_instances = scheduler_->has_available_instances();

    if (has_instances && !http_running) {
      if (http_server_.Start(http_endpoint_, &http_options_) == 0) {
        LOG(INFO) << "HTTP server started, instances available, endpoint: "
                  << http_endpoint_;
        http_running = true;
      } else {
        LOG(ERROR) << "Failed to start HTTP server on: " << http_endpoint_;
      }
    } else if (!has_instances && http_running) {
      LOG(WARNING) << "No available instances, stopping HTTP server.";
      http_server_.Stop(0);
      http_server_.Join();
      http_running = false;
      LOG(INFO) << "HTTP server stopped, waiting for instances to recover.";
    }

    // Sleep in small increments to be responsive to `stopped_` signal.
    const auto end_time =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(FLAGS_readiness_check_interval_s);
    while (!stopped_.load() && std::chrono::steady_clock::now() < end_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  if (http_running) {
    http_server_.Stop(0);
    http_server_.Join();
  }
}

bool Master::start_rpc_server() {
  if (rpc_server_.AddService(rpc_service_.get(),
                             brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(FATAL) << "Failed to add rpc service.";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = options_.rpc_idle_timeout_s();
  options.num_threads = options_.rpc_num_threads();
  options.max_concurrency = options_.rpc_max_concurrency();

  butil::EndPoint endpoint;
  if (!options_.server_host().empty()) {
    rpc_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.rpc_port());
    if (butil::str2endpoint(rpc_server_address_.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << rpc_server_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, options_.rpc_port());
  }

  if (rpc_server_.Start(endpoint, &options) != 0) {
    LOG(FATAL) << "Failed to start rpc server on: " << endpoint;
    return false;
  }

  LOG(INFO) << "Xllm rpc server started on: " << endpoint;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  rpc_server_.RunUntilAskedToQuit();
  return true;
}

}  // namespace xllm_service

static std::atomic<uint32_t> g_signal_received{0};
void shutdown_handler(int signal) {
  LOG(WARNING) << "Received signal " << signal << ", stopping master...";
  exit(1);
}

int main(int argc, char* argv[]) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting xllm master service.";

  // check port available or not
  if (!xllm_service::utils::is_port_available(FLAGS_http_server_port)) {
    LOG(ERROR)
        << "Http server port " << FLAGS_http_server_port
        << " is already in use. "
        << "Please specify a different port using --http_server_port flag.";
    return -1;
  }
  if (!xllm_service::utils::is_port_available(FLAGS_rpc_server_port)) {
    LOG(ERROR)
        << "Rpc server port " << FLAGS_rpc_server_port << " is already in use. "
        << "Please specify a different port using --rpc_server_port flag.";
    return -1;
  }

  xllm_service::Options options;
  options.server_host(FLAGS_server_host)
      .http_port(FLAGS_http_server_port)
      .http_idle_timeout_s(FLAGS_http_server_idle_timeout_s)
      .http_num_threads(FLAGS_http_server_num_threads)
      .http_max_concurrency(FLAGS_http_server_max_concurrency)
      .rpc_port(FLAGS_rpc_server_port)
      .rpc_idle_timeout_s(FLAGS_rpc_server_idle_timeout_s)
      .rpc_num_threads(FLAGS_rpc_server_num_threads)
      .rpc_max_concurrency(FLAGS_rpc_server_max_concurrency)
      .num_threads(FLAGS_num_threads)
      .max_concurrency(FLAGS_max_concurrency)
      .timeout_ms(FLAGS_timeout_ms)
      .connect_timeout_ms(FLAGS_connect_timeout_ms)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
      .load_balance_policy(FLAGS_load_balance_policy)
      .xxh3_128bits_seed(FLAGS_xxh3_128bits_seed)
      .service_name(xllm_service::utils::get_local_ip() + ":" +
                    std::to_string(FLAGS_rpc_server_port))
      .detect_disconnected_instance_interval(
          FLAGS_detect_disconnected_instance_interval)
      .instance_delete_probe_timeout_ms(FLAGS_instance_delete_probe_timeout_ms)
      .instance_delete_probe_attempts(FLAGS_instance_delete_probe_attempts)
      .lease_lost_heartbeat_timeout_ms(FLAGS_lease_lost_heartbeat_timeout_ms)
      .enable_request_trace(FLAGS_enable_request_trace)
      .block_size(FLAGS_block_size)
      .tokenizer_path(FLAGS_tokenizer_path)
      .tool_call_parser(FLAGS_tool_call_parser)
      .reasoning_parser(FLAGS_reasoning_parser);

  xllm_service::Master master(options);

  if (!master.start()) {
    LOG(ERROR) << "Failed to start master service.";
    return -1;
  }

  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  while (g_signal_received.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // wait here
  master.stop();

  return 0;
}
