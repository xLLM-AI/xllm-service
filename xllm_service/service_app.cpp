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

#include "service_app.h"

#include <chrono>
#include <utility>

#include "common/utils.h"
namespace xllm_service {
ServiceApp::ServiceApp(const ServiceConfig& options) : options_(options) {}

ServiceApp::~ServiceApp() { stop(); }

bool ServiceApp::start() {
  if (started_) {
    return false;
  }
  started_ = true;
  stopped_ = false;
  const std::string username =
      utils::get_optional_string_env("ETCD_USERNAME").value_or("");
  const std::string password =
      utils::get_optional_string_env("ETCD_PASSWORD").value_or("");
  CHECK_EQ(username.empty(), password.empty())
      << "ETCD_USERNAME and ETCD_PASSWORD must be set together";
  if (username.empty()) {
    etcd_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                         options_.etcd_namespace());
  } else {
    etcd_ = std::make_shared<EtcdClient>(
        options_.etcd_addr(), username, password, options_.etcd_namespace());
  }
  instances_ = std::make_shared<InstanceState>();
  routing_state_ = std::make_shared<RoutingState>(options_, instances_);
  instance_client_ = std::make_unique<InstanceClient>(options_);
  requests_ = std::make_unique<RequestManager>(instances_, routing_state_);
  controller_ = std::make_unique<InstanceController>(
      options_,
      instances_,
      *instance_client_,
      [this](const std::string& name, const std::string& incarnation) {
        return requests_->retire_instance(name, incarnation);
      });
  discovery_ = std::make_unique<InstanceDiscovery>(
      etcd_,
      instances_,
      [this](InstanceEvent event) {
        controller_->on_instance_event(std::move(event));
      },
      [this](LoadEvent event) {
        controller_->on_load_event(std::move(event));
      });
  membership_ = std::make_unique<ServiceMembership>(
      options_,
      etcd_,
      [this] { discovery_->set_as_master(); },
      [this] { discovery_->upload_load_metrics(); });
  // Construct model-dependent state before starting event producers.
  if (!options_.kv_events_config_file().empty()) {
    KVDirectoryConfig config;
    std::string error;
    CHECK(load_kv_directory_config(
        options_.kv_events_config_file(), &config, &error))
        << error;
    kv_directory_ = std::make_unique<KVDirectory>(std::move(config));
  }
  router_ =
      std::make_unique<Router>(options_, routing_state_, kv_directory_.get());
  http_service_ = std::make_unique<XllmHttpServiceImpl>(options_,
                                                        router_.get(),
                                                        requests_.get(),
                                                        instances_.get(),
                                                        controller_.get(),
                                                        kv_directory_.get(),
                                                        instance_client_.get());
  rpc_service_ = std::make_unique<XllmRpcService>(
      options_, requests_.get(), instances_.get(), controller_.get());
  if (!setup_http_server() || !start_rpc_server() || !membership_->start()) {
    stop();
    return false;
  }
  controller_->start();
  discovery_->start(membership_->is_master());
  if (kv_directory_) {
    kv_directory_->start([instances = instances_] {
      return make_kv_event_instances(instances->snapshot());
    });
  }
  readiness_thread_ =
      std::make_unique<std::thread>([this] { manage_http_server_lifecycle(); });
  return true;
}

void ServiceApp::stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  if (membership_) {
    membership_->stop();
  }
  if (discovery_) {
    discovery_->stop();
  }
  if (controller_) {
    controller_->stop();
  }
  if (kv_directory_) {
    kv_directory_->stop();
  }
  if (requests_) {
    requests_->stop();
  }
  if (readiness_thread_ && readiness_thread_->joinable()) {
    readiness_thread_->join();
  }
  http_server_.Stop(0);
  if (rpc_running_) {
    rpc_server_.Stop(0);
  }
  http_server_.Join();
  if (rpc_running_) {
    rpc_server_.Join();
    rpc_running_ = false;
  }
}

bool ServiceApp::setup_http_server() {
  if (http_server_.AddService(http_service_.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              // for testing
                              "/hello => Hello,"
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/messages => AnthropicMessages,"
                              "/v1/embeddings => Embeddings,"
                              "/v1/models => Models,"
                              "/metrics => Metrics,"
                              "/v1/internal/heartbeat => Heartbeat,"
                              "/v1/internal/kv-cache/query => KVCacheQuery,") !=
      0) {
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

void ServiceApp::manage_http_server_lifecycle() {
  bool http_running = false;
  while (!stopped_.load()) {
    bool has_instances = instances_->has_available_instances();

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
        std::chrono::seconds(options_.readiness_check_interval_s());
    while (!stopped_.load() && std::chrono::steady_clock::now() < end_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  if (http_running) {
    http_server_.Stop(0);
  }
}

bool ServiceApp::start_rpc_server() {
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

  rpc_running_ = true;
  return true;
}

}  // namespace xllm_service
