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
#include <algorithm>
#include <future>

#include "cluster/instance_controller.h"
#include "router/policies/cache_aware_routing.h"
#include "router/request_manager.h"
#include "router/routing_state.h"

namespace xllm_service {
class ServiceTestContext final {
 public:
  explicit ServiceTestContext(std::string policy = "SLO_AWARE") {
    ServiceConfig options;
    options.load_balance_policy(std::move(policy));
    instances = std::make_shared<InstanceState>();
    routing = std::make_shared<RoutingState>(options, instances);
    client = std::make_unique<InstanceClient>(options);
    requests = std::make_unique<RequestManager>(instances, routing);
    controller = std::make_unique<InstanceController>(
        options,
        instances,
        *client,
        [this](const std::string& name, const std::string& incarnation) {
          return requests->retire_instance(name, incarnation);
        });
  }
  ~ServiceTestContext() {
    if (directory) {
      directory->stop();
    }
  }
  std::shared_ptr<InstanceState> instances;
  std::shared_ptr<RoutingState> routing;
  std::unique_ptr<InstanceClient> client;
  std::unique_ptr<RequestManager> requests;
  std::unique_ptr<InstanceController> controller;
  std::unique_ptr<KVDirectory> directory;
};
class RoutingHarness final {
 public:
  RoutingHarness(ServiceTestContext& context, bool use_directory, double lambda)
      : context_(context),
        policy_(0.5, lambda),
        use_directory_(use_directory) {}
  bool select_instances_pair(const std::shared_ptr<Request>& request) {
    std::vector<KVDirectoryMatch> matches;
    if (use_directory_ && context_.directory) {
      matches = context_.directory->match(request->token_ids);
    }
    return context_.routing->select_route(
        *request, policy_, std::move(matches));
  }

 private:
  ServiceTestContext& context_;
  CacheAwareRouting policy_;
  bool use_directory_;
};
class ServiceTestPeer final {
 public:
  static std::unique_ptr<ServiceTestContext> create() {
    return std::make_unique<ServiceTestContext>();
  }
  static std::unique_ptr<ServiceTestContext> observing(
      KVDirectoryConfig config) {
    auto context = std::make_unique<ServiceTestContext>("CAR");
    context->directory = std::make_unique<KVDirectory>(std::move(config));
    context->directory->start();
    return context;
  }
  static std::unique_ptr<ServiceTestContext> discovering(
      KVDirectoryConfig config) {
    auto context = std::make_unique<ServiceTestContext>("CAR");
    context->directory = std::make_unique<KVDirectory>(std::move(config));
    context->directory->start([instances = context->instances] {
      return make_kv_event_instances(instances->snapshot());
    });
    return context;
  }
  static void add_instance(ServiceTestContext& context,
                           const std::string& name,
                           const std::string& incarnation_id,
                           InstanceType type) {
    InstanceMetaInfo info;
    info.name = name;
    info.incarnation_id = incarnation_id;
    info.type = type;
    info.dp_size = 1;
    info.kv_split_size = 1;
    CHECK(context.instances->add_instance(std::move(info)));
  }
  static void instances(ServiceTestContext& context,
                        std::vector<InstanceMetaInfo> values) {
    const auto previous = context.instances->snapshot();
    for (const auto& view : previous) {
      InstanceMetaInfo removed;
      CHECK(context.instances->remove_instance(
          view.info->name, view.info->incarnation_id, &removed));
    }
    for (auto& value : values) {
      const std::string name = value.name;
      const std::string incarnation = value.incarnation_id;
      CHECK(context.instances->add_instance(std::move(value)));
      const auto old = std::find_if(
          previous.begin(), previous.end(), [&](const InstanceView& view) {
            return view.info->name == name &&
                   view.info->incarnation_id == incarnation;
          });
      if (old != previous.end() && old->load.has_value()) {
        context.instances->apply_load_event(LoadEvent{name, *old->load, false});
      }
    }
  }
  static bool bind(ServiceTestContext& context,
                   const std::shared_ptr<Request>& request) {
    if (!context.routing->bind_request(*request)) {
      return false;
    }
    context.routing->update_request_metrics(request, RequestAction::SCHEDULE);
    return true;
  }
  static bool publish(ServiceTestContext& context,
                      std::shared_ptr<Request> request) {
    return context.requests->publish_request(std::move(request));
  }
  static void remove(ServiceTestContext& context, const std::string& name) {
    context.controller->deregister_instance(name);
  }
  static bool contains(ServiceTestContext& context,
                       const std::string& request_id) {
    std::lock_guard lock(context.requests->request_mutex_);
    return context.requests->requests_.count(request_id) != 0;
  }
  static RequestMetrics metrics(ServiceTestContext& context,
                                const std::string& name) {
    return context.routing->request_metrics(name);
  }
  static void drain(ServiceTestContext& context) { context.requests->drain(); }
  static std::unique_ptr<RoutingHarness> routing(ServiceTestContext& context,
                                                 bool use_directory = true,
                                                 double prefill_lambda = 0.5) {
    return std::make_unique<RoutingHarness>(
        context, use_directory, prefill_lambda);
  }
  static void load(ServiceTestContext& context,
                   const std::string& name,
                   uint64_t waiting,
                   float usage) {
    context.instances->apply_load_event(
        LoadEvent{name, LoadMetrics(waiting, usage), false});
  }
};
}  // namespace xllm_service
