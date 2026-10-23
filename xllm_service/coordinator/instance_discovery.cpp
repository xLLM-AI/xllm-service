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

#include "coordinator/instance_discovery.h"

#include <array>
#include <future>
#include <utility>

#include "cluster/instance_state.h"

namespace {
constexpr std::array<const char*, 4> kInstancePrefixes{"XLLM:DEFAULT:",
                                                       "XLLM:PREFILL:",
                                                       "XLLM:DECODE:",
                                                       "XLLM:MIX:"};
const std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";
}  // namespace
namespace xllm_service {
InstanceDiscovery::InstanceDiscovery(std::shared_ptr<EtcdClient> etcd,
                                     std::shared_ptr<InstanceState> instances,
                                     InstanceCallback instance_callback,
                                     LoadCallback load_callback)
    : etcd_client_(std::move(etcd)),
      instances_(std::move(instances)),
      instance_callback_(std::move(instance_callback)),
      load_callback_(std::move(load_callback)) {}

InstanceDiscovery::~InstanceDiscovery() { stop(); }

void InstanceDiscovery::start(bool master) {
  stopped_ = false;
  if (master) {
    master_ = true;
  }
  events_ = std::make_unique<ThreadPool>();
  // Serialize initial state before watch callbacks on the same event queue.
  auto watches_ready = std::make_shared<std::promise<void>>();
  auto ready = watches_ready->get_future().share();
  events_->schedule([this, ready] {
    ready.wait();
    for (const auto* prefix : kInstancePrefixes) {
      std::unordered_map<std::string, InstanceMetaInfo> instances;
      etcd_client_->get_prefix(prefix, &instances);
      for (auto& [name, info] : instances) {
        if (stopped_) {
          return;
        }
        instance_callback_(InstanceEvent{name, std::move(info), false});
      }
    }
    std::unordered_map<std::string, LoadMetrics> metrics;
    etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &metrics);
    for (auto& [name, load] : metrics) {
      if (stopped_) {
        return;
      }
      load_callback_(LoadEvent{name, load, false});
    }
  });
  for (const auto* prefix : kInstancePrefixes) {
    etcd_client_->add_watch(
        prefix,
        [this](const etcd::Response& response, const uint64_t& prefix_len) {
          handle_instances(response, prefix_len);
        });
  }
  if (!master_) {
    etcd_client_->add_watch(
        ETCD_LOADMETRICS_PREFIX,
        [this](const etcd::Response& response, const uint64_t& prefix_len) {
          handle_load(response, prefix_len);
        });
  }
  watches_ready->set_value();
}

void InstanceDiscovery::stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  for (const auto* prefix : kInstancePrefixes) {
    etcd_client_->remove_watch(prefix);
  }
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
  events_.reset();
}

void InstanceDiscovery::set_as_master() { master_ = true; }

void InstanceDiscovery::handle_instances(const etcd::Response& response,
                                         uint64_t prefix_len) {
  if (stopped_) {
    return;
  }
  for (const auto& event : response.events()) {
    InstanceEvent update;
    update.name = get_event_key_suffix(event, prefix_len);
    if (update.name.empty()) {
      continue;
    }
    update.deleted = event.event_type() == etcd::Event::EventType::DELETE_;
    if (!update.deleted && event.event_type() != etcd::Event::EventType::PUT) {
      continue;
    }
    const std::string value = get_event_value(event);
    if ((!update.deleted && value.empty()) ||
        (!value.empty() && !update.info.parse_from_json(value))) {
      LOG(ERROR) << "Invalid instance metadata for " << update.name;
      continue;
    }
    events_->schedule([this, update = std::move(update)]() mutable {
      if (!stopped_) {
        instance_callback_(std::move(update));
      }
    });
  }
}

void InstanceDiscovery::handle_load(const etcd::Response& response,
                                    uint64_t prefix_len) {
  if (stopped_ || master_) {
    return;
  }
  for (const auto& event : response.events()) {
    LoadEvent update;
    update.name = get_event_key_suffix(event, prefix_len);
    update.deleted = event.event_type() == etcd::Event::EventType::DELETE_;
    if (update.name.empty()) {
      continue;
    }
    if (!update.deleted &&
        !update.metrics.parse_from_json(get_event_value(event))) {
      LOG(ERROR) << "Invalid load metadata for " << update.name;
      continue;
    }
    events_->schedule([this, update = std::move(update)]() mutable {
      if (!stopped_ && !master_) {
        load_callback_(std::move(update));
      }
    });
  }
}

bool InstanceDiscovery::upload_load_metrics() {
  const LoadUpdates updates = instances_->take_load_updates();
  const bool changed =
      etcd_client_->set(ETCD_LOADMETRICS_PREFIX, updates.changed);
  const bool removed =
      etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, updates.removed);
  return changed && removed;
}

}  // namespace xllm_service
