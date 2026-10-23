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

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "scheduler/request_metrics.h"
#include "scheduler/scheduler.h"

namespace {
using xllm_service::InstanceRuntimeState;
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

constexpr char kHealthPath[] = "/health";
constexpr int64_t kDeleteProbeRetryBackoffMs = 100;
constexpr long double kDecodeTokenBalanceSlack = 1.05L;

uint64_t current_time_ms() {
  return static_cast<uint64_t>(
      absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch()));
}

bool is_instance_schedulable(const xllm_service::InstanceMetaInfo& info) {
  // LEASE_LOST instances can still be reused while heartbeats continue.
  return info.runtime_state != InstanceRuntimeState::SUSPECT;
}

bool select_next_schedulable_instance(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    uint64_t* next_index,
    std::string* instance_name) {
  if (index.empty()) {
    return false;
  }

  const uint64_t start_index = *next_index % index.size();
  for (uint64_t offset = 0; offset < index.size(); ++offset) {
    const uint64_t index_pos = (start_index + offset) % index.size();
    auto it = instances.find(index[index_pos]);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    *instance_name = index[index_pos];
    *next_index = index_pos + 1;
    return true;
  }
  return false;
}

size_t count_schedulable_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index) {
  size_t count = 0;
  for (const auto& name : index) {
    auto it = instances.find(name);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    ++count;
  }
  return count;
}

InstanceType get_cleanup_type(const xllm_service::InstanceMetaInfo& info) {
  if (info.type == InstanceType::DEFAULT) {
    return InstanceType::PREFILL;
  }
  if (info.type == InstanceType::MIX) {
    return info.current_type;
  }
  return info.type;
}
}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         Scheduler* scheduler)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client),
      scheduler_(scheduler) {
  if (!options_.prefill_time_profile_file().empty()) {
    std::ifstream input(options_.prefill_time_profile_file());
    CHECK(input.is_open()) << "Cannot open prefill time profile";
    const auto profiles = nlohmann::json::parse(input, nullptr, false);
    CHECK(profiles.is_object()) << "Expected instance-to-TTFT-profile mapping";
    for (const auto& [name, data] : profiles.items()) {
      auto points = data.get<std::vector<std::pair<int32_t, double>>>();
      TimePredictor predictor(points, {});
      CHECK(!points.empty() &&
            predictor.predict_prefill_time(points.front().first, 0).has_value())
          << "Invalid TTFT profile for " << name;
      prefill_time_profiles_.emplace(name, std::move(points));
    }
  }
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();

  state_reconcile_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::reconcile_instance_states, this);
}

std::vector<InstanceMetaInfo> InstanceMgr::get_instance_snapshot() const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  std::vector<InstanceMetaInfo> result;
  result.reserve(instances_.size());
  for (const auto& entry : instances_) {
    result.emplace_back(entry.second);
  }
  return result;
}

void InstanceMgr::init() {
  std::unordered_map<std::string, InstanceMetaInfo> loaded_instances;
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->get_prefix(it.second, &loaded_instances);
  }
  LOG(INFO) << "Load instance info from etcd:" << loaded_instances.size();

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    prefill_index_.reserve(loaded_instances.size());
    decode_index_.reserve(loaded_instances.size());
  }

  for (auto& pair : loaded_instances) {
    if (!register_instance(pair.first, pair.second)) {
      LOG(ERROR) << "Fail to register instance: " << pair.first;
    }
  }

  std::unordered_map<std::string, LoadMetrics> loaded_metrics;
  etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &loaded_metrics);
  {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    load_metrics_ = std::move(loaded_metrics);
  }

  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    for (int i = 0; i < prefill_index_.size(); i++) {
      LOG(INFO) << i << " : " << prefill_index_[i];
    }
  }
}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::can_route_prefill_without_decode_locked(
    const std::string& prefill_name) const {
  auto selected_it = instances_.find(prefill_name);
  if (selected_it == instances_.end() ||
      selected_it->second.type != InstanceType::DEFAULT) {
    LOG(ERROR) << "No decode instance and selected prefill is not default, "
               << "instance: " << prefill_name;
    return false;
  }
  return true;
}

bool InstanceMgr::get_next_instance_pair(Routing* routing) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }

  routing->decode_name.clear();
  if (suspect_instances_.empty()) {
    // Fast path for the common case: no suspect instances, keep plain RR.
    next_prefill_index_ = next_prefill_index_ % prefill_index_.size();
    routing->prefill_name = prefill_index_[next_prefill_index_];
    next_prefill_index_++;

    if (decode_index_.empty()) {
      return can_route_prefill_without_decode_locked(routing->prefill_name);
    }

    next_decode_index_ = next_decode_index_ % decode_index_.size();
    routing->decode_name = decode_index_[next_decode_index_];
    next_decode_index_++;
    return true;
  }

  if (!select_next_schedulable_instance(instances_,
                                        prefill_index_,
                                        &next_prefill_index_,
                                        &routing->prefill_name)) {
    LOG(ERROR) << "No schedulable prefill or default instance found!";
    return false;
  }

  if (decode_index_.empty()) {
    return can_route_prefill_without_decode_locked(routing->prefill_name);
  }

  select_next_schedulable_instance(
      instances_, decode_index_, &next_decode_index_, &routing->decode_name);
  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE &&
        is_instance_schedulable(inst.second)) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if ((inst.second.type == InstanceType::PREFILL ||
         inst.second.type == InstanceType::DEFAULT) &&
        is_instance_schedulable(inst.second)) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

RoutingCandidates InstanceMgr::get_routing_candidates() const {
  std::shared_lock<std::shared_mutex> inst_lock(cluster_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(metrics_mutex_);
  const auto collect = [this](const std::vector<std::string>& index) {
    std::vector<RoutingCandidate> result;
    result.reserve(index.size());
    for (const auto& name : index) {
      const auto instance = instances_.find(name);
      if (instance == instances_.end() ||
          !is_instance_schedulable(instance->second)) {
        continue;
      }
      // Prefer this master's latest heartbeat over the periodically replicated
      // metrics. Instances without a first load report are not ready for CAR.
      const auto updated = updated_metrics_.find(name);
      const auto previous = load_metrics_.find(name);
      if (updated == updated_metrics_.end() &&
          previous == load_metrics_.end()) {
        continue;
      }
      const auto& load = updated != updated_metrics_.end() ? updated->second
                                                           : previous->second;
      if (!std::isfinite(load.gpu_cache_usage_perc) ||
          load.gpu_cache_usage_perc < 0 || load.gpu_cache_usage_perc > 1) {
        continue;
      }
      const auto& info = instance->second;
      result.emplace_back(RoutingCandidate{name,
                                           info.incarnation_id,
                                           info.register_ts_ms,
                                           info.dp_size,
                                           info.type,
                                           load});
    }
    return result;
  };
  return {collect(prefill_index_), collect(decode_index_)};
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::unordered_map<std::string, LoadMetrics> upload_snapshot;
  std::unordered_set<std::string> remove_snapshot;
  {
    std::unique_lock<std::shared_mutex> lk(metrics_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, iter.second);
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
    upload_snapshot = updated_metrics_;
    remove_snapshot = removed_instance_;
    updated_metrics_.clear();
    removed_instance_.clear();
  }
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, upload_snapshot);
  status = status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, remove_snapshot);
  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  // Bind the selected routing to a concrete incarnation before dispatch.
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();

  if (!request->routing.prefill_name.empty()) {
    auto prefill_it = instances_.find(request->routing.prefill_name);
    if (prefill_it == instances_.end()) {
      LOG(ERROR) << "Prefill instance is not registered when binding request: "
                 << request->routing.prefill_name;
      return false;
    }
    if (!is_instance_schedulable(prefill_it->second)) {
      LOG(ERROR) << "Prefill instance is not schedulable when binding request: "
                 << request->routing.prefill_name << ", state: "
                 << runtime_state_name(prefill_it->second.runtime_state);
      return false;
    }
    request->prefill_incarnation_id = prefill_it->second.incarnation_id;
  }

  if (!request->routing.decode_name.empty()) {
    auto decode_it = instances_.find(request->routing.decode_name);
    if (decode_it == instances_.end()) {
      LOG(ERROR) << "Decode instance is not registered when binding request: "
                 << request->routing.decode_name;
      return false;
    }
    if (!is_instance_schedulable(decode_it->second)) {
      LOG(ERROR) << "Decode instance is not schedulable when binding request: "
                 << request->routing.decode_name << ", state: "
                 << runtime_state_name(decode_it->second.runtime_state);
      return false;
    }
    request->decode_incarnation_id = decode_it->second.incarnation_id;
  }

  return true;
}

bool InstanceMgr::request_instances_match(const Request& request) const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const auto matches = [this](const std::string& name,
                              const std::string& incarnation_id) {
    if (name.empty()) {
      return true;
    }
    auto it = instances_.find(name);
    return it != instances_.end() && is_instance_schedulable(it->second) &&
           it->second.incarnation_id == incarnation_id;
  };
  return matches(request.routing.prefill_name,
                 request.prefill_incarnation_id) &&
         matches(request.routing.decode_name, request.decode_incarnation_id);
}

bool InstanceMgr::record_instance_heartbeat(const std::string& instance_name,
                                            const std::string& incarnation_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  auto it = instances_.find(instance_name);
  if (it == instances_.end()) {
    LOG(WARNING) << "Ignore heartbeat from unknown instance: " << instance_name;
    return false;
  }

  if (it->second.incarnation_id != incarnation_id) {
    LOG(WARNING) << "Ignore stale heartbeat from instance: " << instance_name
                 << ", current incarnation_id: " << it->second.incarnation_id
                 << ", heartbeat incarnation_id: " << incarnation_id;
    return false;
  }

  it->second.latest_timestamp = current_time_ms();
  if (it->second.runtime_state == InstanceRuntimeState::SUSPECT) {
    // A recovered suspect instance first goes back to LEASE_LOST and must
    // keep heartbeating before becoming fully active again via registration.
    clear_suspect_instance(instance_name, incarnation_id);
    it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
    LOG(WARNING) << "Heartbeat recovered for suspect instance, move to "
                 << "lease lost state: " << instance_name
                 << ", incarnation_id: " << incarnation_id;
  }
  return true;
}

bool InstanceMgr::init_brpc_channel(
    const std::string& instance_name,
    const InstanceMetaInfo& info,
    std::shared_ptr<brpc::Channel>* out_channel) {
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  if (info.backend_type == "vllm") {
    // Pass the bare host:port to Init: brpc treats any "scheme://" prefix as a
    // naming service, and "http" is not one, so a "http://" target fails with
    // "Unknown naming service". options.protocol selects HTTP instead.
    options.protocol = "http";
    options.timeout_ms = options_.vllm_http_timeout_ms();
    options.max_retry = 0;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  } else {
    options.timeout_ms = options_.timeout_ms();
    options.max_retry = 3;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  }
  std::string load_balancer = "";
  if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize channel for " << instance_name
               << " (backend=" << info.backend_type << ")";
    return false;
  }
  *out_channel = std::move(channel);
  return true;
}

bool InstanceMgr::probe_instance_health(const std::string& instance_name) {
  const int attempts = std::max(1, options_.instance_delete_probe_attempts());
  const int timeout_ms =
      std::max(1, options_.instance_delete_probe_timeout_ms());
  const std::string url = "http://" + instance_name + kHealthPath;

  for (int attempt = 1; attempt <= attempts; ++attempt) {
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = timeout_ms;
    options.connect_timeout_ms = timeout_ms;
    options.max_retry = 0;
    if (channel.Init(url.c_str(), "", &options) != 0) {
      LOG(WARNING) << "Failed to initialize health probe channel, instance: "
                   << instance_name << ", attempt: " << attempt << "/"
                   << attempts;
    } else {
      brpc::Controller cntl;
      cntl.http_request().uri() = url;
      cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
      channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
      if (!cntl.Failed() && cntl.http_response().status_code() == 200) {
        return true;
      }
      LOG(WARNING) << "Health probe failed, instance: " << instance_name
                   << ", attempt: " << attempt << "/" << attempts << ", error: "
                   << (cntl.Failed() ? cntl.ErrorText()
                                     : std::to_string(
                                           cntl.http_response().status_code()));
    }

    if (attempt < attempts) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDeleteProbeRetryBackoffMs));
    }
  }

  return false;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    for (const auto& event : response.events()) {
      const std::string instance_name = get_event_key_suffix(event, prefix_len);
      if (instance_name.empty()) {
        continue;
      }

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = get_event_value(event);
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "Parse instance json failed: " << json_str;
          continue;
        }

        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          lock.unlock();
          if (!register_instance(instance_name, metainfo)) {
            LOG(ERROR) << "Fail to register instance: " << instance_name;
          }
          continue;
        }

        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          const auto previous_state = existing_it->second.runtime_state;
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          if (previous_state != InstanceRuntimeState::ACTIVE) {
            LOG(INFO) << "Instance registration restored, back to active: "
                      << instance_name
                      << ", incarnation_id: " << metainfo.incarnation_id
                      << ", previous_state: "
                      << runtime_state_name(previous_state);
          }
          continue;
        }

        const std::string old_incarnation_id =
            existing_it->second.incarnation_id;
        LOG(WARNING) << "Detected instance replacement, instance_name: "
                     << instance_name
                     << ", old incarnation_id: " << old_incarnation_id
                     << ", new incarnation_id: " << metainfo.incarnation_id;
        lock.unlock();
        deregister_instance(instance_name, old_incarnation_id);
        if (!register_instance(instance_name, metainfo)) {
          LOG(ERROR) << "Fail to register replacement instance: "
                     << instance_name;
        }
        continue;
      }

      if (event.event_type() != etcd::Event::EventType::DELETE_) {
        continue;
      }

      InstanceMetaInfo deleted_info;
      std::string deleted_incarnation_id;
      const auto deleted_value = get_event_value(event);
      if (!deleted_value.empty() &&
          deleted_info.parse_from_json(deleted_value)) {
        deleted_incarnation_id = deleted_info.incarnation_id;
      }
      std::string tracked_incarnation_id;
      {
        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          continue;
        }

        if (!deleted_incarnation_id.empty() &&
            existing_it->second.incarnation_id != deleted_incarnation_id) {
          LOG(INFO) << "Ignore stale delete for replaced instance: "
                    << instance_name
                    << ", deleted incarnation_id: " << deleted_incarnation_id
                    << ", current incarnation_id: "
                    << existing_it->second.incarnation_id;
          continue;
        }
        tracked_incarnation_id = existing_it->second.incarnation_id;
      }

      // Keep delete handling event-driven: use this event, one health probe,
      // and later heartbeats or PUT events to drive recovery.
      const bool probe_success = probe_instance_health(instance_name);

      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      auto existing_it = instances_.find(instance_name);
      if (existing_it == instances_.end() ||
          existing_it->second.incarnation_id != tracked_incarnation_id) {
        continue;
      }

      if (probe_success) {
        // Keep the instance in service temporarily and wait for heartbeats.
        clear_suspect_instance(instance_name, tracked_incarnation_id);
        existing_it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
        existing_it->second.latest_timestamp = current_time_ms();
        LOG(WARNING) << "Instance lease deleted, probe succeeded, enter "
                     << "lease lost state: " << instance_name
                     << ", incarnation_id: " << tracked_incarnation_id;
        continue;
      }

      mark_instance_suspect(instance_name, tracked_incarnation_id);
      LOG(WARNING) << "Instance lease deleted, probe failed, enter suspect "
                   << "state: " << instance_name
                   << ", incarnation_id: " << tracked_incarnation_id;
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMgr::reconcile_instance_states() {
  const auto suspect_interval_ms =
      std::max<int64_t>(1, options_.detect_disconnected_instance_interval()) *
      1000;
  const auto heartbeat_timeout_ms =
      std::max<int64_t>(1, options_.lease_lost_heartbeat_timeout_ms());
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::pair<std::string, std::string>> to_deregister;

    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      if (exited_) {
        return;
      }

      const uint64_t now_ms = current_time_ms();
      for (auto& [instance_name, info] : instances_) {
        // LEASE_LOST is a grace period after etcd delete but before hard
        // eviction.
        if (info.runtime_state != InstanceRuntimeState::LEASE_LOST) {
          continue;
        }
        if (now_ms - info.latest_timestamp < heartbeat_timeout_ms) {
          continue;
        }
        mark_instance_suspect(instance_name, info.incarnation_id);
        LOG(WARNING)
            << "Lease lost instance heartbeat timed out, enter suspect "
            << "state: " << instance_name
            << ", incarnation_id: " << info.incarnation_id;
      }

      for (auto it = suspect_instances_.begin();
           it != suspect_instances_.end();) {
        const std::string instance_name = it->first;
        const std::string incarnation_id = it->second.incarnation_id;
        const uint64_t enter_ts_ms = it->second.enter_ts_ms;
        ++it;

        if (now_ms - enter_ts_ms < suspect_interval_ms) {
          continue;
        }

        auto inst_it = instances_.find(instance_name);
        if (inst_it == instances_.end() ||
            inst_it->second.incarnation_id != incarnation_id) {
          suspect_instances_.erase(instance_name);
          continue;
        }

        LOG(WARNING) << "Suspect window expired, deregister instance: "
                     << instance_name << ", incarnation_id: " << incarnation_id;
        to_deregister.emplace_back(instance_name, incarnation_id);
      }
    }

    for (const auto& p : to_deregister) {
      deregister_instance(p.first, p.second);
    }
  }
}

void InstanceMgr::refresh_instance_registration(const std::string& name,
                                                const InstanceMetaInfo& info) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    return;
  }

  // Preserve local scheduling/index state across etcd refreshes.
  const auto instance_index = it->second.instance_index;
  const auto current_type = it->second.current_type;

  it->second = info;
  it->second.instance_index = instance_index;
  it->second.current_type = current_type;
  it->second.latest_timestamp = current_time_ms();
  it->second.runtime_state = InstanceRuntimeState::ACTIVE;
}

void InstanceMgr::mark_instance_suspect(const std::string& name,
                                        const std::string& incarnation_id) {
  SuspectInstanceInfo info;
  info.incarnation_id = incarnation_id;
  info.enter_ts_ms = current_time_ms();
  suspect_instances_[name] = std::move(info);
  auto it = instances_.find(name);
  if (it != instances_.end() && it->second.incarnation_id == incarnation_id) {
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
  }
}

void InstanceMgr::clear_suspect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  auto it = suspect_instances_.find(name);
  if (it == suspect_instances_.end()) {
    return;
  }
  if (!incarnation_id.empty() && it->second.incarnation_id != incarnation_id) {
    return;
  }
  suspect_instances_.erase(it);
}

void InstanceMgr::update_request_metrics(
    const std::shared_ptr<Request>& request,
    RequestAction action,
    int64_t generated_tokens) {
  const bool ds_load_balance =
      options_.load_balance_policy() == "ds_load_balance";
  if (options_.load_balance_policy() != "SLO_AWARE" && !ds_load_balance) {
    return;
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);

  if (ds_load_balance && (!request->ds_load_balance_reserved ||
                          action == RequestAction::SCHEDULE)) {
    return;
  }

  // An instance may disappear after KV handoff. Release the surviving peer's
  // reservation even when the failed peer's metrics have already been removed.
  auto prefill_it = request_metrics_.find(request->routing.prefill_name);
  auto prefill_instance = instances_.find(request->routing.prefill_name);
  if (prefill_it != request_metrics_.end() &&
      prefill_instance != instances_.end() &&
      prefill_instance->second.incarnation_id ==
          request->prefill_incarnation_id) {
    apply_prefill_request_metrics(*request, action, prefill_it->second);
  }
  auto decode_it = request_metrics_.find(request->routing.decode_name);
  auto decode_instance = instances_.find(request->routing.decode_name);
  if (decode_it != request_metrics_.end() &&
      decode_instance != instances_.end() &&
      decode_instance->second.incarnation_id ==
          request->decode_incarnation_id) {
    apply_decode_request_metrics(
        *request, action, decode_it->second, generated_tokens);
    if (!ds_load_balance && decode_it->second.decode_request_num == 0) {
      flip_decode_to_prefill(request->routing.decode_name);
    }
  }
  if (ds_load_balance && (action == RequestAction::CANCEL ||
                          action == RequestAction::FINISH_DECODE)) {
    request->ds_load_balance_reserved = false;
  }
}

bool InstanceMgr::select_instance_pair_on_ds_load_balance(
    const std::shared_ptr<Request>& request,
    const std::unordered_map<std::string, PrefillCacheHint>& cache_hints) {
  if (request->token_ids.empty() ||
      request->token_ids.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  std::scoped_lock lock(cluster_mutex_, metrics_mutex_);
  if (request->ds_load_balance_reserved) {
    return false;
  }
  const int32_t length = static_cast<int32_t>(request->token_ids.size());
  std::vector<std::string> decode_candidates;
  decode_candidates.reserve(decode_index_.size());
  long double total_decode_tokens = length;
  for (size_t offset = 0; offset < decode_index_.size(); ++offset) {
    const std::string& name =
        decode_index_[(next_decode_index_ + offset) % decode_index_.size()];
    const auto instance = instances_.find(name);
    if (instance == instances_.end() ||
        !is_instance_schedulable(instance->second) ||
        instance->second.backend_type != "xllm") {
      continue;
    }
    decode_candidates.emplace_back(name);
    total_decode_tokens += request_metrics_.at(name).decode_token_num;
  }
  // Adapt DualPath's within-group rule to one incoming request. Below the
  // post-admission token threshold, balance sequences; otherwise balance KV
  // length. Engine-side admission still enforces actual HBM capacity.
  const long double token_threshold = decode_candidates.empty()
                                          ? 0
                                          : kDecodeTokenBalanceSlack *
                                                total_decode_tokens /
                                                decode_candidates.size();
  std::string decode_name;
  bool selected_high_tokens = true;
  int64_t min_decode_requests = std::numeric_limits<int64_t>::max();
  int64_t min_decode_tokens = std::numeric_limits<int64_t>::max();
  for (const std::string& name : decode_candidates) {
    const auto& metrics = request_metrics_.at(name);
    const bool high_tokens =
        static_cast<long double>(metrics.decode_token_num) + length >
        token_threshold;
    const bool better_same_category =
        high_tokens
            ? std::tie(metrics.decode_token_num, metrics.decode_request_num) <
                  std::tie(min_decode_tokens, min_decode_requests)
            : std::tie(metrics.decode_request_num, metrics.decode_token_num) <
                  std::tie(min_decode_requests, min_decode_tokens);
    if (decode_name.empty() || (!high_tokens && selected_high_tokens) ||
        (high_tokens == selected_high_tokens && better_same_category)) {
      decode_name = name;
      selected_high_tokens = high_tokens;
      min_decode_requests = metrics.decode_request_num;
      min_decode_tokens = metrics.decode_token_num;
    }
  }

  std::string prefill_name;
  int32_t selected_cached_tokens = 0;
  int64_t selected_cost = 0;
  int64_t min_queued_time = std::numeric_limits<int64_t>::max();
  for (size_t offset = 0; offset < prefill_index_.size(); ++offset) {
    const std::string& name =
        prefill_index_[(next_prefill_index_ + offset) % prefill_index_.size()];
    const auto instance = instances_.find(name);
    if (instance == instances_.end() ||
        !is_instance_schedulable(instance->second) ||
        instance->second.backend_type != "xllm" ||
        (decode_name.empty() &&
         instance->second.type != InstanceType::DEFAULT)) {
      continue;
    }
    int32_t cached_tokens = 0;
    const auto hint = cache_hints.find(name);
    if (hint != cache_hints.end() &&
        hint->second.incarnation_id == instance->second.incarnation_id) {
      cached_tokens = std::clamp(hint->second.cached_tokens, 0, length - 1);
    }
    auto cost =
        get_time_predictor(name).predict_prefill_time(length, cached_tokens);
    // Reject an invalid fit rather than treating it as a free engine.
    if (!cost.has_value() || *cost > 1e12) {
      continue;
    }
    const int64_t reservation = static_cast<int64_t>(std::ceil(*cost));
    const int64_t queued = request_metrics_.at(name).estimated_prefill_time;
    if (queued < 0 ||
        queued > std::numeric_limits<int64_t>::max() - reservation) {
      continue;
    }
    if (queued < min_queued_time) {
      min_queued_time = queued;
      prefill_name = name;
      selected_cost = reservation;
      selected_cached_tokens = cached_tokens;
    }
  }
  if (prefill_name.empty()) {
    LOG(ERROR)
        << "ds_load_balance needs a healthy prefill engine with valid TTFT "
           "profiling (at least three distinct lengths) and a decode peer";
    return false;
  }
  request->routing.prefill_name = prefill_name;
  request->routing.decode_name = decode_name;
  request->prefill_incarnation_id = instances_.at(prefill_name).incarnation_id;
  request->decode_incarnation_id =
      decode_name.empty() ? "" : instances_.at(decode_name).incarnation_id;
  request->estimated_ttft = selected_cost;
  request->store_cached_tokens = selected_cached_tokens;
  request->ds_load_balance_reserved = true;
  auto& prefill_metrics = request_metrics_.at(prefill_name);
  prefill_metrics.estimated_prefill_time += selected_cost;
  apply_prefill_request_metrics(
      *request, RequestAction::SCHEDULE, prefill_metrics);
  if (!decode_name.empty()) {
    apply_decode_request_metrics(
        *request, RequestAction::SCHEDULE, request_metrics_.at(decode_name));
  }
  next_prefill_index_ =
      std::find(prefill_index_.begin(), prefill_index_.end(), prefill_name) -
      prefill_index_.begin() + 1;
  if (!decode_name.empty()) {
    next_decode_index_ =
        std::find(decode_index_.begin(), decode_index_.end(), decode_name) -
        decode_index_.begin() + 1;
  }
  VLOG(1) << "ds_load_balance prefill=" << prefill_name
          << " decode=" << decode_name
          << " store_hit_tokens=" << selected_cached_tokens
          << " compute_ms=" << selected_cost
          << " queued_compute_ms=" << min_queued_time
          << " decode_requests=" << min_decode_requests
          << " decode_tokens=" << min_decode_tokens;
  return true;
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);
  const bool has_unschedulable_instances = !suspect_instances_.empty();

  std::string min_prefill_instance;
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  int64_t total_prefill_time = 0;
  size_t schedulable_prefill_count = 0;
  for (const auto& prefill_instance : prefill_index_) {
    if (has_unschedulable_instances) {
      auto it = instances_.find(prefill_instance);
      if (it == instances_.end() || !is_instance_schedulable(it->second)) {
        continue;
      }
    }

    int64_t prefill_time =
        request_metrics_[prefill_instance].estimated_prefill_time;
    total_prefill_time += prefill_time;
    if (prefill_time < min_prefill_time) {
      min_prefill_instance = prefill_instance;
      min_prefill_time = prefill_time;
    }
    ++schedulable_prefill_count;
  }

  if (schedulable_prefill_count == 0) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }
  int64_t avg_prefill_time = total_prefill_time / schedulable_prefill_count;

  std::string min_decode_instance;
  int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
  std::string target_decode_instance;
  size_t schedulable_decode_count = 0;
  for (const auto& decode_instance : decode_index_) {
    if (has_unschedulable_instances) {
      auto it = instances_.find(decode_instance);
      if (it == instances_.end() || !is_instance_schedulable(it->second)) {
        continue;
      }
    }

    int64_t token_num = request_metrics_[decode_instance].decode_token_num;
    int64_t request_num = request_metrics_[decode_instance].decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    int64_t estimated_tpot = time_predictor.predict_tpot(
        token_num + request->token_ids.size(), request_num + 1);
    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
      target_decode_instance = decode_instance;
    }

    if (estimated_tpot < min_estimated_tpot) {
      min_decode_instance = decode_instance;
      min_estimated_tpot = estimated_tpot;
    }
    ++schedulable_decode_count;
  }

  if (schedulable_decode_count == 0) {
    LOG(ERROR) << "No decode instance found!";
    return false;
  }

  if (!target_decode_instance.empty()) {
    request->routing.decode_name = target_decode_instance;
  } else {
    request->routing.decode_name = min_decode_instance;
  }

  // select prefill instance
  float tpot_threshold =
      (schedulable_decode_count - 1.0f) / schedulable_decode_count;
  // When the prefill instances are already overloaded and there are other
  // instances with lower loads in the decode group, we will dispatch the
  // prefill requests to those instances to alleviate the pressure on the
  // prefill instances.
  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      request_metrics_[min_decode_instance].estimated_prefill_time <
          min_prefill_time) {
    request->routing.prefill_name = min_decode_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_decode_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_decode_instance].estimated_prefill_time +=
        request->estimated_ttft;
  } else {
    request->routing.prefill_name = min_prefill_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_prefill_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_prefill_instance].estimated_prefill_time +=
        request->estimated_ttft;
  }

  // If there are no decode instances that meet the requirements, switch a
  // prefill instance to decode if the number of instances allows. Since the
  // current disaggregated PD mode does not support prefill and decode using the
  // same instance, we only switch the instance here, without dispatching the
  // decode request to this instance.
  float ttft_threshold =
      (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
  if (target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       schedulable_decode_count < schedulable_prefill_count)) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (count_schedulable_instances(instances_, prefill_index_) <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to decode_index_
  instances_[instance_name].current_type = InstanceType::DECODE;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (count_schedulable_instances(instances_, decode_index_) <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to prefill_index
  instances_[instance_name].current_type = InstanceType::PREFILL;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

bool InstanceMgr::call_link_instance(const std::string& target_rpc_addr,
                                     const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::call_unlink_instance(const std::string& target_rpc_addr,
                                       const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::register_instance(const std::string& name,
                                    InstanceMetaInfo& info) {
  info.runtime_state = InstanceRuntimeState::ACTIVE;
  info.latest_timestamp = current_time_ms();
  info.name = name;

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(ERROR) << "Instance is already registered, instance_name: " << name;
      return false;
    }
  }

  std::shared_ptr<brpc::Channel> channel;
  if (!init_brpc_channel(name, info, &channel)) {
    LOG(ERROR) << "create channel fail: " << name;
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(WARNING) << "Instance registered concurrently during channel init: "
                   << name;
      return false;
    }
    cached_channels_[name] = std::move(channel);
  }

  add_instance_resources(name, info);

  std::vector<std::pair<std::string, InstanceMetaInfo>> link_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (!gather_link_operations(info, &link_ops)) {
      remove_instance_resources(name);
      return false;
    }
  }

  if (!run_link_operations(link_ops)) {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    remove_instance_resources(name);
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    add_instance_to_index(name, info);
    instances_.insert(std::make_pair(name, info));
  }
  return true;
}

void InstanceMgr::deregister_instance(
    const std::string& name,
    const std::string& expected_incarnation_id) {
  InstanceMetaInfo info;
  std::vector<std::pair<std::string, InstanceMetaInfo>> unlink_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      LOG(ERROR) << "Instance is not registered, instance_name: " << name;
      return;
    }

    if (!expected_incarnation_id.empty() &&
        it->second.incarnation_id != expected_incarnation_id) {
      LOG(INFO) << "Skip deregistering stale incarnation, instance_name: "
                << name
                << ", current incarnation_id: " << it->second.incarnation_id
                << ", expected incarnation_id: " << expected_incarnation_id;
      return;
    }

    info = it->second;
    clear_suspect_instance(name, info.incarnation_id);
    gather_unlink_operations(name, info, &unlink_ops);
  }

  for (const auto& op : unlink_ops) {
    call_unlink_instance(op.first, op.second);
  }

  // Admission revalidates the bound incarnation under this same barrier.
  // Keep it held until removal is complete, including queued cancellations.
  std::lock_guard<std::mutex> admission_lock(scheduler_->admission_mutex_);
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end() ||
        it->second.incarnation_id != info.incarnation_id) {
      return;
    }
    remove_instance_from_index(name, it->second);
  }

  scheduler_->clear_requests_on_failed_instance(
      name, info.incarnation_id, get_cleanup_type(info));

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_resources(name);
    instances_.erase(it);
  }
  LOG(INFO) << "delete instance: " << name;
}

void InstanceMgr::add_instance_resources(const std::string& name,
                                         const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  const auto profile = prefill_time_profiles_.find(name);
  const auto& ttft_data = profile == prefill_time_profiles_.end()
                              ? info.ttft_profiling_data
                              : profile->second;
  time_predictors_.insert_or_assign(
      name, TimePredictor(ttft_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMgr::remove_instance_resources(const std::string& name) {
  // Caller must hold cluster_mutex_ (cached_channels_ is L1).
  cached_channels_.erase(name);
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  time_predictors_.erase(name);
  request_metrics_.erase(name);
  latency_metrics_.erase(name);
  updated_metrics_.erase(name);
  removed_instance_.insert(name);
  load_metrics_.erase(name);
}

bool InstanceMgr::gather_link_operations(
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.backend_type == "vllm") {
    return true;
  }
  switch (info.type) {
    case InstanceType::DEFAULT:
      break;
    case InstanceType::PREFILL: {
      for (auto& d_name : decode_index_) {
        out_ops->emplace_back(instances_[d_name].rpc_address, info);
      }
      break;
    }
    case InstanceType::DECODE: {
      for (auto& p_name : prefill_index_) {
        out_ops->emplace_back(info.rpc_address, instances_[p_name]);
      }
      break;
    }
    case InstanceType::MIX: {
      for (const auto& [peer_name, peer_info] : instances_) {
        if (peer_name == info.name) {
          continue;
        }
        out_ops->emplace_back(info.rpc_address, peer_info);
      }
      break;
    }
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(info.type);
      return false;
  }
  return true;
}

bool InstanceMgr::run_link_operations(
    const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (!call_link_instance(ops[i].first, ops[i].second)) {
      LOG(ERROR) << "Fail to link instance during registration, op index " << i;
      for (size_t j = 0; j < i; ++j) {
        call_unlink_instance(ops[j].first, ops[j].second);
      }
      return false;
    }
  }
  return true;
}

void InstanceMgr::gather_unlink_operations(
    const std::string& name,
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.type == InstanceType::PREFILL) {
    for (auto& d_name : decode_index_) {
      out_ops->emplace_back(instances_[d_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::DECODE) {
    for (auto& p_name : prefill_index_) {
      out_ops->emplace_back(instances_[p_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::MIX) {
    for (const auto& [peer_name, peer_info] : instances_) {
      if (peer_name == name) {
        continue;
      }
      out_ops->emplace_back(peer_info.rpc_address, info);
    }
  }
}

void InstanceMgr::add_instance_to_index(const std::string& name,
                                        InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new default instance, instance name : " << name;
      break;
    case InstanceType::PREFILL:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new prefill instance, instance name : " << name;
      break;
    case InstanceType::DECODE:
      info.instance_index = decode_index_.size();
      decode_index_.emplace_back(name);
      LOG(INFO) << "Register a new decode instance, instance name : " << name;
      break;
    case InstanceType::MIX:
      if (decode_index_.size() > 0) {
        info.instance_index = prefill_index_.size();
        info.current_type = InstanceType::PREFILL;
        prefill_index_.emplace_back(name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << name;
      } else {
        info.instance_index = decode_index_.size();
        info.current_type = InstanceType::DECODE;
        decode_index_.emplace_back(name);
        LOG(INFO) << "Register a new decode instance, instance name : " << name;
      }
      break;
    default:
      break;
  }
}

void InstanceMgr::remove_instance_from_index(const std::string& name,
                                             const InstanceMetaInfo& info) {
  uint64_t index = info.instance_index;
  if (index == -1) return;

  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    if (index >= vec.size()) return;
    std::swap(vec[index], vec.back());
    instances_[vec[index]].instance_index = index;
    vec.pop_back();
  };

  switch (info.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      remove_from_vec(prefill_index_);
      break;
    case InstanceType::DECODE:
      remove_from_vec(decode_index_);
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        remove_from_vec(prefill_index_);
      } else {
        remove_from_vec(decode_index_);
      }
      break;
    default:
      break;
  }
}

bool InstanceMgr::has_available_instances() const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  bool has_default = false;
  bool has_prefill = false;
  bool has_decode = false;
  bool has_mix_as_prefill = false;
  bool has_mix_as_decode = false;

  for (const auto& [name, info] : instances_) {
    if (!is_instance_schedulable(info)) continue;

    switch (info.type) {
      case InstanceType::DEFAULT:
        has_default = true;
        break;
      case InstanceType::PREFILL:
        has_prefill = true;
        break;
      case InstanceType::DECODE:
        has_decode = true;
        break;
      case InstanceType::MIX:
        if (info.current_type == InstanceType::PREFILL) {
          has_mix_as_prefill = true;
        } else if (info.current_type == InstanceType::DECODE) {
          has_mix_as_decode = true;
        }
        break;
      default:
        break;
    }

    // Early exit: any satisfied condition is enough
    if (has_default || (has_prefill && has_decode) ||
        (has_mix_as_prefill && has_mix_as_decode)) {
      return true;
    }
  }

  return has_default || (has_prefill && has_decode) ||
         (has_mix_as_prefill && has_mix_as_decode);
}

}  // namespace xllm_service
