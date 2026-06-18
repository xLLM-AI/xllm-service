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

#pragma once

#include <brpc/channel.h>

#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
class Scheduler;

class InstanceMgr final {
 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service,
                       Scheduler* scheduler);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  bool get_next_instance_pair(Routing* routing);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  bool bind_request_instance_incarnations(
      const std::shared_ptr<Request>& request);
  bool record_instance_heartbeat(const std::string& instance_name,
                                 const std::string& incarnation_id);
  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  // update the recent token latency metrics for the corresponding instance
  void update_latency_metrics(const std::string& instance_name,
                              const proto::LatencyMetrics& latency_metrics);

  // update request metrics under different actions
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // select instances based on the SLO
  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  void set_as_master();

  // Returns true if at least one valid instance group is available:
  // - a single DEFAULT instance, or
  // - a PREFILL + DECODE pair, or
  // - two MIX instances with complementary current_type (one PREFILL, one
  // DECODE)
  bool has_available_instances() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  // Caller must hold cluster_mutex_.
  bool can_route_prefill_without_decode_locked(
      const std::string& prefill_name) const;

  void init();

  // brpc::Channel::Init only; must NOT be called while holding cluster_mutex_.
  bool init_brpc_channel(const std::string& target_uri,
                         const InstanceMetaInfo& info,
                         std::shared_ptr<brpc::Channel>* out_channel);
  bool probe_instance_health(const std::string& instance_name);
  void reconcile_instance_states();
  void refresh_instance_registration(const std::string& name,
                                     const InstanceMetaInfo& info);
  void mark_instance_suspect(const std::string& name,
                             const std::string& incarnation_id);
  void clear_suspect_instance(const std::string& name,
                              const std::string& incarnation_id = "");
  // use etcd as ServiceDiscovery
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  TimePredictor& get_time_predictor(const std::string& instance_name);

  void flip_prefill_to_decode(std::string& instance_name);
  void flip_decode_to_prefill(std::string& instance_name);

  // Register a new instance with all necessary resources and connections
  bool register_instance(const std::string& name, InstanceMetaInfo& info);
  // Remove an instance and clean up its resources and connections
  void deregister_instance(const std::string& name,
                           const std::string& expected_incarnation_id = "");
  // Initialize internal resources for an instance (predictors, metrics)
  void add_instance_resources(const std::string& name,
                              const InstanceMetaInfo& info);
  // Release internal resources for an instance
  void remove_instance_resources(const std::string& name);
  // Build LinkInstance RPC list; caller must hold cluster_mutex_.
  bool gather_link_operations(
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  // Run LinkInstance calls without holding cluster_mutex_.
  bool run_link_operations(
      const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops);
  // Build UnlinkInstance RPC list; caller must hold cluster_mutex_.
  void gather_unlink_operations(
      const std::string& name,
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  // Add instance to prefill or decode index according to its type
  void add_instance_to_index(const std::string& name, InstanceMetaInfo& info);
  // Remove instance from prefill or decode index
  void remove_instance_from_index(const std::string& name,
                                  const InstanceMetaInfo& info);
  bool call_link_instance(const std::string& target_rpc_addr,
                          const InstanceMetaInfo& peer_info);
  bool call_unlink_instance(const std::string& target_rpc_addr,
                            const InstanceMetaInfo& peer_info);

  // Locking (scheme B): only two mutexes participate in ordering.
  // L1 cluster_mutex_: instances_, indices, cached_channels_.
  // L2 metrics_mutex_: load_metrics_, request_metrics_, latency_metrics_,
  // time_predictors_, updated_metrics_, removed_instance_.
  // Order when both needed: always lock L1 before L2 (use std::scoped_lock).
  // get_time_predictor() requires metrics_mutex_ held by caller.
  // remove_instance_resources() requires cluster_mutex_ held by caller.

  Options options_;

  bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  std::shared_ptr<EtcdClient> etcd_client_;

  // L1 — cluster topology & channels
  mutable std::shared_mutex cluster_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  struct SuspectInstanceInfo {
    std::string incarnation_id;
    uint64_t enter_ts_ms = 0;
  };
  std::unordered_map<std::string, SuspectInstanceInfo> suspect_instances_;
  std::vector<std::string> prefill_index_;
  std::vector<std::string> decode_index_;
  uint64_t next_prefill_index_ = 0;
  uint64_t next_decode_index_ = 0;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;

  // L2 — metrics & predictors (single lock to avoid order ambiguity)
  std::shared_mutex metrics_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;

  // not own
  // NOTE: need to refactor with scheduler in future
  Scheduler* scheduler_;

  ThreadPool threadpool_;
  std::unique_ptr<std::thread> state_reconcile_thread_;
};

}  // namespace xllm_service
