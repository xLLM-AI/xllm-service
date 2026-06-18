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

#include <glog/logging.h>

#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/hash_util.h"
#include "nlohmann/json.hpp"

namespace xllm_service {

constexpr const char* ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";
constexpr const char* ETCD_XSERVICE_KEY_PREFIX = "XLLM:SERVICE:";
constexpr const char* ETCD_MASTER_SERVICE_NAME = "MASTER";

struct CacheLocations;
using XXH3KeyCacheMap = std::unordered_map<XXH3Key,
                                           CacheLocations,
                                           FixedStringKeyHash,
                                           FixedStringKeyEqual>;

struct Routing {
  std::string prefill_name;
  std::string decode_name;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["prefill_name"] = prefill_name;
    json_val["decode_name"] = decode_name;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }
};

enum class ErrorCode : int32_t {
  OK = 0,
  INTERNAL_ERROR = 1,
  INSTANCE_EXISTED = 2,
  INSTANCE_NOT_EXISTED = 3,
};

class ConvertErrorCode {
 public:
  static int32_t to_int(ErrorCode code) noexcept {
    return static_cast<int32_t>(code);
  }

  static ErrorCode from_int(int32_t code) noexcept {
    return static_cast<ErrorCode>(code);
  }
};

enum class InstanceType : int8_t {
  DEFAULT = 0,
  // prefill instance
  PREFILL = 1,
  // decode instance
  DECODE = 2,
  // mix instance
  MIX = 3,
};

enum class InstanceRuntimeState : int8_t {
  ACTIVE = 0,
  LEASE_LOST = 1,
  SUSPECT = 2,
};

inline const char* runtime_state_name(InstanceRuntimeState state) {
  switch (state) {
    case InstanceRuntimeState::ACTIVE:
      return "ACTIVE";
    case InstanceRuntimeState::LEASE_LOST:
      return "LEASE_LOST";
    case InstanceRuntimeState::SUSPECT:
      return "SUSPECT";
    default:
      return "UNKNOWN";
  }
}

struct LoadMetrics {
  LoadMetrics() : waiting_requests_num(0), gpu_cache_usage_perc(0) {};
  LoadMetrics(const uint64_t& waiting_reqs_num, const float& usage)
      : waiting_requests_num(waiting_reqs_num), gpu_cache_usage_perc(usage) {};

  uint64_t waiting_requests_num;
  float gpu_cache_usage_perc;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["waiting_requests_num"] = waiting_requests_num;
    json_val["gpu_cache_usage_perc"] = gpu_cache_usage_perc;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);

      waiting_requests_num =
          json_value.at("waiting_requests_num").get<uint64_t>();
      gpu_cache_usage_perc = json_value.at("gpu_cache_usage_perc").get<float>();

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to loadmetrics error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return false; }
};

// Record the latency monitoring metrics of the instance over the recent period
struct LatencyMetrics {
  LatencyMetrics() : recent_max_ttft(0), recent_max_tbt(0) {}

  LatencyMetrics(const int64_t& recent_max_ttft, const int64_t& recent_max_tbt)
      : recent_max_ttft(recent_max_ttft), recent_max_tbt(recent_max_tbt) {}

  // The unit is milliseconds.
  int64_t recent_max_ttft;
  int64_t recent_max_tbt;
};

enum class RequestAction : int32_t {
  SCHEDULE = 0,
  FINISH_PREFILL = 1,
  GENERATE = 2,
  FINISH_DECODE = 3,
  CANCEL = 4,
};

// Record the request metrics of the instance
struct RequestMetrics {
  RequestMetrics()
      : prefill_request_num(0),
        prefill_token_num(0),
        decode_request_num(0),
        decode_token_num(0),
        estimated_prefill_time(0) {}

  int64_t prefill_request_num;
  int64_t prefill_token_num;

  int64_t decode_request_num;
  int64_t decode_token_num;

  // Estimated execution time for all prefill requests on the instance.
  // The unit is milliseconds.
  int64_t estimated_prefill_time;
};

struct InstanceMetaInfo {
 public:
  InstanceMetaInfo() { set_init_timestamp(); }
  InstanceMetaInfo(const std::string& inst_name, const std::string rpc_addr)
      : name(inst_name), rpc_address(rpc_addr) {
    set_init_timestamp();
  }
  InstanceMetaInfo(const std::string& inst_name,
                   const std::string rpc_addr,
                   const InstanceType& inst_type)
      : name(inst_name), rpc_address(rpc_addr), type(inst_type) {
    set_init_timestamp();
  }

  std::string name = "";
  std::string rpc_address = "";
  std::string incarnation_id = "";
  uint64_t register_ts_ms = 0;
  InstanceType type = InstanceType::DEFAULT;
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  int32_t dp_size;
  int32_t kv_split_size;
  // transfer listen ports
  std::vector<uint16_t> ports;
  // ttft profiling data
  std::vector<std::pair<int32_t, double>> ttft_profiling_data;
  // tpot profiling data
  std::vector<std::tuple<int32_t, int32_t, double>> tpot_profiling_data;

  // latest heatbeat timestamp
  uint64_t latest_timestamp = 0;

  // Runtime-only instance state, not persisted to etcd.
  InstanceRuntimeState runtime_state = InstanceRuntimeState::ACTIVE;

  uint64_t instance_index = -1;

  // Used to indicate the exact instance type of a MIX type instance currently,
  // only used when the SLO Aware scheduling policy is enabled.
  InstanceType current_type = InstanceType::PREFILL;

  std::string backend_type = "xllm";

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["name"] = name;
    json_val["rpc_address"] = rpc_address;
    json_val["incarnation_id"] = incarnation_id;
    json_val["register_ts_ms"] = register_ts_ms;
    json_val["type"] = int8_t(type);
    json_val["addrs"] = addrs;
    json_val["cluster_ids"] = cluster_ids;
    json_val["dp_size"] = dp_size;
    json_val["kv_split_size"] = kv_split_size;
    json_val["ports"] = ports;
    json_val["ttft_profiling_data"] = ttft_profiling_data;
    json_val["tpot_profiling_data"] = tpot_profiling_data;
    json_val["backend_type"] = backend_type;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      name = json_value.at("name").get<std::string>();
      rpc_address = json_value.at("rpc_address").get<std::string>();
      incarnation_id = json_value.value("incarnation_id", std::string());
      register_ts_ms = json_value.value("register_ts_ms", uint64_t(0));
      type = static_cast<InstanceType>(json_value.at("type").get<int8_t>());
      cluster_ids.clear();
      addrs.clear();
      ports.clear();
      ttft_profiling_data.clear();
      tpot_profiling_data.clear();

      cluster_ids = json_value.value("cluster_ids", std::vector<uint64_t>{});
      addrs = json_value.value("addrs", std::vector<std::string>{});
      dp_size = json_value.value("dp_size", 0);
      ports = json_value.value("ports", std::vector<uint16_t>{});

      kv_split_size = json_value.value("kv_split_size", 1);

      if (json_value.contains("ttft_profiling_data")) {
        for (const auto& item : json_value.at("ttft_profiling_data")) {
          if (item.is_array() && item.size() == 2) {
            ttft_profiling_data.emplace_back(item[0], item[1]);
          }
        }
      }

      if (json_value.contains("tpot_profiling_data")) {
        for (const auto& item : json_value.at("tpot_profiling_data")) {
          if (item.is_array() && item.size() == 3) {
            tpot_profiling_data.emplace_back(item[0], item[1], item[2]);
          }
        }
      }

      backend_type = json_value.value("backend_type", std::string("xllm"));

      runtime_state = InstanceRuntimeState::ACTIVE;
      set_init_timestamp();
    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to instancemetainfo error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return rpc_address == ""; }

 private:
  void set_init_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    latest_timestamp = timestamp_ms;
  }
};

struct CacheLocations {
  std::unordered_set<std::string> hbm_instance_set;
  std::unordered_set<std::string> dram_instance_set;
  std::unordered_set<std::string> ssd_instance_set;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["hbm_instance_set"] = hbm_instance_set;
    json_val["dram_instance_set"] = dram_instance_set;
    json_val["ssd_instance_set"] = ssd_instance_set;
    return json_val;
  }

  std::string debug_string() { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      for (const auto& item :
           json_value.at("hbm_instance_set").get<std::vector<std::string>>()) {
        hbm_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("dram_instance_set").get<std::vector<std::string>>()) {
        dram_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("ssd_instance_set").get<std::vector<std::string>>()) {
        ssd_instance_set.insert(item);
      }

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to cachelocation error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const {
    return hbm_instance_set.empty() && dram_instance_set.empty() &&
           ssd_instance_set.empty();
  }
};

/**
 * @brief Records the prefix cache match lengths for different instances on
 * current request
 *
 * This struct stores and manages prefix cache matching information across
 * multiple instances, supporting different storage types (HBM, DRAM, SSD) for
 * match length recording, and tracks information about the best matching
 * instance.
 */
struct OverlapScores {
  // Set of matched instance names
  std::unordered_set<std::string> instances;
  // HBM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> hbm_instance_score;
  // DRAM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> dram_instance_score;
  // SSD storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> ssd_instance_score;
  uint32_t max_block_num = 0;
  uint32_t max_matched_block_num = 0;
  std::string max_matched_instance_name = "";

  std::string debug_string() {
    nlohmann::json json_val;
    json_val["instances"] = instances;
    json_val["hbm_instance_score"] = hbm_instance_score;
    json_val["dram_instance_score"] = dram_instance_score;
    json_val["ssd_instance_score"] = ssd_instance_score;
    json_val["max_block_num"] = max_block_num;
    json_val["max_matched_block_num"] = max_matched_block_num;
    json_val["max_matched_instance_name"] = max_matched_instance_name;
    return json_val.dump(2);
  }
};

struct LoadBalanceInfos {
  OverlapScores overlap_scores;
  std::unordered_map<std::string, LoadMetrics> prefill_load_metrics;
  std::unordered_map<std::string, LoadMetrics> decode_load_metrics;
  uint64_t prefill_max_waiting_requests_num = 0;
  uint64_t decode_max_waiting_requests_num = 0;

  std::string debug_string() {
    nlohmann::json json_val;

    json_val["overlap_scores"] =
        nlohmann::json::parse(overlap_scores.debug_string());

    nlohmann::json prefill_json;
    for (auto& [key, metrics] : prefill_load_metrics) {
      prefill_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["prefill_load_metrics"] = prefill_json;

    nlohmann::json decode_json;
    for (auto& [key, metrics] : decode_load_metrics) {
      decode_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["decode_load_metrics"] = decode_json;

    json_val["prefill_max_waiting_requests_num"] =
        prefill_max_waiting_requests_num;
    json_val["decode_max_waiting_requests_num"] =
        decode_max_waiting_requests_num;

    return json_val.dump(2);
  }
};

// Function call related types
struct JsonFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters;

  JsonFunction() = default;
  JsonFunction(const std::string& func_name,
               const std::string& desc,
               const nlohmann::json& params)
      : name(func_name), description(desc), parameters(params) {}
};

struct JsonTool {
  std::string type;  // "function"
  JsonFunction function;

  JsonTool() : type("function") {}
  JsonTool(const std::string& tool_type, const JsonFunction& func)
      : type(tool_type), function(func) {}
};

}  // namespace xllm_service
