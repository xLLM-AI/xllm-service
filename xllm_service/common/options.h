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

#pragma once

#include <cstdint>
#include <string>

#include "common/macros.h"

namespace xllm_service {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  // http server options
  PROPERTY(std::string, server_host);

  PROPERTY(int32_t, http_port) = 9998;

  PROPERTY(int32_t, http_idle_timeout_s) = -1;

  PROPERTY(int32_t, http_num_threads) = 32;

  PROPERTY(int32_t, http_max_concurrency) = 0;

  // rpc server options
  PROPERTY(int32_t, rpc_port) = 9999;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_num_threads) = 32;

  PROPERTY(int32_t, rpc_max_concurrency) = 0;

  PROPERTY(int32_t, num_threads) = 32;

  PROPERTY(int32_t, max_concurrency) = 32;

  PROPERTY(int32_t, timeout_ms) = -1;

  PROPERTY(int32_t, connect_timeout_ms) = -1;

  // instance manager options
  PROPERTY(std::string, etcd_addr);

  PROPERTY(std::string, etcd_namespace);

  PROPERTY(int32_t, detect_disconnected_instance_interval) = 15;

  PROPERTY(int32_t, instance_delete_probe_timeout_ms) = 1000;

  PROPERTY(int32_t, instance_delete_probe_attempts) = 2;

  PROPERTY(int32_t, lease_lost_heartbeat_timeout_ms) = 3000;

  // scheduler options
  PROPERTY(std::string, load_balance_policy);

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  PROPERTY(std::string, service_name);

  // tokenizer options
  PROPERTY(std::string, tokenizer_path);

  // trace options
  PROPERTY(bool, enable_request_trace) = false;

  // parser options
  PROPERTY(std::string, tool_call_parser);

  PROPERTY(std::string, reasoning_parser);

  // backend options (vLLM/xLLM dual-backend support)
  PROPERTY(std::string, default_backend_type) = "xllm";

  PROPERTY(int32_t, vllm_http_timeout_ms) = 60000;

  PROPERTY(std::string, internal_api_token);
};

}  // namespace xllm_service
