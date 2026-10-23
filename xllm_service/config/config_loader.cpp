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

#include "config/config_loader.h"

#include "common/global_gflags.h"
#include "common/utils.h"
namespace xllm_service {
ServiceConfig load_service_config() {
  ServiceConfig options;
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
      .car_host_hit_weight(FLAGS_car_host_hit_weight)
      .car_prefill_lambda(FLAGS_car_prefill_lambda)
      .service_name(utils::get_local_ip() + ":" +
                    std::to_string(FLAGS_rpc_server_port))
      .detect_disconnected_instance_interval(
          FLAGS_detect_disconnected_instance_interval)
      .instance_delete_probe_timeout_ms(FLAGS_instance_delete_probe_timeout_ms)
      .instance_delete_probe_attempts(FLAGS_instance_delete_probe_attempts)
      .lease_lost_heartbeat_timeout_ms(FLAGS_lease_lost_heartbeat_timeout_ms)
      .enable_request_trace(FLAGS_enable_request_trace)
      .block_size(FLAGS_block_size)
      .kv_events_config_file(FLAGS_kv_events_config_file)
      .tokenizer_path(FLAGS_tokenizer_path)
      .default_backend_type(FLAGS_default_backend_type)
      .vllm_http_timeout_ms(FLAGS_vllm_http_timeout_ms)
      .internal_api_token(FLAGS_internal_api_token)
      .target_ttft(FLAGS_target_ttft)
      .target_tpot(FLAGS_target_tpot)
      .readiness_check_interval_s(FLAGS_readiness_check_interval_s);
  return options;
}
}  // namespace xllm_service
