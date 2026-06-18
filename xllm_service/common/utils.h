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

#include <optional>
#include <string>

namespace xllm_service {
namespace utils {

bool enable_debug_log();
bool is_port_available(int port);
bool get_bool_env(const std::string& key, bool defaultValue);
std::optional<std::string> get_optional_string_env(const std::string& key);
std::string get_local_ip();
std::string normalize_etcd_namespace(const std::string& etcd_namespace);
std::string build_etcd_key_with_namespace(
    const std::string& normalized_namespace_prefix,
    const std::string& logical_key);

}  // namespace utils
}  // namespace xllm_service
