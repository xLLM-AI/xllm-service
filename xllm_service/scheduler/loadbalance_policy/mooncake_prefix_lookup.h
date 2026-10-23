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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace xllm_service {

// Metadata only. A failed lookup is an explicit zero-hit routing estimate;
// the engine still decides which blocks it can actually load.
// Implementations must support concurrent batch_exists calls.
class StoreKeyLookup {
 public:
  virtual ~StoreKeyLookup() = default;
  virtual bool batch_exists(const std::vector<std::string>& keys,
                            std::vector<bool>* exists) = 0;
};

struct StorePrefixLayout {
  std::vector<std::string> instances;
  // Exact binary xllm-kv-v3 prefixes, including every TP shard/cache component.
  std::vector<std::string> key_prefixes;
  int32_t block_size = 128;
  uint64_t hash_seed = 1024;
  int32_t hash_lookahead = 0;
};

class MooncakePrefixLookup final {
 public:
  MooncakePrefixLookup(std::vector<StorePrefixLayout> layouts,
                       int32_t max_keys,
                       std::unique_ptr<StoreKeyLookup> client);

  std::unordered_map<std::string, PrefillCacheHint> match(
      const std::vector<int32_t>& tokens,
      const std::vector<InstanceMetaInfo>& instances);

 private:
  std::vector<StorePrefixLayout> layouts_;
  int32_t max_keys_;
  std::unique_ptr<StoreKeyLookup> client_;
};

std::unique_ptr<MooncakePrefixLookup> create_mooncake_prefix_lookup(
    const std::string& config_file);

// Implemented in an isolated C++20 translation unit when the SDK is enabled.
std::unique_ptr<StoreKeyLookup> create_mooncake_key_lookup(
    std::string master_address,
    std::string tenant_id,
    int32_t timeout_ms);

}  // namespace xllm_service
