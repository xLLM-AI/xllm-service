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

#pragma once

#include <string.h>
#include <xxhash.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/slice.h"

namespace xllm_service {
constexpr uint32_t XXH3_128BITS_HASH_VALUE_LEN = sizeof(XXH128_hash_t);

struct XXH3Key {
  uint8_t data[XXH3_128BITS_HASH_VALUE_LEN];

  XXH3Key() {}
  XXH3Key(const uint8_t* const input_data) {
    memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }
  XXH3Key(const char* const input_data) {
    memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }

  std::string to_string() const {
    return std::string(reinterpret_cast<const char*>(data),
                       XXH3_128BITS_HASH_VALUE_LEN);
  }

  bool operator==(const XXH3Key& other) {
    return strncmp(reinterpret_cast<const char*>(data),
                   reinterpret_cast<const char*>(other.data),
                   XXH3_128BITS_HASH_VALUE_LEN);
  }
};

struct FixedStringKeyHash {
  size_t operator()(const XXH3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const XXH3Key& left, const XXH3Key& right) const {
    return memcmp(reinterpret_cast<const char*>(left.data),
                  reinterpret_cast<const char*>(right.data),
                  sizeof(left.data)) == 0;
  }
};

void print_hex_array(uint8_t* array);

void xxh3_128bits_hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value);

}  // namespace xllm_service
