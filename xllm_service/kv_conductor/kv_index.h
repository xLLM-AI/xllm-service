/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

namespace xllm_service {

enum class KVEventType : uint8_t { STORED, REMOVED, CLEARED };
enum class KVMedium : uint8_t { DEVICE = 1, HOST = 2 };

struct KVEvent {
  KVEventType type = KVEventType::CLEARED;
  KVMedium medium = KVMedium::DEVICE;
  std::vector<uint64_t> hashes;
  std::optional<uint64_t> parent_hash;
  std::vector<int32_t> token_ids;
};

struct KVEventBatch {
  uint64_t sequence = 0;
  int32_t dp_rank = 0;
  std::vector<KVEvent> events;
};

struct KVPrefixMatch {
  bool synchronized = false;
  uint64_t sequence = 0;
  uint64_t device_hit_tokens = 0;
  uint64_t host_only_hit_tokens = 0;
  uint64_t reusable_prefix_tokens = 0;
  uint64_t device_blocks = 0;
  uint64_t host_blocks = 0;
  std::string error;
};

// One engine/DP stream. Engine IDs resolve removals; content keys resolve
// requests. Device and Host residency are independent bits of the same block.
class KVIndex final {
 public:
  KVIndex(uint32_t block_size, uint64_t hash_seed, size_t max_blocks);
  ~KVIndex();
  KVIndex(const KVIndex&) = delete;
  KVIndex& operator=(const KVIndex&) = delete;

  bool apply(const KVEventBatch& batch);
  bool restore(const KVEventBatch& snapshot,
               const std::vector<KVEventBatch>& following);
  void invalidate(std::string error);
  KVPrefixMatch match(const std::vector<int32_t>& token_ids) const;

 private:
  class State;
  bool apply_events(State& state,
                    const std::vector<KVEvent>& events,
                    bool snapshot = false) const;

  const uint32_t block_size_;
  const uint64_t hash_seed_;
  const size_t max_blocks_;
  mutable std::shared_mutex mutex_;
  std::unique_ptr<State> state_;
  uint64_t sequence_ = 0;
  bool synchronized_ = false;
  std::string error_ = "waiting for snapshot";
};

}  // namespace xllm_service
