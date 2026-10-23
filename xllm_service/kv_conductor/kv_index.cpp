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

#include "kv_conductor/kv_index.h"

#include <glog/logging.h>
#include <xxhash.h>

#include <array>
#include <mutex>
#include <unordered_map>

namespace xllm_service {
namespace {

std::string content_key(const std::string& parent,
                        const int32_t* tokens,
                        uint32_t count,
                        uint64_t seed) {
  std::vector<uint8_t> bytes;
  bytes.reserve(parent.size() + sizeof(int32_t) * count);
  bytes.insert(bytes.end(), parent.begin(), parent.end());
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t token = static_cast<uint32_t>(tokens[index]);
    for (uint32_t shift = 0; shift < 32; shift += 8) {
      bytes.emplace_back(static_cast<uint8_t>(token >> shift));
    }
  }
  const XXH128_hash_t hash =
      XXH3_128bits_withSeed(bytes.data(), bytes.size(), seed);
  XXH128_canonical_t canonical;
  XXH128_canonicalFromHash(&canonical, hash);
  return std::string(reinterpret_cast<const char*>(canonical.digest),
                     sizeof(canonical.digest));
}

}  // namespace

class KVIndex::State final {
 public:
  struct Block {
    std::string key;
    uint8_t media = 0;
  };
  std::unordered_map<uint64_t, Block> blocks;
  std::unordered_map<std::string, std::array<uint32_t, 2>> residents;
  uint64_t device_blocks = 0;
  uint64_t host_blocks = 0;
};

KVIndex::KVIndex(uint32_t block_size, uint64_t hash_seed, size_t max_blocks)
    : block_size_(block_size),
      hash_seed_(hash_seed),
      max_blocks_(max_blocks),
      state_(std::make_unique<State>()) {
  CHECK_GT(block_size_, 0);
  CHECK_GT(max_blocks_, 0);
}

KVIndex::~KVIndex() = default;

bool KVIndex::apply_events(State& state,
                           const std::vector<KVEvent>& events,
                           bool snapshot) const {
  for (const auto& event : events) {
    if (event.type == KVEventType::CLEARED) {
      continue;
    }
    const uint8_t medium = static_cast<uint8_t>(event.medium);
    if ((event.medium != KVMedium::DEVICE && event.medium != KVMedium::HOST) ||
        (event.type == KVEventType::STORED && !event.token_ids.empty() &&
         (event.token_ids.size() % block_size_ != 0 ||
          event.token_ids.size() / block_size_ != event.hashes.size()))) {
      return false;
    }
    const size_t tier = event.medium == KVMedium::DEVICE ? 0 : 1;
    std::string parent;
    if (event.type == KVEventType::STORED && !event.token_ids.empty() &&
        event.parent_hash) {
      const auto found = state.blocks.find(*event.parent_hash);
      if (found == state.blocks.end()) {
        // A current snapshot may contain an isolated suffix whose parent has
        // been evicted from both tiers. It cannot extend any reusable prefix.
        // xLLM snapshots order parent blocks before children, so skip it and
        // retain other independent prefixes. Incremental loss needs recovery.
        if (snapshot) {
          continue;
        }
        return false;
      }
      parent = found->second.key;
    }
    for (size_t index = 0; index < event.hashes.size(); ++index) {
      const uint64_t hash = event.hashes[index];
      auto found = state.blocks.find(hash);
      if (event.type == KVEventType::REMOVED) {
        if (found == state.blocks.end() || !(found->second.media & medium)) {
          continue;
        }
        auto resident = state.residents.find(found->second.key);
        --resident->second[tier];
        if (resident->second[0] == 0 && resident->second[1] == 0) {
          state.residents.erase(resident);
        }
        found->second.media &= static_cast<uint8_t>(~medium);
        if (tier == 0) {
          --state.device_blocks;
        } else {
          --state.host_blocks;
        }
        // Retain the content key for future children or tokenless Host stores.
        // max_blocks bounds this metadata; a snapshot compacts dead entries.
        continue;
      }
      if (!event.token_ids.empty()) {
        parent = content_key(parent,
                             event.token_ids.data() + index * block_size_,
                             block_size_,
                             hash_seed_);
        if (found != state.blocks.end() && found->second.key != parent) {
          return false;
        }
        if (found == state.blocks.end()) {
          if (state.blocks.size() >= max_blocks_) {
            return false;
          }
          found = state.blocks.emplace(hash, State::Block{parent, 0}).first;
        }
      }
      if (found == state.blocks.end()) {
        return false;
      }
      if (found->second.media & medium) {
        continue;
      }
      found->second.media |= medium;
      ++state.residents[found->second.key][tier];
      if (tier == 0) {
        ++state.device_blocks;
      } else {
        ++state.host_blocks;
      }
    }
  }
  return true;
}

bool KVIndex::apply(const KVEventBatch& batch) {
  const bool snapshot = !batch.events.empty() &&
                        batch.events.front().type == KVEventType::CLEARED;
  if (snapshot) {
    return restore(batch, {});
  }
  // Destroy failed state after releasing the lock, as in invalidate().
  std::unique_ptr<State> retired;
  std::unique_lock lock(mutex_);
  if (!synchronized_) {
    return false;
  }
  if (batch.sequence <= sequence_) {
    return true;
  }
  if (batch.sequence != sequence_ + 1 || !apply_events(*state_, batch.events)) {
    retired = std::make_unique<State>();
    state_.swap(retired);
    synchronized_ = false;
    error_ = "sequence gap, unresolved content, or index capacity exceeded";
    return false;
  }
  sequence_ = batch.sequence;
  return true;
}

bool KVIndex::restore(const KVEventBatch& snapshot,
                      const std::vector<KVEventBatch>& following) {
  auto state = std::make_unique<State>();
  if (snapshot.events.empty() ||
      snapshot.events.front().type != KVEventType::CLEARED ||
      !apply_events(*state, snapshot.events, /*snapshot=*/true)) {
    invalidate("invalid snapshot or index capacity exceeded");
    return false;
  }
  uint64_t sequence = snapshot.sequence;
  for (const auto& batch : following) {
    if (batch.sequence <= sequence) {
      continue;
    }
    if (batch.sequence != sequence + 1 || !apply_events(*state, batch.events)) {
      invalidate("sequence gap while merging snapshot and live events");
      return false;
    }
    sequence = batch.sequence;
  }
  std::unique_lock lock(mutex_);
  state_.swap(state);
  sequence_ = sequence;
  synchronized_ = true;
  error_.clear();
  return true;
}

void KVIndex::invalidate(std::string error) {
  auto empty = std::make_unique<State>();
  std::unique_lock lock(mutex_);
  state_.swap(empty);
  synchronized_ = false;
  error_ = std::move(error);
}

KVPrefixMatch KVIndex::match(const std::vector<int32_t>& token_ids) const {
  std::shared_lock lock(mutex_);
  KVPrefixMatch result;
  result.synchronized = synchronized_;
  result.sequence = sequence_;
  result.error = error_;
  if (!synchronized_) {
    return result;
  }
  result.device_blocks = state_->device_blocks;
  result.host_blocks = state_->host_blocks;
  std::string key;
  for (size_t offset = 0; offset + block_size_ <= token_ids.size();
       offset += block_size_) {
    key = content_key(key, token_ids.data() + offset, block_size_, hash_seed_);
    const auto found = state_->residents.find(key);
    if (found == state_->residents.end()) {
      break;
    }
    if (found->second[0] != 0) {
      result.device_hit_tokens += block_size_;
    } else {
      result.host_only_hit_tokens += block_size_;
    }
    result.reusable_prefix_tokens += block_size_;
  }
  return result;
}

}  // namespace xllm_service
