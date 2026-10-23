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

#include "scheduler/loadbalance_policy/mooncake_prefix_lookup.h"

#include <glog/logging.h>
#include <xxhash.h>

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

namespace xllm_service {
namespace {

std::string decode_prefix(const std::string& hex) {
  CHECK(!hex.empty() && hex.size() % 2 == 0 && hex.size() <= 8192)
      << "Expected an even-length hex Store key prefix";
  const std::string digits = "0123456789abcdef";
  std::string prefix;
  prefix.reserve(hex.size() / 2);
  for (size_t offset = 0; offset < hex.size(); offset += 2) {
    const size_t high = digits.find(hex[offset]);
    const size_t low = digits.find(hex[offset + 1]);
    CHECK(high != std::string::npos && low != std::string::npos)
        << "Store prefix must use lowercase hex";
    prefix += static_cast<char>((high << 4) | low);
  }
  CHECK(prefix.rfind("xllm-kv-v3:", 0) == 0 && prefix.size() > 26)
      << "Expected a complete xllm-kv-v3 prefix including schema hash";
  return prefix;
}

std::vector<std::string> block_hashes(const std::vector<int32_t>& tokens,
                                      const StorePrefixLayout& layout,
                                      size_t max_blocks) {
  const size_t lookahead = static_cast<size_t>(layout.hash_lookahead);
  const size_t block_size = static_cast<size_t>(layout.block_size);
  // One token remains for logits even when the input is block-aligned.
  const size_t count =
      std::min(max_blocks,
               (tokens.size() - std::max<size_t>(1, lookahead)) / block_size);
  std::vector<std::string> hashes;
  hashes.reserve(count);
  std::string parent;
  std::string bytes;
  bytes.reserve(16 + (block_size + lookahead) * sizeof(int32_t));
  for (size_t block = 0; block < count; ++block) {
    bytes = parent;
    for (size_t index = block * block_size;
         index < (block + 1) * block_size + lookahead;
         ++index) {
      // xLLM's native int32/hash bytes on supported little-endian hosts.
      const uint32_t token = static_cast<uint32_t>(tokens[index]);
      for (int32_t shift = 0; shift < 32; shift += 8) {
        bytes += static_cast<char>((token >> shift) & 0xff);
      }
    }
    const XXH128_hash_t hash =
        XXH3_128bits_withSeed(bytes.data(), bytes.size(), layout.hash_seed);
    parent.clear();
    for (uint64_t word : {hash.low64, hash.high64}) {
      for (int32_t shift = 0; shift < 64; shift += 8) {
        parent += static_cast<char>((word >> shift) & 0xff);
      }
    }
    hashes.emplace_back(parent);
  }
  return hashes;
}

}  // namespace

MooncakePrefixLookup::MooncakePrefixLookup(
    std::vector<StorePrefixLayout> layouts,
    int32_t max_keys,
    std::unique_ptr<StoreKeyLookup> client)
    : layouts_(std::move(layouts)),
      max_keys_(max_keys),
      client_(std::move(client)) {
  CHECK(client_ != nullptr);
  CHECK(!layouts_.empty());
  CHECK_GT(max_keys_, 0);
  size_t required_keys = 0;
  std::unordered_set<std::string> names;
  for (const auto& layout : layouts_) {
    CHECK_GT(layout.block_size, 0);
    CHECK_LE(layout.block_size, 4096);
    CHECK(layout.hash_lookahead == 0 || layout.hash_lookahead == 1);
    CHECK(!layout.key_prefixes.empty() && !layout.instances.empty());
    required_keys += layout.key_prefixes.size();
    for (const std::string& name : layout.instances) {
      CHECK(names.emplace(name).second)
          << "Duplicate Store layout for " << name;
    }
  }
  CHECK_LE(required_keys, static_cast<size_t>(max_keys_));
}

std::unordered_map<std::string, PrefillCacheHint> MooncakePrefixLookup::match(
    const std::vector<int32_t>& tokens,
    const std::vector<InstanceMetaInfo>& instances) {
  std::unordered_map<std::string, PrefillCacheHint> result;
  if (tokens.empty()) {
    return result;
  }
  std::unordered_map<std::string, std::string> incarnations;
  for (const auto& instance : instances) {
    incarnations.emplace(instance.name, instance.incarnation_id);
  }
  std::vector<std::string> keys;
  keys.reserve(static_cast<size_t>(max_keys_));
  std::vector<size_t> block_counts;
  block_counts.reserve(layouts_.size());
  // Divide the query budget between layouts so every engine has the same
  // conservative prefix horizon. All required shards must exist per block.
  size_t keys_per_block = 0;
  for (const auto& layout : layouts_) {
    keys_per_block += layout.key_prefixes.size();
  }
  const size_t max_blocks = static_cast<size_t>(max_keys_) / keys_per_block;
  for (const auto& layout : layouts_) {
    const auto hashes = block_hashes(tokens, layout, max_blocks);
    block_counts.emplace_back(hashes.size());
    for (const std::string& hash : hashes) {
      for (const std::string& prefix : layout.key_prefixes) {
        keys.emplace_back(prefix + hash);
      }
    }
  }
  if (keys.empty()) {
    return result;
  }
  std::vector<bool> exists;
  if (!client_->batch_exists(keys, &exists) || exists.size() != keys.size()) {
    LOG_EVERY_N(WARNING, 100) << "Mooncake prefix metadata unavailable; "
                                 "estimating uncached prefill";
    return result;
  }
  size_t offset = 0;
  for (size_t index = 0; index < layouts_.size(); ++index) {
    const auto& layout = layouts_[index];
    const size_t shards = layout.key_prefixes.size();
    size_t hit_blocks = 0;
    for (; hit_blocks < block_counts[index]; ++hit_blocks) {
      const auto begin = exists.begin() + offset + hit_blocks * shards;
      if (!std::all_of(begin, begin + shards, [](bool hit) { return hit; })) {
        break;
      }
    }
    offset += block_counts[index] * shards;
    for (const std::string& name : layout.instances) {
      const auto incarnation = incarnations.find(name);
      if (incarnation == incarnations.end()) {
        continue;
      }
      result.emplace(name,
                     PrefillCacheHint{
                         incarnation->second,
                         static_cast<int32_t>(hit_blocks * layout.block_size)});
    }
  }
  return result;
}

std::unique_ptr<MooncakePrefixLookup> create_mooncake_prefix_lookup(
    const std::string& config_file) {
  if (config_file.empty()) {
    return nullptr;
  }
  std::ifstream input(config_file);
  CHECK(input.is_open()) << "Cannot open " << config_file;
  const auto config = nlohmann::json::parse(input, nullptr, false);
  CHECK(config.is_object()) << "Invalid Mooncake prefix configuration";
  CHECK(config.contains("master_address") &&
        config["master_address"].is_string());
  const std::string address = config["master_address"].get<std::string>();
  CHECK(address.find(':') != std::string::npos);
  const int32_t timeout_ms = config.value("timeout_ms", 20);
  const int32_t max_keys = config.value("max_keys", 4096);
  CHECK_GT(timeout_ms, 0);
  CHECK_LE(timeout_ms, 1000);
  CHECK_GT(max_keys, 0);
  CHECK_LE(max_keys, 65536);
  CHECK(config.contains("layouts") && config["layouts"].is_array());
  std::vector<StorePrefixLayout> layouts;
  layouts.reserve(config["layouts"].size());
  for (const auto& item : config["layouts"]) {
    StorePrefixLayout layout;
    layout.instances = item.at("instances").get<std::vector<std::string>>();
    layout.block_size = item.value("block_size", 128);
    layout.hash_seed = item.value("hash_seed", uint64_t{1024});
    layout.hash_lookahead = item.value("hash_lookahead", 0);
    const auto prefixes =
        item.at("key_prefixes_hex").get<std::vector<std::string>>();
    layout.key_prefixes.reserve(prefixes.size());
    for (const std::string& prefix : prefixes) {
      layout.key_prefixes.emplace_back(decode_prefix(prefix));
    }
    layouts.emplace_back(std::move(layout));
  }
  auto client = create_mooncake_key_lookup(
      address, config.value("tenant_id", std::string("default")), timeout_ms);
  return std::make_unique<MooncakePrefixLookup>(
      std::move(layouts), max_keys, std::move(client));
}

#ifndef USE_MOONCAKE_PREFIX_LOOKUP
std::unique_ptr<StoreKeyLookup> create_mooncake_key_lookup(
    std::string /*master_address*/,
    std::string /*tenant_id*/,
    int32_t /*timeout_ms*/) {
  LOG(FATAL) << "Rebuild with USE_MOONCAKE_PREFIX_LOOKUP=ON to query Store";
  return nullptr;
}
#endif

}  // namespace xllm_service
