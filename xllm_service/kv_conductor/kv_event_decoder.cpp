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

#include "kv_conductor/kv_event_decoder.h"

#include <limits>
#include <nlohmann/json.hpp>

namespace xllm_service {
namespace {

bool unsigned_value(const nlohmann::json& value, uint64_t* output) {
  if (!value.is_number_integer() ||
      (value.is_number_integer() && !value.is_number_unsigned() &&
       value.get<int64_t>() < 0)) {
    return false;
  }
  *output = value.get<uint64_t>();
  return true;
}

bool decode_event(const nlohmann::json& value,
                  uint32_t block_size,
                  KVEvent* event) {
  if (!value.is_object() || !value.contains("type") ||
      !value["type"].is_string()) {
    return false;
  }
  const std::string type = value["type"].get<std::string>();
  if (type == "AllBlocksCleared") {
    event->type = KVEventType::CLEARED;
    return true;
  }
  if (type != "BlockStored" && type != "BlockRemoved") {
    return false;
  }
  event->type =
      type == "BlockStored" ? KVEventType::STORED : KVEventType::REMOVED;
  if (!value.contains("medium") || !value["medium"].is_string()) {
    return false;
  }
  const std::string medium = value["medium"].get<std::string>();
  if (medium != "gpu" && medium != "cpu") {
    return false;
  }
  event->medium = medium == "gpu" ? KVMedium::DEVICE : KVMedium::HOST;
  if (!value.contains("block_hashes") || !value["block_hashes"].is_array() ||
      value["block_hashes"].empty()) {
    return false;
  }
  event->hashes.reserve(value["block_hashes"].size());
  for (const auto& hash : value["block_hashes"]) {
    uint64_t parsed = 0;
    if (!unsigned_value(hash, &parsed)) {
      return false;
    }
    event->hashes.emplace_back(parsed);
  }
  if (event->type == KVEventType::REMOVED) {
    return true;
  }
  // This adapter supports xLLM text caches, not additional KV hash contexts.
  for (const std::string& key : {"lora_id", "lora_name", "extra_keys"}) {
    if (value.contains(key) && !value[key].is_null()) {
      return false;
    }
  }
  uint64_t size = 0;
  if (!value.contains("block_size") ||
      !unsigned_value(value["block_size"], &size) || size != block_size ||
      !value.contains("token_ids") || !value["token_ids"].is_array()) {
    return false;
  }
  if (value.contains("parent_block_hash") &&
      !value["parent_block_hash"].is_null()) {
    uint64_t parent = 0;
    if (!unsigned_value(value["parent_block_hash"], &parent)) {
      return false;
    }
    event->parent_hash = parent;
  }
  const auto& tokens = value["token_ids"];
  if (!tokens.empty() && tokens.size() != event->hashes.size() * block_size) {
    return false;
  }
  event->token_ids.reserve(tokens.size());
  for (const auto& token : tokens) {
    uint64_t parsed = 0;
    if (!unsigned_value(token, &parsed) ||
        parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    event->token_ids.emplace_back(static_cast<int32_t>(parsed));
  }
  return true;
}

}  // namespace

bool decode_kv_batch(const std::string& sequence,
                     const std::string& payload,
                     int32_t dp_rank,
                     uint32_t block_size,
                     KVEventBatch* output) {
  if (sequence.size() != sizeof(uint64_t) || payload.empty() ||
      block_size == 0 || output == nullptr) {
    return false;
  }
  const nlohmann::json batch =
      nlohmann::json::from_msgpack(payload,
                                   /*strict=*/true,
                                   /*allow_exceptions=*/false);
  if (!batch.is_array() || batch.size() != 3 || !batch[0].is_number() ||
      !batch[1].is_array()) {
    return false;
  }
  uint64_t rank = 0;
  if (!unsigned_value(batch[2], &rank) ||
      rank != static_cast<uint64_t>(dp_rank)) {
    return false;
  }
  KVEventBatch result;
  result.dp_rank = dp_rank;
  for (unsigned char byte : sequence) {
    result.sequence = (result.sequence << 8) | byte;
  }
  if (result.sequence == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  result.events.reserve(batch[1].size());
  for (const auto& value : batch[1]) {
    KVEvent event;
    if (!decode_event(value, block_size, &event) ||
        (event.type == KVEventType::CLEARED && !result.events.empty())) {
      return false;
    }
    result.events.emplace_back(std::move(event));
  }
  *output = std::move(result);
  return true;
}

}  // namespace xllm_service
