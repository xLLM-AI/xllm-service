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

#include <string>

#include "kv_conductor/kv_index.h"

namespace xllm_service {

// Decodes the current xLLM vLLM-style map events, with an array batch envelope.
// Unknown event kinds and media invalidate the stream instead of inventing
// hits.
bool decode_kv_batch(const std::string& sequence,
                     const std::string& payload,
                     int32_t dp_rank,
                     uint32_t block_size,
                     KVEventBatch* output);

}  // namespace xllm_service
