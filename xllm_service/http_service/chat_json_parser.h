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

#include <string>
#include <utility>

namespace xllm_service {

struct ChatJsonResult {
  bool ok = true;
  std::string json;
  std::string error;
};

// Normalize OpenAI-style chat JSON for the text-only ChatRequest proto.
// Text-only content arrays are collapsed to a string before protobuf parsing.
ChatJsonResult normalize_chat_json(std::string json_str);

}  // namespace xllm_service
