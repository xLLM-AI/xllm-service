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

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"

namespace xllm_service {

struct Message;
using ChatMessages = std::vector<Message>;

// Renders messages/tools into a prompt.
class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const = 0;

  // Whether the tokenizer should add special tokens when encoding the prompt.
  virtual bool encode_add_special_tokens() const = 0;
};

}  // namespace xllm_service
