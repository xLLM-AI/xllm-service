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

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "chat_template/chat_template.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm {
class DeepseekV4CppTemplate;
}  // namespace xllm

namespace xllm_service {

// Renders DeepSeek V4 prompts via the upstream xllm::DeepseekV4CppTemplate; all
// xllm/torch types are kept out of this header. The V4 template writes BOS/role
// markers itself, so encode_add_special_tokens() is false to avoid a double
// BOS.
class DeepseekV4CppChatTemplate : public ChatTemplate {
 public:
  explicit DeepseekV4CppChatTemplate(const TokenizerArgs& args);
  ~DeepseekV4CppChatTemplate() override;

  bool encode_add_special_tokens() const override { return false; }

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const override;

 private:
  std::unique_ptr<xllm::DeepseekV4CppTemplate> impl_;
};

}  // namespace xllm_service
