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

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "chat_template/chat_template.h"
#include "common/types.h"
#include "request/chat_message.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

// A chat template implementation that uses jinja2 as the template engine.
class JinjaChatTemplate : public ChatTemplate {
 public:
  JinjaChatTemplate(const TokenizerArgs& args);

  bool encode_add_special_tokens() const override { return true; }

  std::optional<std::string> apply(const ChatMessages& messages) const;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm_service::JsonTool>& json_tools) const;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm_service::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const override;

  // expose this function for testing
  // apply the template to the values in the json object
  std::optional<std::string> apply(nlohmann::ordered_json& messages) const;

  std::optional<std::string> apply(nlohmann::ordered_json& messages,
                                   const nlohmann::ordered_json& tools) const;

  std::optional<std::string> apply(
      nlohmann::ordered_json& messages,
      const nlohmann::ordered_json& tools,
      const nlohmann::ordered_json& chat_template_kwargs) const;

 private:
  nlohmann::ordered_json get_mm_content(const Message::MMContentVec& vec) const;

 private:
  TokenizerArgs args_;
  std::unique_ptr<minja::chat_template> template_;
};

}  // namespace xllm_service
