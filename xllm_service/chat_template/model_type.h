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

#include <optional>
#include <string>

namespace xllm_service {

enum class ChatTemplateKind {
  kJinja,
  kDeepseekV4Cpp,
};

ChatTemplateKind select_chat_template_kind(
    const std::optional<std::string>& model_type);

std::optional<std::string> load_model_type(
    const std::string& model_weights_path);

}  // namespace xllm_service
