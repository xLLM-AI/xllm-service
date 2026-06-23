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

#include "chat_template/mm_content_text.h"

namespace xllm_service {

std::string flat_text(const Message::MMContentVec& blocks) {
  std::string text;
  bool first = true;
  for (const auto& block : blocks) {
    if (block.type != "text") {
      continue;
    }
    if (!first) {
      text += '\n';
    }
    text += block.text;
    first = false;
  }
  return text;
}

bool needs_content_vec(const Message::MMContentVec& blocks) {
  if (blocks.size() > 1) {
    return true;
  }
  return !blocks.empty() && blocks.front().type != "text";
}

}  // namespace xllm_service
