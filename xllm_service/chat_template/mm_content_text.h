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

#include "chat_template/jinja_chat_template.h"

namespace xllm_service {

// Torch-free helpers over a multimodal content vector, in a standalone header
// so callers can reuse them without pulling in xllm/torch headers.

std::string flat_text(const Message::MMContentVec& blocks);

bool needs_content_vec(const Message::MMContentVec& blocks);

}  // namespace xllm_service
