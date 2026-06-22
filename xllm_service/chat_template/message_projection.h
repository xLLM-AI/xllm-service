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

#include "chat.pb.h"
#include "chat_template/jinja_chat_template.h"

namespace xllm_service {

// Flattens an ordered multimodal content vector into the single text string
// that the proto transport carries: only `text` blocks, joined by '\n'.
std::string flat_text(const Message::MMContentVec& blocks);

// Whether an ordered content vector must be kept as a structured vector on the
// canonical Message (more than one block, or a single non-text block).
bool needs_content_vec(const Message::MMContentVec& blocks);

// Projects a canonical Message onto the lossy proto ChatMessage transport.
// Writes into the pre-allocated `out` (arena friendly). Pure projection: it
// reproduces, field for field, what the request path used to build by hand.
void to_proto(const Message& msg, xllm::proto::ChatMessage* out);

}  // namespace xllm_service
