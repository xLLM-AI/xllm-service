/* Copyright 2026 The xLLM Authors.

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

#include "chat_template/chat_template.h"
#include "config/service_config.h"
#include "request/request.h"
#include "tokenizer/tokenizer_args.h"
#include "tokenizer/tokenizer_pool.h"
namespace xllm_service {
class RequestPreprocessor final {
 public:
  explicit RequestPreprocessor(const ServiceConfig& options);
  bool process(Request& request);

 private:
  bool is_vllm_ = false;
  TokenizerArgs tokenizer_args_;
  std::unique_ptr<ChatTemplate> chat_template_;
  std::unique_ptr<TokenizerPool> tokenizer_pool_;
};
}  // namespace xllm_service
