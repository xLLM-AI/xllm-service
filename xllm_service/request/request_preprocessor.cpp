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

#include "request/request_preprocessor.h"

#include "chat_template/deepseek_v4_cpp_chat_template.h"
#include "chat_template/jinja_chat_template.h"
#include "chat_template/model_type.h"
#include "tokenizer/tokenizer_factory.h"
namespace xllm_service {
RequestPreprocessor::RequestPreprocessor(const ServiceConfig& options)
    : is_vllm_(options.default_backend_type() == "vllm") {
  std::optional<std::string> model_type;

  // vLLM-backend clusters forward the raw client JSON to vLLM, which does its
  // own tokenization / chat templating. Skip building the local tokenizer and
  // chat template so the master does not require a tokenizer.model / template.
  if (options.default_backend_type() != "vllm") {
    CHECK_GT(options.num_threads(), 0);
    tokenizer_pool_ = std::make_unique<TokenizerPool>(
        TokenizerFactory::create_tokenizer(options.tokenizer_path(),
                                           &tokenizer_args_),
        static_cast<size_t>(options.num_threads()));

    // Select the chat template by config.json model_type (jinja by default).
    model_type = load_model_type(options.tokenizer_path());
    switch (select_chat_template_kind(model_type)) {
      case ChatTemplateKind::kDeepseekV4Cpp:
        chat_template_ =
            std::make_unique<DeepseekV4CppChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected DeepseekV4CppChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
      case ChatTemplateKind::kJinja:
        chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected JinjaChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
    }
  }
}

bool RequestPreprocessor::process(Request& request) {
  const bool is_vllm = is_vllm_;
  // apply chat template
  if (!is_vllm && request.messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request.tool_choice == "none" ? empty_tools : request.tools;
    auto prompt = chat_template_->apply(
        request.messages, tools_for_template, request.chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return false;
    }
    request.prompt = prompt.value();
  }

  // encode prompt
  if (!is_vllm && request.prompt.size() != 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }
    if (!tokenizer_pool_->encode(request.prompt,
                                 &request.token_ids,
                                 chat_template_->encode_add_special_tokens())) {
      LOG(ERROR) << "Encode prompt failed: " << request.prompt;
      return false;
    }
  }

  return true;
}
}  // namespace xllm_service
