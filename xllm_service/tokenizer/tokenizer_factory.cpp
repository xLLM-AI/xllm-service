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

#include "tokenizer_factory.h"

#include <filesystem>

#include "tokenizer_args.h"

namespace xllm_service {

std::unique_ptr<Tokenizer> TokenizerFactory::create_tokenizer(
    const std::string& model_weights_path,
    TokenizerArgs* tokenizer_args) {
  load_tokenizer_args(model_weights_path, *tokenizer_args);

  const std::string tokenizer_json_path =
      model_weights_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path)) {
    // 1. fast tokenizer
    LOG(INFO) << "Create fast tokenizer.";
    return std::make_unique<FastTokenizer>(tokenizer_json_path);
  } else if (tokenizer_args->tokenizer_type() == "tiktoken" ||
             tokenizer_args->tokenizer_class() == "TikTokenTokenizer") {
    // 2. create tiktoken tokenizer
    LOG(INFO) << "Create Tiktoken tokenizer.";
    return std::make_unique<TiktokenTokenizer>(model_weights_path,
                                               *tokenizer_args);
  } else {
    // 3. create sentencepiece tokenizer
    LOG(INFO) << "Create SentencePiece tokenizer.";
    return std::make_unique<SentencePieceTokenizer>(model_weights_path,
                                                    *tokenizer_args);
  }
}

}  // namespace xllm_service
