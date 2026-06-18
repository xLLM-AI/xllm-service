/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "tokenizer_args.h"

#include <fstream>

#include "common/json_reader.h"

namespace xllm_service {
namespace {
std::optional<std::string> load_chat_template_file(const std::string& dir) {
  // chat_template.json
  const std::string chat_template_path = dir + "/chat_template.json";
  JsonReader reader;
  if (reader.parse(chat_template_path);
      auto v = reader.value<std::string>("chat_template")) {
    return v;
  }
  // chat_template.jinja
  const std::string raw_chat_template_path = dir + "/chat_template.jinja";
  std::ifstream file(raw_chat_template_path);
  if (file.is_open()) {
    std::ostringstream content;
    content << file.rdbuf();
    file.close();
    return content.str();
  }
  return std::nullopt;
}
}  // namespace

void load_tokenizer_args(const std::string& model_weights_path,
                         TokenizerArgs& tokenizer_args) {
  // tokenizer args from tokenizer_config.json
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path + "/tokenizer_config.json";
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
    if (auto v = load_chat_template_file(model_weights_path)) {
      tokenizer_args.chat_template() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args.chat_template() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_bos_token")) {
      tokenizer_args.add_bos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_eos_token")) {
      tokenizer_args.add_eos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<std::string>("tokenizer_class")) {
      tokenizer_args.tokenizer_class() = v.value();
    }
    // read bos_token
    if (auto v = tokenizer_reader.value<std::string>("bos_token.content")) {
      tokenizer_args.bos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("bos_token")) {
      tokenizer_args.bos_token() = v.value();
    }
    // read eos_token
    if (auto v = tokenizer_reader.value<std::string>("eos_token.content")) {
      tokenizer_args.eos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("eos_token")) {
      tokenizer_args.eos_token() = v.value();
    }
    // read pad_token
    if (auto v = tokenizer_reader.value<std::string>("pad_token.content")) {
      tokenizer_args.pad_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("pad_token")) {
      tokenizer_args.pad_token() = v.value();
    }
  }
}

}  // namespace xllm_service