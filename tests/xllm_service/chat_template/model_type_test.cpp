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

#include "chat_template/model_type.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace xllm_service {
namespace {

// Writes `content` to config.json in a fresh temp dir and returns that dir.
std::string make_model_dir_with_config(const std::string& content) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("model_type_test_" +
                    std::to_string(reinterpret_cast<uintptr_t>(&content)) +
                    std::to_string(std::rand()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "config.json") << content;
  return dir.string();
}

TEST(SelectChatTemplateKind, DeepseekV4SelectsCppTemplate) {
  EXPECT_EQ(select_chat_template_kind("deepseek_v4"),
            ChatTemplateKind::kDeepseekV4Cpp);
}

TEST(SelectChatTemplateKind, UnknownModelTypeFallsBackToJinja) {
  EXPECT_EQ(select_chat_template_kind("qwen3"), ChatTemplateKind::kJinja);
}

TEST(SelectChatTemplateKind, MissingModelTypeFallsBackToJinja) {
  EXPECT_EQ(select_chat_template_kind(std::nullopt), ChatTemplateKind::kJinja);
}

TEST(LoadModelType, ReadsModelTypeFromConfigJson) {
  const auto dir =
      make_model_dir_with_config(R"({"model_type": "deepseek_v4"})");
  EXPECT_EQ(load_model_type(dir), "deepseek_v4");
}

TEST(LoadModelType, MissingConfigJsonReturnsNullopt) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("model_type_test_missing_" + std::to_string(std::rand()));
  std::filesystem::create_directories(dir);
  EXPECT_FALSE(load_model_type(dir.string()).has_value());
}

TEST(LoadModelType, ConfigWithoutModelTypeReturnsNullopt) {
  const auto dir = make_model_dir_with_config(R"({"hidden_size": 4096})");
  EXPECT_FALSE(load_model_type(dir).has_value());
}

}  // namespace
}  // namespace xllm_service
