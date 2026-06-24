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

#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

// Records the add_special_tokens flag passed to encode.
class RecordingTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    last_add_special_tokens = add_special_tokens;
    return true;
  }
  std::string decode(const Slice<int32_t>&, bool) const override { return ""; }
  std::optional<int32_t> token_to_id(const std::string_view&) const override {
    return std::nullopt;
  }
  std::string id_to_token(int32_t) const override { return ""; }
  size_t vocab_size() const override { return 0; }
  std::unique_ptr<Tokenizer> clone() const override { return nullptr; }

  mutable bool last_add_special_tokens = false;
};

}  // namespace

TEST(Tokenizer, EncodeDefaultsToAddingSpecialTokens) {
  RecordingTokenizer tokenizer;
  std::vector<int32_t> ids;
  tokenizer.encode("hi", &ids);
  EXPECT_TRUE(tokenizer.last_add_special_tokens);
}

TEST(Tokenizer, EncodeHonorsExplicitFalse) {
  RecordingTokenizer tokenizer;
  std::vector<int32_t> ids;
  tokenizer.encode("hi", &ids, /*add_special_tokens=*/false);
  EXPECT_FALSE(tokenizer.last_add_special_tokens);
}

}  // namespace xllm_service
