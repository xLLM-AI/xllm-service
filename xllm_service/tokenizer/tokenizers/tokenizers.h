/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

// The C API
#ifdef __cplusplus
extern "C" {
#endif

// The C interface to the hf-tokenizers library
// ported from https://github.com/mlc-ai/tokenizers-cpp
#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

typedef struct {
  int* token_ids;
  size_t len;
} TokenizerEncodeResult;

TokenizerHandle tokenizers_new_from_path(const char* path);

void tokenizers_encode(TokenizerHandle handle,
                       const char* data,
                       size_t len,
                       int add_special_token,
                       TokenizerEncodeResult* result);

void tokenizers_decode(TokenizerHandle handle,
                       const uint32_t* data,
                       size_t len,
                       int skip_special_tokens,
                       const char** decode_data,
                       size_t* decode_len);

void tokenizers_id_to_token(TokenizerHandle handle,
                            uint32_t id,
                            const char** data,
                            size_t* len);

// tokenizers_token_to_id stores -1 to *id if the token is not in the vocab
void tokenizers_token_to_id(TokenizerHandle handle,
                            const char* token,
                            size_t len,
                            int32_t* id);

void tokenizers_free(TokenizerHandle handle);

void tokenizers_get_vocab_size(TokenizerHandle handle, size_t* size);

#ifdef __cplusplus
}
#endif
