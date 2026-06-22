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

#include <google/protobuf/service.h>

#include "butil/macros.h"

namespace xllm_service {

// RAII: Call Run() of the closure on destruction.
class ClosureGuard {
 public:
  ClosureGuard() : done_(nullptr) {}

  // Constructed with a closure which will be Run() inside dtor.
  explicit ClosureGuard(google::protobuf::Closure* done) : done_(done) {}

  // Run internal closure if it's not NULL.
  ~ClosureGuard() {
    if (done_) {
      done_->Run();
    }
  }

  // Run internal closure if it's not NULL and set it to `done'.
  void reset(google::protobuf::Closure* done) {
    if (done_) {
      done_->Run();
    }
    done_ = done;
  }

  // Return and set internal closure to NULL.
  google::protobuf::Closure* release() {
    google::protobuf::Closure* const prev_done = done_;
    done_ = nullptr;
    return prev_done;
  }

  // True if no closure inside.
  bool empty() const { return done_ == nullptr; }

  // Exchange closure with another guard.
  void swap(ClosureGuard& other) { std::swap(done_, other.done_); }

 private:
  // Copying this object makes no sense.
  DISALLOW_COPY_AND_ASSIGN(ClosureGuard);

  google::protobuf::Closure* done_ = nullptr;
};

}  // namespace xllm_service
