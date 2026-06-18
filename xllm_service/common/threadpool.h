/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
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

#include <functional>
#include <thread>

#include "concurrent_queue.h"

namespace xllm_service {

class ThreadPool final {
 public:
  using Task = std::function<void()>;

  // constructors
  ThreadPool() : ThreadPool(1) {}

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads);

  // destructor
  ~ThreadPool();

  // schedule a task to be executed
  void schedule(Task task);

 private:
  void internal_loop();

  std::vector<std::thread> threads_;
  ConcurrentQueue<Task> queue_;
};

}  // namespace xllm_service
