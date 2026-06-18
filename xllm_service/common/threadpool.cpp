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

#include "common/threadpool.h"

#include <thread>

#include "common/concurrent_queue.h"

namespace xllm_service {

ThreadPool::ThreadPool(size_t num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() { internal_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < threads_.size(); ++i) {
    queue_.push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : threads_) {
    thread.join();
  }
}

// schedule a task to be executed
void ThreadPool::schedule(Task task) {
  if (task == nullptr) {
    return;
  }
  queue_.push(std::move(task));
}

void ThreadPool::internal_loop() {
  while (true) {
    Task task = queue_.pop();
    if (task == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    task();
  }
}

}  // namespace xllm_service
