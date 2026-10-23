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

#include "request/output_dispatcher.h"

#include <future>
#include <utility>
namespace xllm_service {
void OutputDispatcher::schedule(const std::string& request_id,
                                ThreadPool::Task task) {
  workers_[std::hash<std::string>{}(request_id) % kWorkerCount].schedule(
      std::move(task));
}

void OutputDispatcher::drain() {
  for (auto& worker : workers_) {
    std::promise<void> done;
    auto completed = done.get_future();
    worker.schedule([&done] { done.set_value(); });
    completed.wait();
  }
}
}  // namespace xllm_service
