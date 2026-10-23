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

#include "router/request_manager.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <utility>

#include "cluster/instance_state.h"
#include "common/metrics.h"
#include "common/xllm/status.h"
#include "request/request_metrics.h"
#include "request/request_output.h"
#include "router/routing_state.h"
namespace xllm_service {
RequestManager::RequestManager(std::shared_ptr<InstanceState> instances,
                               std::shared_ptr<RoutingState> routing)
    : instances_(std::move(instances)), routing_(std::move(routing)) {}

RequestManager::~RequestManager() { stop(); }

void RequestManager::drain() { outputs_.drain(); }

bool RequestManager::publish_request(std::shared_ptr<Request> request) {
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  std::lock_guard<std::mutex> request_lock(request_mutex_);
  if (stopped_) {
    routing_->update_request_metrics(request, RequestAction::CANCEL);
    return false;
  }
  if (requests_.find(request->service_request_id) != requests_.end()) {
    LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                  "are not allowed. "
               << request->service_request_id;
    return false;
  }
  // schedule() precedes callback registration. The selected instance may have
  // been removed or replaced in that interval, before cleanup could see us.
  if (!instances_->matches(request->routing,
                           request->prefill_incarnation_id,
                           request->decode_incarnation_id)) {
    routing_->update_request_metrics(request, RequestAction::CANCEL);
    LOG(ERROR) << "Instance changed before request admission: "
               << request->service_request_id;
    return false;
  }
  requests_.emplace(request->service_request_id, request);
  COUNTER_INC(server_request_in_total);
  return true;
}

void RequestManager::finish_request(const std::string& service_request_id,
                                    bool error) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it != requests_.end()) {
      request = it->second;
      routing_->update_request_metrics(
          request,
          error ? RequestAction::CANCEL : RequestAction::FINISH_DECODE);
      requests_.erase(it);
    }
  }
}

void RequestManager::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::shared_ptr<Request>> cleared_requests;
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    for (const auto& [service_request_id, request] : requests_) {
      const bool clear_prefill =
          ((type == InstanceType::DEFAULT || type == InstanceType::PREFILL ||
            type == InstanceType::MIX) &&
           request->routing.prefill_name == instance_name &&
           request->prefill_incarnation_id == incarnation_id &&
           !request->prefill_stage_finished);
      const bool clear_decode =
          ((type == InstanceType::DECODE || type == InstanceType::MIX) &&
           request->routing.decode_name == instance_name &&
           request->decode_incarnation_id == incarnation_id);
      if (!clear_prefill && !clear_decode) {
        continue;
      }
      cleared_requests.emplace_back(request);
    }
  }
  for (const auto& request : cleared_requests) {
    llm::RequestOutput output;
    output.service_request_id = request->service_request_id;
    output.status = llm::Status(llm::StatusCode::CANCELLED,
                                "Instance is failed and deleted");
    handle_generation(std::move(output));
    LOG(INFO) << "Clear request on failed instance: " << instance_name
              << ", incarnation_id: " << incarnation_id
              << ", service_request_id: " << request->service_request_id;
  }
}

bool RequestManager::handle_generation(llm::RequestOutput request_output) {
  const bool finished_on_prefill_instance =
      request_output.finished_on_prefill_instance;
  std::string service_request_id = request_output.service_request_id;
  const bool status_error =
      request_output.status.has_value() && !request_output.status.value().ok();

  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it == requests_.end()) {
      LOG(ERROR) << "Can not found the callback for the received request "
                    "output, request id is: "
                 << service_request_id;
      return false;
    }
    request = it->second;

    // check client connection
    if (request->is_disconnected && request->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      requests_.erase(it);
      routing_->update_request_metrics(request, RequestAction::CANCEL);

      return false;
    } else if (!status_error) {
      const bool first_token =
          update_request_metrics(request, finished_on_prefill_instance);
      update_token_latency_metrics(request, first_token);
    }
    const std::string dispatch_id = service_request_id;
    outputs_.schedule(dispatch_id,
                      [this,
                       service_request_id = std::move(service_request_id),
                       request = std::move(request),
                       request_output = std::move(request_output)]() mutable {
                        const std::optional<bool> finished =
                            deliver_request_output(*request, request_output);
                        if (finished.has_value()) {
                          finish_request(service_request_id, *finished);
                        }
                      });
  }

  return true;
}

bool RequestManager::update_request_metrics(
    const std::shared_ptr<Request>& request,
    bool finished_on_prefill_instance) {
  const bool first_token =
      record_generation(*request, finished_on_prefill_instance);
  routing_->update_request_metrics(
      request,
      first_token ? RequestAction::FINISH_PREFILL : RequestAction::GENERATE);
  return first_token;
}

void RequestManager::update_token_latency_metrics(
    const std::shared_ptr<Request>& request,
    bool first_token) {
  int64_t tbt_milliseconds =
      absl::ToInt64Milliseconds(absl::Now() - request->latest_generate_time);
  request->latest_generate_time = absl::Now();
  if (first_token) {
    HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                      tbt_milliseconds);
  } else {
    HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
  }
}

void RequestManager::stop() {
  {
    std::lock_guard admission_lock(admission_mutex_);
    std::lock_guard request_lock(request_mutex_);
    stopped_ = true;
    for (const auto& [id, request] : requests_) {
      llm::RequestOutput output;
      output.service_request_id = id;
      output.status =
          llm::Status(llm::StatusCode::CANCELLED, "Service is stopping");
      outputs_.schedule(id, [this, request, output = std::move(output)] {
        const auto finished = deliver_request_output(*request, output);
        if (finished.has_value()) {
          finish_request(request->service_request_id, *finished);
        }
      });
    }
  }
  drain();
}
bool RequestManager::retire_instance(const std::string& name,
                                     const std::string& incarnation_id) {
  std::lock_guard admission_lock(admission_mutex_);
  InstanceMetaInfo removed;
  if (!instances_->remove_instance(name, incarnation_id, &removed)) {
    return false;
  }
  // A concurrent publication either precedes this cleanup or revalidates
  // against the removed incarnation while holding this same barrier.
  clear_requests_on_failed_instance(name, incarnation_id, removed.type);
  routing_->forget_instance(name, incarnation_id);
  return true;
}

}  // namespace xllm_service
