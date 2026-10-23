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

#include "router/routing_state.h"

#include <algorithm>
#include <utility>

#include "request/request_metrics.h"

namespace xllm_service {
RoutingState::Reservation::Reservation(const InstanceMetaInfo& info)
    : incarnation_id(info.incarnation_id),
      predictor(info.ttft_profiling_data, info.tpot_profiling_data) {}

RoutingState::RoutingState(const ServiceConfig& options,
                           std::shared_ptr<InstanceState> instances)
    : enabled_(options.load_balance_policy() == "SLO_AWARE"),
      instances_(std::move(instances)) {}

RoutingState::Reservation& RoutingState::reservation_for(
    const InstanceMetaInfo& info) {
  auto& reservation = reservations_[info.name];
  if (!reservation || reservation->incarnation_id != info.incarnation_id) {
    reservation = std::make_unique<Reservation>(info);
  }
  return *reservation;
}

bool RoutingState::bind_request(Request& request) const {
  request.prefill_incarnation_id.clear();
  request.decode_incarnation_id.clear();
  const auto snapshot = instances_->snapshot();
  for (const auto& view : snapshot) {
    const auto& info = *view.info;
    if (info.name == request.routing.prefill_name) {
      request.prefill_incarnation_id = info.incarnation_id;
    }
    if (info.name == request.routing.decode_name) {
      request.decode_incarnation_id = info.incarnation_id;
    }
  }
  return instances_->matches(request.routing,
                             request.prefill_incarnation_id,
                             request.decode_incarnation_id);
}

bool RoutingState::select_route(Request& request,
                                RoutingPolicy& policy,
                                std::vector<KVDirectoryMatch> matches) {
  request.routing = {};
  request.estimated_ttft = 0;
  const bool slo = enabled_ && !request.token_ids.empty();
  std::unique_lock reservation_lock(mutex_, std::defer_lock);
  if (slo) {
    reservation_lock.lock();
  }
  const auto snapshot = instances_->snapshot();
  RoutingInput input;
  input.candidates.prefill.reserve(snapshot.size());
  input.candidates.decode.reserve(snapshot.size());
  input.prompt_tokens = request.token_ids.size();
  input.prefix_matches = std::move(matches);
  for (const auto& view : snapshot) {
    if (view.runtime_state == InstanceRuntimeState::SUSPECT) {
      continue;
    }
    const auto& info = *view.info;
    RoutingCandidate candidate;
    candidate.name = info.name;
    candidate.incarnation_id = info.incarnation_id;
    candidate.register_ts_ms = info.register_ts_ms;
    candidate.dp_size = info.dp_size;
    candidate.type = info.type;
    candidate.has_load = view.load.has_value();
    if (view.load.has_value()) {
      candidate.load = *view.load;
    }
    if (slo) {
      auto& reservation = reservation_for(info);
      candidate.prefill_time = reservation.metrics.estimated_prefill_time;
      candidate.estimated_ttft =
          reservation.predictor.predict_ttft(input.prompt_tokens);
      if (view.role == InstanceType::DECODE) {
        candidate.estimated_tpot = reservation.predictor.predict_tpot(
            reservation.metrics.decode_token_num + input.prompt_tokens,
            reservation.metrics.decode_request_num + 1);
      }
    }
    auto& target = view.role == InstanceType::DECODE ? input.candidates.decode
                                                     : input.candidates.prefill;
    target.emplace_back(std::move(candidate));
  }
  RoutingDecision decision;
  if (!policy.select(input, &decision)) {
    return false;
  }
  request.routing = std::move(decision.routing);
  request.estimated_ttft = decision.estimated_ttft;
  // Bind the exact incarnations used by the policy, rather than a replacement
  // discovered between selection and admission.
  request.prefill_incarnation_id.clear();
  request.decode_incarnation_id.clear();
  for (const auto& view : snapshot) {
    const auto& info = *view.info;
    if (info.name == request.routing.prefill_name) {
      request.prefill_incarnation_id = info.incarnation_id;
    }
    if (info.name == request.routing.decode_name) {
      request.decode_incarnation_id = info.incarnation_id;
    }
  }
  if (!instances_->matches(request.routing,
                           request.prefill_incarnation_id,
                           request.decode_incarnation_id)) {
    request.routing = {};
    return false;
  }
  if (slo) {
    // Selection and its local reservations remain one serialized transaction.
    update_locked(request, RequestAction::SCHEDULE);
    if (decision.flip_prefill_to_decode) {
      instances_->change_role(request.routing.prefill_name,
                              request.prefill_incarnation_id,
                              InstanceType::DECODE);
    }
  }
  return true;
}

void RoutingState::update_locked(const Request& request, RequestAction action) {
  const auto get_current = [this, action](
                               const std::string& name,
                               const std::string& incarnation) -> Reservation* {
    if (name.empty() || !instances_->contains_instance(name, incarnation)) {
      return nullptr;
    }
    const auto found = reservations_.find(name);
    if (found != reservations_.end() &&
        found->second->incarnation_id == incarnation) {
      return found->second.get();
    }
    if (action != RequestAction::SCHEDULE) {
      return nullptr;
    }
    const auto info = instances_->find_instance(name);
    if (!info.has_value() || info->incarnation_id != incarnation) {
      return nullptr;
    }
    return &reservation_for(*info);
  };
  auto* prefill =
      get_current(request.routing.prefill_name, request.prefill_incarnation_id);
  if (prefill != nullptr) {
    if (action == RequestAction::SCHEDULE) {
      prefill->metrics.estimated_prefill_time += request.estimated_ttft;
    }
    apply_prefill_request_metrics(request, action, prefill->metrics);
  }
  auto* decode =
      get_current(request.routing.decode_name, request.decode_incarnation_id);
  if (decode != nullptr) {
    apply_decode_request_metrics(request, action, decode->metrics);
    if (decode->metrics.decode_request_num == 0) {
      instances_->change_role(request.routing.decode_name,
                              request.decode_incarnation_id,
                              InstanceType::PREFILL);
    }
  }
}

void RoutingState::update_request_metrics(
    const std::shared_ptr<Request>& request,
    RequestAction action) {
  if (!enabled_) {
    return;
  }
  std::lock_guard lock(mutex_);
  update_locked(*request, action);
}

void RoutingState::forget_instance(const std::string& name,
                                   const std::string& incarnation_id) {
  std::lock_guard lock(mutex_);
  const auto found = reservations_.find(name);
  if (found != reservations_.end() &&
      found->second->incarnation_id == incarnation_id) {
    reservations_.erase(found);
  }
}

RequestMetrics RoutingState::request_metrics(const std::string& name) const {
  std::lock_guard lock(mutex_);
  const auto found = reservations_.find(name);
  if (found == reservations_.end() ||
      !instances_->contains_instance(name, found->second->incarnation_id)) {
    return RequestMetrics{};
  }
  return found->second->metrics;
}
}  // namespace xllm_service
