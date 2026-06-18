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

#include "scheduler/scheduler.h"

#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/anthropic_stream_encoder.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "scheduler/xllm_chat_parse_bridge.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options) : options_(options) {
  // vLLM-backend clusters forward the raw client JSON to vLLM, which does its
  // own tokenization / chat templating. Skip building the local tokenizer and
  // chat template so the master does not require a tokenizer.model / template.
  if (options_.default_backend_type() != "vllm") {
    tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                    &tokenizer_args_);
    chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);
  }

  const std::string etcd_username =
      utils::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      utils::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(FATAL) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                etcd_username,
                                                etcd_password,
                                                options_.etcd_namespace());
  } else {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                options_.etcd_namespace());
  }

  if (!register_current_service()) {
    LOG(FATAL)
        << "Failed to register current xllm_service in etcd, service_name: "
        << options_.service_name();
  }

  auto handle_xservice = std::bind(&Scheduler::handle_xservice_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
  etcd_client_->add_watch(ETCD_XSERVICE_KEY_PREFIX, handle_xservice);

  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, this);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }
}

Scheduler::~Scheduler() { etcd_client_->stop_watch(); }

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  // For vLLM backend clusters we forward the client's raw OpenAI JSON straight
  // through to vLLM, which applies its own chat template and tokenization. So
  // skip the local chat-template + tokenize steps entirely (leaving token_ids
  // empty) and only run instance selection. See default_backend_type option.
  const bool is_vllm = (options_.default_backend_type() == "vllm");

  // apply chat template
  if (!is_vllm && request->messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request->tool_choice == "none" ? empty_tools : request->tools;
    auto prompt = chat_template_->apply(
        request->messages, tools_for_template, request->chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return false;
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (!is_vllm && request->prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(request->prompt, &request->token_ids)) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return false;
    }
  }

  auto ret = lb_policy_->select_instances_pair(request);
  if (!ret) {
    return false;
  }

  if (!instance_mgr_->bind_request_instance_incarnations(request)) {
    LOG(ERROR) << "Failed to bind request to instance incarnation ids. "
               << request->routing.debug_string();
    return false;
  }
  DLOG(INFO) << request->routing.debug_string();

  // update request metrics
  if (!is_vllm && request->prompt.size() != 0) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  return true;
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    global_kvcache_mgr_->upload_kvcache();

    instance_mgr_->upload_load_metrics();
  }
}

bool Scheduler::register_current_service() {
  const std::string service_key =
      ETCD_XSERVICE_KEY_PREFIX + options_.service_name();

  if (etcd_client_->set(
          service_key, options_.service_name(), kHeartbeatInterval)) {
    return true;
  }

  LOG(ERROR) << "Service key already exists, registration failed: "
             << service_key
             << ". Please ensure service_name is unique across xllm_service "
                "instances.";
  return false;
}

bool Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return false;
  }
  if (!instance_mgr_->record_instance_heartbeat(req->name(),
                                                req->incarnation_id())) {
    return false;
  }
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
  instance_mgr_->update_latency_metrics(req->name(), req->latency_metrics());
  return true;
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
}

void Scheduler::handle_xservice_watch(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() != etcd::Event::EventType::DELETE_) {
      continue;
    }

    std::string deleted_service;
    if (event.has_prev_kv()) {
      deleted_service = event.prev_kv().key().substr(prefix_len);
    } else if (event.has_kv()) {
      deleted_service = event.kv().key().substr(prefix_len);
    }

    if (deleted_service.empty()) {
      continue;
    }

    if (deleted_service == options_.service_name()) {
      LOG(INFO) << "Current xllm_service registration expired, re-registering";
      register_current_service();
      continue;
    }

    if (deleted_service == ETCD_MASTER_SERVICE_NAME) {
      continue;
    }

    if (!is_master_service_) {
      continue;
    }

    LOG(INFO) << "Detected xllm_service offline: " << deleted_service;
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

std::vector<std::string> Scheduler::get_static_prefill_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_prefill_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }

    request->latest_generate_time = absl::Now();
    auto tools_for_parse =
        (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                        : request->tools);
    auto tool_call_parser_pref = options_.tool_call_parser();
    auto reasoning_parser_pref = options_.reasoning_parser();
    const auto parser_formats = resolve_chat_parser_formats_with_xllm(
        request->model, tool_call_parser_pref, reasoning_parser_pref);
    const bool force_reasoning = get_enable_thinking_from_request(
        request->chat_template_kwargs, parser_formats.reasoning_parser);
    std::shared_ptr<ChatStreamParseState> stream_state;
    if (request->stream) {
      stream_state = response_handler_.create_chat_stream_parse_state(
          tools_for_parse,
          request->model,
          tool_call_parser_pref,
          reasoning_parser_pref,
          force_reasoning);
    }

    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         tools = std::move(tools_for_parse),
         tool_call_parser = std::move(tool_call_parser_pref),
         reasoning_parser = std::move(reasoning_parser_pref),
         force_reasoning,
         stream_state = std::move(stream_state),
         service_request_id = request->service_request_id,
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      include_usage,
                                                      created_time,
                                                      model,
                                                      req_output,
                                                      stream_state);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        return response_handler_.send_result_to_client(call_data,
                                                       created_time,
                                                       model,
                                                       req_output,
                                                       tools,
                                                       tool_call_parser,
                                                       reasoning_parser,
                                                       force_reasoning);
      }
      return true;
    };
    requests_.emplace(request->service_request_id, request);
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->service_request_id] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

bool Scheduler::record_new_request(std::shared_ptr<AnthropicCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }

    request->latest_generate_time = absl::Now();
    auto tools_for_parse =
        (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                        : request->tools);
    auto tool_call_parser_pref = options_.tool_call_parser();
    auto reasoning_parser_pref = options_.reasoning_parser();
    const auto parser_formats = resolve_chat_parser_formats_with_xllm(
        request->model, tool_call_parser_pref, reasoning_parser_pref);
    const bool force_reasoning = get_enable_thinking_from_request(
        request->chat_template_kwargs, parser_formats.reasoning_parser);
    auto stream_encoder =
        request->stream
            ? std::make_shared<AnthropicStreamEncoder>(request->model)
            : nullptr;
    auto stream_parser =
        request->stream
            ? create_stream_output_parser_with_xllm(tools_for_parse,
                                                    request->model,
                                                    tool_call_parser_pref,
                                                    reasoning_parser_pref,
                                                    force_reasoning)
            : nullptr;
    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         stream_encoder,
         stream_parser,
         tools = std::move(tools_for_parse),
         tool_call_parser = std::move(tool_call_parser_pref),
         reasoning_parser = std::move(reasoning_parser_pref),
         force_reasoning](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(
            call_data, model, req_output, *stream_encoder, stream_parser);
      } else if (!req_output.finished_on_prefill_instance) {
        return response_handler_.send_result_to_client(call_data,
                                                       model,
                                                       req_output,
                                                       tools,
                                                       tool_call_parser,
                                                       reasoning_parser,
                                                       force_reasoning);
      }
      return true;
    };
    requests_.emplace(request->service_request_id, request);
    COUNTER_INC(server_request_in_total);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->service_request_id] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }

    request->latest_generate_time = absl::Now();

    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         service_request_id = request->service_request_id,
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(
            call_data, include_usage, created_time, model, req_output);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        return response_handler_.send_result_to_client(
            call_data, created_time, model, req_output);
      }
      return true;
    };
    requests_.emplace(request->service_request_id, request);
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->service_request_id] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

void Scheduler::finish_request(const std::string& service_request_id,
                               bool error) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it != requests_.end()) {
      request = it->second;
      requests_.erase(it);
    }
  }

  if (request != nullptr) {
    if (error) {
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    } else {
      instance_mgr_->update_request_metrics(request,
                                            RequestAction::FINISH_DECODE);
    }
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::string> cleared_request_ids;
  std::lock_guard<std::mutex> lock(request_mutex_);
  for (auto it = requests_.begin(); it != requests_.end();) {
    const bool clear_prefill =
        ((type == InstanceType::DEFAULT || type == InstanceType::PREFILL) &&
         it->second->routing.prefill_name == instance_name &&
         it->second->prefill_incarnation_id == incarnation_id &&
         !it->second->prefill_stage_finished);
    const bool clear_decode =
        (type == InstanceType::DECODE &&
         it->second->routing.decode_name == instance_name &&
         it->second->decode_incarnation_id == incarnation_id);
    if (clear_prefill || clear_decode) {
      auto service_request_id = it->second->service_request_id;
      llm::RequestOutput req_output;
      req_output.status = llm::Status(llm::StatusCode::CANCELLED,
                                      "Instance is failed and deleted");
      // call request callback
      it->second->output_callback(req_output);
      LOG(INFO) << "Clear request on failed instance: " << instance_name
                << ", incarnation_id: " << incarnation_id
                << ", service_request_id: " << service_request_id;
      cleared_request_ids.emplace_back(service_request_id);
      it = requests_.erase(it);
    } else {
      ++it;
    }
  }

  if (!cleared_request_ids.empty()) {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    for (const auto& service_request_id : cleared_request_ids) {
      remote_requests_output_thread_map_.erase(service_request_id);
    }
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  bool finished_on_prefill_instance =
      request_output.finished_on_prefill_instance;
  const std::string& service_request_id = request_output.service_request_id;
  bool status_error =
      request_output.status.has_value() && !request_output.status.value().ok();

  OutputCallback cb;
  std::shared_ptr<Request> request;
  bool client_disconnected = false;
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
    cb = request->output_callback;

    // check client connection
    if (request->call_data->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      requests_.erase(it);
      client_disconnected = true;
    }
  }

  if (client_disconnected) {
    instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
    return false;
  }

  if (!status_error) {
    // no error, update instance request metrics
    update_request_metrics(request, finished_on_prefill_instance);
    update_token_latency_metrics(request, finished_on_prefill_instance);
  }

  size_t req_thread_idx = -1;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    auto it = remote_requests_output_thread_map_.find(service_request_id);
    if (it == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Can not found the thread for the received request output, "
                    "request id is: "
                 << service_request_id;
      return false;
    }
    req_thread_idx = it->second;
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       service_request_id,
       cb,
       status_error,
       request_output = std::move(request_output)]() mutable {
        if (!cb(request_output) || status_error) {
          finish_request(service_request_id, true);
          return;
        }
        if (request_output.finished) {
          finish_request(service_request_id);
          return;
        }
      });

  return true;
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    request->prefill_stage_finished = true;
    // update instance request metrics for prefill finished request
    instance_mgr_->update_request_metrics(request,
                                          RequestAction::FINISH_PREFILL);
  } else {
    // update instance request metrics
    instance_mgr_->update_request_metrics(request, RequestAction::GENERATE);
  }
}

void Scheduler::update_token_latency_metrics(
    std::shared_ptr<Request> request,
    bool finished_on_prefill_instance) {
  int64_t tbt_milliseconds =
      absl::ToInt64Milliseconds(absl::Now() - request->latest_generate_time);
  request->latest_generate_time = absl::Now();
  if (finished_on_prefill_instance) {
    HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                      tbt_milliseconds);
  } else {
    HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
  }
}

bool Scheduler::has_available_instances() const {
  return instance_mgr_->has_available_instances();
}

}  // namespace xllm_service
