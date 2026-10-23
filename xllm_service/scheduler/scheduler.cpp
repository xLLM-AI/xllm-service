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

#include "scheduler/scheduler.h"

#include <utility>

#include "chat_template/deepseek_v4_cpp_chat_template.h"
#include "chat_template/model_type.h"
#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/anthropic_stream_encoder.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "scheduler/loadbalance_policy/ds_load_balance.h"
#include "scheduler/request_metrics.h"
#include "scheduler/request_output.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds
constexpr int32_t kRegistrationMaxRetries = 5;

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options) : options_(options) {
  if (!options_.kv_events_config_file().empty()) {
    KVDirectoryConfig config;
    std::string error;
    CHECK(load_kv_directory_config(
        options_.kv_events_config_file(), &config, &error))
        << error;
    kv_directory_ = std::make_unique<KVDirectory>(std::move(config));
  }
  std::optional<std::string> model_type;

  // vLLM-backend clusters forward the raw client JSON to vLLM, which does its
  // own tokenization / chat templating. Skip building the local tokenizer and
  // chat template so the master does not require a tokenizer.model / template.
  if (options_.default_backend_type() != "vllm") {
    CHECK_GT(options_.num_threads(), 0);
    tokenizer_pool_ = std::make_unique<TokenizerPool>(
        TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                           &tokenizer_args_),
        static_cast<size_t>(options_.num_threads()));

    // Select the chat template by config.json model_type (jinja by default).
    model_type = load_model_type(options_.tokenizer_path());
    switch (select_chat_template_kind(model_type)) {
      case ChatTemplateKind::kDeepseekV4Cpp:
        chat_template_ =
            std::make_unique<DeepseekV4CppChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected DeepseekV4CppChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
      case ChatTemplateKind::kJinja:
        chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected JinjaChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
    }
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
  start_kv_event_discovery();

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_,
                                            kv_directory_.get(),
                                            options.car_host_hit_weight(),
                                            options.car_prefill_lambda());
  } else if (options.load_balance_policy() == "ds_load_balance") {
    CHECK_NE(options.default_backend_type(), "vllm")
        << "ds_load_balance requires service-side tokenization";
    lb_policy_ = std::make_unique<DsLoadBalance>(
        instance_mgr_,
        create_mooncake_prefix_lookup(options.mooncake_prefix_config_file()));
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

Scheduler::~Scheduler() {
  if (kv_directory_) {
    kv_directory_->stop();
  }
  if (etcd_client_) {
    etcd_client_->stop_watch();
  }
}

void Scheduler::start_kv_event_discovery() {
  if (!kv_directory_) {
    return;
  }
  // The directory joins its discovery thread before the manager is destroyed.
  kv_directory_->start([manager = instance_mgr_] {
    const auto snapshot = manager->get_instance_snapshot();
    std::vector<KVEventInstance> result;
    result.reserve(snapshot.size());
    for (const auto& instance : snapshot) {
      if (instance.backend_type != "xllm" ||
          instance.runtime_state == InstanceRuntimeState::SUSPECT ||
          instance.type == InstanceType::DECODE ||
          (instance.type == InstanceType::MIX &&
           instance.current_type != InstanceType::PREFILL)) {
        continue;
      }
      result.emplace_back(KVEventInstance{instance.name,
                                          instance.incarnation_id,
                                          instance.register_ts_ms,
                                          instance.dp_size});
    }
    return result;
  });
}

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
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }
    if (!tokenizer_pool_->encode(request->prompt,
                                 &request->token_ids,
                                 chat_template_->encode_add_special_tokens())) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return false;
    }
  }

  auto ret = lb_policy_->select_instances_pair(request);
  if (!ret) {
    return false;
  }

  if (!request->ds_load_balance_reserved &&
      !instance_mgr_->bind_request_instance_incarnations(request)) {
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

  // Key already exists — likely a stale lease from a previous process.
  // Since the lease TTL is short (3 seconds), we can safely wait for it
  // to expire naturally and retry registration without deleting the key.
  LOG(WARNING) << "Service key already exists: " << service_key
               << ". Waiting for the stale lease to expire and retrying.";

  for (int attempt = 1; attempt <= kRegistrationMaxRetries; ++attempt) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (etcd_client_->set(
            service_key, options_.service_name(), kHeartbeatInterval)) {
      LOG(INFO) << "Service registered successfully on retry " << attempt;
      return true;
    }
    LOG(WARNING) << "Registration retry " << attempt << " failed, "
                 << "waiting for old lease to expire...";
  }

  LOG(ERROR) << "Registration failed after " << kRegistrationMaxRetries
             << " retries: " << service_key;
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

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  request->call_data = call_data;
  request->output_callback =
      [this, call_data, stream = request->stream](
          const llm::RequestOutput& req_output) mutable -> bool {
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        return call_data->finish_with_error(status);
      }
    }

    if (req_output.finished_on_prefill_instance) {
      return true;
    }
    if (stream) {
      return response_handler_.send_delta_to_client(call_data, req_output);
    }
    return response_handler_.send_result_to_client(call_data, req_output);
  };
  return publish_request(std::move(request));
}

bool Scheduler::record_new_request(std::shared_ptr<AnthropicCallData> call_data,
                                   std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  auto stream_encoder =
      request->stream ? std::make_shared<AnthropicStreamEncoder>(request->model)
                      : nullptr;
  request->call_data = call_data;
  request->output_callback =
      [this,
       call_data,
       model = request->model,
       stream = request->stream,
       stream_encoder](const llm::RequestOutput& req_output) mutable -> bool {
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        return call_data->finish_with_error(status);
      }
    }

    if (req_output.finished_on_prefill_instance) {
      return true;
    }
    if (stream) {
      return response_handler_.send_delta_to_client(
          call_data, req_output, *stream_encoder);
    }
    return response_handler_.send_result_to_client(
        call_data, model, req_output);
  };
  return publish_request(std::move(request));
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
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
        return call_data->finish_with_error(status);
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
  return publish_request(std::move(request));
}

bool Scheduler::publish_request(std::shared_ptr<Request> request) {
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  std::lock_guard<std::mutex> request_lock(request_mutex_);
  const auto existing = requests_.find(request->service_request_id);
  if (existing != requests_.end()) {
    if (existing->second != request) {
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    }
    LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                  "are not allowed. "
               << request->service_request_id;
    return false;
  }
  // schedule() precedes callback registration. The selected instance may have
  // been removed or replaced in that interval, before cleanup could see us.
  if (!instance_mgr_->request_instances_match(*request)) {
    instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    LOG(ERROR) << "Instance changed before request admission: "
               << request->service_request_id;
    return false;
  }
  remote_requests_output_thread_map_.emplace(request->service_request_id,
                                             next_thread_idx);
  next_thread_idx = (next_thread_idx + 1) % kOutputTheadNum_;
  requests_.emplace(request->service_request_id, request);
  COUNTER_INC(server_request_in_total);
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
      instance_mgr_->update_request_metrics(
          request,
          error ? RequestAction::CANCEL : RequestAction::FINISH_DECODE);
      requests_.erase(it);
    }
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::shared_ptr<Request>> cleared_requests;
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    for (const auto& [service_request_id, request] : requests_) {
      const bool clear_prefill =
          ((type == InstanceType::DEFAULT || type == InstanceType::PREFILL) &&
           request->routing.prefill_name == instance_name &&
           request->prefill_incarnation_id == incarnation_id &&
           !request->prefill_stage_finished);
      const bool clear_decode =
          (type == InstanceType::DECODE &&
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

bool Scheduler::handle_generation(llm::RequestOutput request_output) {
  std::string service_request_id = request_output.service_request_id;
  const bool status_error =
      request_output.status.has_value() && !request_output.status.value().ok();

  std::shared_ptr<Request> request;
  size_t req_thread_idx = 0;
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
    if (request->call_data->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      requests_.erase(it);
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
      remote_requests_output_thread_map_.erase(service_request_id);
      return false;
    } else if (!status_error) {
      const int64_t previous_tokens = request->num_generated_tokens;
      const bool first_token = update_request_metrics(request, request_output);
      if (first_token || request->num_generated_tokens > previous_tokens) {
        update_token_latency_metrics(request, first_token);
      }
    }
    auto thread_it =
        remote_requests_output_thread_map_.find(service_request_id);
    if (thread_it == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Can not found the thread for the received request output, "
                    "request id is: "
                 << service_request_id;
      return false;
    }
    req_thread_idx = thread_it->second;
  }

  output_threadpools_[req_thread_idx].schedule(
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

  return true;
}

bool Scheduler::update_request_metrics(const std::shared_ptr<Request>& request,
                                       const llm::RequestOutput& output) {
  const GenerationProgress progress = record_generation(*request, output);
  instance_mgr_->update_request_metrics(request,
                                        progress.finished_prefill
                                            ? RequestAction::FINISH_PREFILL
                                            : RequestAction::GENERATE,
                                        progress.generated_tokens);
  return progress.finished_prefill;
}

void Scheduler::update_token_latency_metrics(
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

bool Scheduler::has_available_instances() const {
  return instance_mgr_->has_available_instances();
}

}  // namespace xllm_service
