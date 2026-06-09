/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "http_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <brpc/progressive_reader.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <functional>
#include <nlohmann/json.hpp>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/closure_guard.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "common/xllm/uuid.h"
#include "completion.pb.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/chat_json_parser.h"
#include "scheduler/scheduler.h"
#include "xllm_service.pb.h"

namespace xllm_service {

namespace {
thread_local llm::ShortUUID short_uuid;
std::string generate_service_request_id(const std::string& method) {
  std::stringstream ss;
  ss << method << "-";
  ss << std::this_thread::get_id();
  ss << "-";
  ss << short_uuid.random();
  return ss.str();
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value);

nlohmann::json proto_struct_to_json(const google::protobuf::Struct& pb_struct) {
  nlohmann::json result = nlohmann::json::object();
  for (const auto& field : pb_struct.fields()) {
    result[field.first] = proto_value_to_json(field.second);
  }
  return result;
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value) {
  switch (pb_value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nlohmann::json(nullptr);
    case google::protobuf::Value::kNumberValue:
      return nlohmann::json(pb_value.number_value());
    case google::protobuf::Value::kStringValue:
      return nlohmann::json(pb_value.string_value());
    case google::protobuf::Value::kBoolValue:
      return nlohmann::json(pb_value.bool_value());
    case google::protobuf::Value::kStructValue:
      return proto_struct_to_json(pb_value.struct_value());
    case google::protobuf::Value::kListValue: {
      nlohmann::json result = nlohmann::json::array();
      for (const auto& item : pb_value.list_value().values()) {
        result.push_back(proto_value_to_json(item));
      }
      return result;
    }
    case google::protobuf::Value::KIND_NOT_SET:
    default:
      return nlohmann::json(nullptr);
  }
}
std::vector<JsonTool> parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<::xllm::proto::Tool>&
        proto_tools) {
  std::vector<JsonTool> tools;
  tools.reserve(proto_tools.size());

  for (const auto& proto_tool : proto_tools) {
    JsonTool json_tool;
    json_tool.type = proto_tool.type();
    json_tool.function.name = proto_tool.function().name();
    json_tool.function.description = proto_tool.function().description();
    if (proto_tool.function().has_parameters()) {
      json_tool.function.parameters =
          proto_struct_to_json(proto_tool.function().parameters());
    } else {
      json_tool.function.parameters = nlohmann::json::object();
    }
    tools.emplace_back(std::move(json_tool));
  }
  return tools;
}
}  // namespace

XllmHttpServiceImpl::XllmHttpServiceImpl(const Options& options,
                                         Scheduler* scheduler)
    : options_(options), scheduler_(scheduler) {
  initialized_ = true;
  thread_pool_ = std::make_unique<ThreadPool>(options_.num_threads());
  request_tracer_ =
      std::make_unique<RequestTracer>(options_.enable_request_trace());
}

XllmHttpServiceImpl::~XllmHttpServiceImpl() {}

void XllmHttpServiceImpl::Hello(::google::protobuf::RpcController* controller,
                                const proto::HttpHelloRequest* request,
                                proto::HttpHelloResponse* response,
                                ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  LOG(INFO) << "Get request: " << request->ping();

  response->set_pong(request->ping());
}

namespace {
template <typename T>
void handle_non_stream_response(brpc::Controller* cntl,
                                std::shared_ptr<T> call_data) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    call_data->finish_with_error(cntl->ErrorText());
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    return;
  }
  call_data->write_and_finish(cntl->response_attachment().to_string());
}

// fire and forget
template <typename T>
void handle_first_send_request(brpc::Controller* cntl,
                               std::shared_ptr<T> call_data,
                               Scheduler* scheduler,
                               std::string service_request_id,
                               bool stream) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    call_data->finish_with_error(cntl->ErrorText());
    scheduler->finish_request(service_request_id, /*error*/ true);
    return;
  }
}

template <typename T>
class CustomProgressiveReader : public brpc::ProgressiveReader {
 public:
  explicit CustomProgressiveReader(brpc::Controller* redirect_cntl,
                                   std::shared_ptr<T> call_data)
      : redirect_cntl_(redirect_cntl), call_data_(call_data) {}

  virtual ~CustomProgressiveReader() { delete redirect_cntl_; }

  // Called when one part was read.
  // Error returned is treated as *permanent* and the socket where the
  // data was read will be closed.
  // A temporary error may be handled by blocking this function, which
  // may block the HTTP parsing on the socket.
  virtual butil::Status OnReadOnePart(const void* data, size_t length) {
    call_data_->write(std::string((char*)data, length));
    return butil::Status::OK();
  }

  // Called when there's nothing to read anymore. The `status' is a hint for
  // why this method is called.
  // - status.ok(): the message is complete and successfully consumed.
  // - otherwise: socket was broken or OnReadOnePart() failed.
  // This method will be called once and only once. No other methods will
  // be called after. User can release the memory of this object inside.
  virtual void OnEndOfMessage(const butil::Status& status) { delete this; }

 private:
  brpc::Controller* redirect_cntl_ = nullptr;
  std::shared_ptr<T> call_data_;
};
}  // namespace

namespace {

constexpr char kInferContentLength[] = "Infer-Content-Length";
constexpr char kContentLength[] = "Content-Length";

size_t GetJsonContentLength(const brpc::Controller* ctrl) {
  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  if (infer_content_len != nullptr) {
    return std::stoul(*infer_content_len);
  }

  const auto content_len = ctrl->http_request().GetHeader(kContentLength);
  if (content_len != nullptr) {
    return std::stoul(*content_len);
  }

  LOG(FATAL) << "Content-Length header is missing.";
  return (size_t)-1L;
}

}  // namespace

template <typename T>
void XllmHttpServiceImpl::handle(std::shared_ptr<T> call_data,
                                 std::shared_ptr<Request> request) {
  // record request
  auto& req_pb = call_data->request();
  bool success = scheduler_->record_new_request(call_data, request);
  if (!success) {
    LOG(ERROR) << "rpc service add new request error: "
               << request->service_request_id;
    call_data->finish_with_error("Internal runtime error.");
    return;
  }

  // async redistribute the request and wait the response
  // TODO: optimize the thread pool to async mode.
  auto& target_uri = request->routing.prefill_name;
  brpc::Channel* channel_ptr = scheduler_->get_channel(target_uri).get();
  // use stub
  xllm::proto::XllmAPIService_Stub stub(channel_ptr);
  // xllm::proto::Status* resp_pb = new xllm::proto::Status();
  brpc::Controller* redirect_cntl = new brpc::Controller();
  google::protobuf::Closure* done =
      brpc::NewCallback(&handle_first_send_request<T>,
                        redirect_cntl,
                        call_data,
                        scheduler_,
                        request->service_request_id,
                        request->stream);

  if constexpr (std::is_same_v<T, CompletionCallData>) {
    stub.Completions(redirect_cntl, &req_pb, nullptr, done);
  } else if constexpr (std::is_same_v<T, ChatCallData>) {
    stub.ChatCompletions(redirect_cntl, &req_pb, nullptr, done);
  } else if constexpr (std::is_same_v<T, AnthropicCallData>) {
    stub.ChatCompletions(redirect_cntl, &req_pb, nullptr, done);
  } else {
    delete redirect_cntl;
    delete done;
    LOG(ERROR) << "Unknown call_data type";
  }
}

template <typename T>
std::shared_ptr<Request> XllmHttpServiceImpl::generate_request(
    T* req_pb,
    const std::string& method) {
  auto request = std::make_shared<Request>();
  request->model = req_pb->model();

  // TODO: add `created_time` fileds etc.
  // create xllm_service request_id: service_request_id
  request->service_request_id = generate_service_request_id(method);

  if (req_pb->has_stream()) {
    request->stream = req_pb->stream();
  }

  if (req_pb->has_stream_options()) {
    request->include_usage = req_pb->stream_options().include_usage();
  }

  if (options_.enable_request_trace()) {
    request->trace_callback =
        [this, service_request_id = request->service_request_id](
            const std::string& message) {
          request_tracer_->log(service_request_id, message);
        };
  }

  return request;
}

namespace {
void handle_get_model_response(brpc::Controller* cntl,
                               std::shared_ptr<CompletionCallData> call_data,
                               google::protobuf::Closure* done,
                               xllm::proto::ModelListResponse* resp_pb) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<xllm::proto::ModelListResponse> resp_pb_guard(resp_pb);

  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    call_data->finish_with_error(cntl->ErrorText());
    return;
  }
  std::string err_msg;
  std::string json_output;
  if (!json2pb::ProtoMessageToJson(*resp_pb, &json_output, &err_msg)) {
    call_data->finish_with_error(err_msg);
    LOG(ERROR) << "ProtoMessageToJson failed: " << err_msg;
    return;
  }
  LOG(INFO) << "ProtoMessageToJson: " << json_output;
  call_data->write_and_finish(json_output);
}
}  // namespace

void XllmHttpServiceImpl::get_serving_models(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ModelListRequest>(
          arena);

  // auto call_data = std::make_shared<StreamCallData>(cntl, false,
  // done_guard.release());
  auto call_data = std::make_shared<CompletionCallData>(
      cntl, false, done_guard.release(), nullptr, nullptr);

  auto service_request = std::make_shared<Request>();
  if (!scheduler_->schedule(service_request)) {
    cntl->SetFailed("Schedule request failed!");
    LOG(ERROR) << "Schedule request failed!";
    return;
  }

  brpc::Channel* channel_ptr =
      scheduler_->get_channel(service_request->routing.prefill_name).get();

  xllm::proto::XllmAPIService_Stub stub(channel_ptr);
  brpc::Controller* redirect_cntl = new brpc::Controller();
  auto* resp_pb = new xllm::proto::ModelListResponse();
  google::protobuf::Closure* done_callback = brpc::NewCallback(
      &handle_get_model_response, redirect_cntl, call_data, done, resp_pb);
  stub.Models(redirect_cntl, req_pb, resp_pb, done_callback);
}

void XllmHttpServiceImpl::Completions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionResponse>(
          arena);

  std::string attachment = std::move(cntl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    cntl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  auto service_request = generate_request(req_pb, "/v1/completions");

  if (!req_pb->prompt().empty()) {
    service_request->prompt = req_pb->prompt();
    // select instance for request
    if (!scheduler_->schedule(service_request)) {
      cntl->SetFailed("Schedule request failed!");
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Prompt is empty!");
    LOG(ERROR) << "Prompt is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);

  auto call_data = std::make_shared<CompletionCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::ChatCompletions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatResponse>(
          arena);

  auto content_len = GetJsonContentLength(cntl);
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  auto chat_json = normalize_chat_json(std::move(attachment));
  if (!chat_json.ok) {
    cntl->SetFailed(chat_json.error);
    LOG(ERROR) << "normalize chat json failed: " << chat_json.error;
    return;
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      chat_json.json, req_pb, options);
  if (!status.ok()) {
    cntl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto service_request = generate_request(req_pb, "/v1/chat/completions");

  if (req_pb->messages_size() > 0) {
    service_request->messages.reserve(req_pb->messages_size());
    for (const auto& message : req_pb->messages()) {
      service_request->messages.emplace_back(message.role(), message.content());
    }
    if (req_pb->has_chat_template_kwargs()) {
      service_request->chat_template_kwargs =
          proto_struct_to_json(req_pb->chat_template_kwargs());
    }
    service_request->tools = parse_tools_from_proto(req_pb->tools());
    if (req_pb->has_tool_choice()) {
      service_request->tool_choice = req_pb->tool_choice();
    }

    if (!scheduler_->schedule(service_request)) {
      cntl->SetFailed("Schedule request failed!");
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Messages is empty!");
    LOG(ERROR) << "Messages is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);

  auto call_data = std::make_shared<ChatCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::AnthropicMessages(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  auto arena = response->GetArena();
  auto anthropic_req_pb = google::protobuf::Arena::CreateMessage<
      ::xllm::proto::AnthropicMessagesRequest>(arena);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<
      ::xllm::proto::AnthropicMessagesResponse>(arena);

  auto content_len = GetJsonContentLength(cntl);
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  auto parse_result =
      parse_anthropic_json(std::move(attachment), anthropic_req_pb);
  if (!parse_result.ok) {
    cntl->SetFailed(parse_result.error);
    LOG(ERROR) << "parse anthropic json failed: " << parse_result.error;
    return;
  }

  ChatMessages messages;
  auto adapt_result = fill_chat_req(*anthropic_req_pb, req_pb, &messages);
  if (!adapt_result.ok) {
    cntl->SetFailed(adapt_result.error);
    LOG(ERROR) << "adapt anthropic request failed: " << adapt_result.error;
    return;
  }
  req_pb->set_request_id(new_anthropic_id());

  auto service_request = generate_request(req_pb, "/v1/messages");
  service_request->messages = std::move(messages);
  service_request->tools = parse_tools_from_proto(req_pb->tools());
  if (req_pb->has_tool_choice()) {
    service_request->tool_choice = req_pb->tool_choice();
  }

  if (!scheduler_->schedule(service_request)) {
    cntl->SetFailed("Schedule request failed!");
    LOG(ERROR) << "Schedule request failed!";
    return;
  }

  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);

  auto call_data = std::make_shared<AnthropicCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  if (!call_data->x_request_id.empty()) {
    req_pb->set_x_request_id(call_data->x_request_id);
  }
  if (!call_data->x_request_time.empty()) {
    req_pb->set_x_request_time(call_data->x_request_time);
  }
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::Embeddings(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  cntl->SetFailed("not support Embeddings");
  return;
}

void XllmHttpServiceImpl::Models(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  get_serving_models(controller, request, response, done);
}

void XllmHttpServiceImpl::Metrics(::google::protobuf::RpcController* controller,
                                  const proto::HttpRequest* request,
                                  proto::HttpResponse* response,
                                  ::google::protobuf::Closure* done) {
  ClosureGuard done_guard(done);
  // TODO: implement metrics endpoint
}

}  // namespace xllm_service
