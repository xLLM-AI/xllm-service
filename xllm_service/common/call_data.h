/* Copyright 2025 The xLLM Authors. All Rights Reserved.
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

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <functional>
#include <string>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm_service {

// Interface for the classes that are used to handle grpc requests.
class CallData {
 public:
  virtual ~CallData() = default;

  // returns true if the rpc is ok and the call data is not finished
  // returns false if the call data is finished and can be deleted
  virtual bool proceed(bool rpc_ok) = 0;

  virtual bool is_disconnected() const = 0;

  void get_x_request_id(std::string& x_request_id, brpc::Controller* ctrl) {
    x_request_id = "";
    if (ctrl->http_request().GetHeader("x-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-request-id");
    } else if (ctrl->http_request().GetHeader("x-ms-client-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-ms-client-request-id");
    }
    return;
  }

  void get_x_request_time(std::string& x_request_time, brpc::Controller* ctrl) {
    x_request_time = "";
    if (ctrl->http_request().GetHeader("x-request-time")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-time");
    } else if (ctrl->http_request().GetHeader("x-request-timems")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-timems");
    }
    return;
  }

 public:
  std::string x_request_id;
  std::string x_request_time;
};

template <typename Request, typename Response>
class StreamCallData : public CallData {
 public:
  StreamCallData(
      brpc::Controller* controller,
      bool stream,
      ::google::protobuf::Closure* done,
      Request* request,
      Response* response,
      std::function<void(const std::string&)> trace_callback = nullptr)
      : controller_(controller),
        done_(done),
        request_(request),
        response_(response),
        trace_callback_(std::move(trace_callback)) {
    stream_ = stream;
    get_x_request_id(x_request_id, controller_);
    get_x_request_time(x_request_time, controller_);

    if (stream_) {
      pa_ = controller_->CreateProgressiveAttachment();

      controller_->http_response().set_content_type("text/event-stream");
      controller_->http_response().set_status_code(200);
      controller_->http_response().SetHeader("Connection", "keep-alive");
      controller_->http_response().SetHeader("Cache-Control", "no-cache");
      // Done Run first for steam response
      done_->Run();

    } else {
      controller_->http_response().SetHeader("Content-Type",
                                             "text/javascript; charset=utf-8");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
  }

  ~StreamCallData() {
    // For non stream response, call brpc done Run
    if (!stream_) {
      done_->Run();
    }
  }

  bool proceed(bool rpc_ok) override { return true; }

  // For non stream response
  bool write_and_finish(const std::string& attachment /*json string*/) {
    if (trace_callback_) trace_callback_(attachment);
    controller_->response_attachment() = attachment;
    return true;
  }

  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(err_msg);
    }

    if (trace_callback_) {
      std::string str;
      controller_->response_attachment().copy_to(&str);
      trace_callback_(str);
    }

    return true;
  }

  // For non stream response
  bool finish_with_error(const std::string& error_message) {
    if (!stream_) {
      controller_->SetFailed(error_message);
    } else {
      io_buf_.clear();
      io_buf_.append(error_message);
      pa_->Write(io_buf_);
    }

    return true;
  }

  // For stream response
  bool write(const butil::IOBuf& attachment_iobuf) {
    if (trace_callback_) {
      std::string str;
      attachment_iobuf.copy_to(&str);
      trace_callback_(str);
    }
    connection_status_ |= pa_->Write(attachment_iobuf);
    return true;
  }

  // For stream response
  bool write(const std::string& attachment) {
    if (trace_callback_) trace_callback_(attachment);
    io_buf_.clear();
    io_buf_.append(attachment);
    connection_status_ |= pa_->Write(io_buf_);
    if (attachment.find("data: [DONE]") != std::string::npos) {
      finished_ = true;
    }

    return true;
  }

  bool write(Response& response) {
    io_buf_.clear();
    io_buf_.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    io_buf_.append("\n\n");

    if (trace_callback_) {
      std::string str;
      io_buf_.copy_to(&str);
      trace_callback_(str);
    }

    connection_status_ |= pa_->Write(io_buf_);
    return true;
  }

  bool finish() { return write("data: [DONE]\n\n"); }

  bool is_disconnected() const override {
    if (stream_) {
      return connection_status_ != 0;
    } else {
      if (controller_) {
        return controller_->IsCanceled();
      }
      return true;
    }
  }

  Request& request() { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }
  bool finished() { return finished_; }

 private:
  brpc::Controller* controller_;
  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool stream_ = false;
  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;
  butil::IOBuf io_buf_;

  bool finished_ = false;
  json2pb::Pb2JsonOptions json_options_;
  std::function<void(const std::string&)> trace_callback_;

  int connection_status_ = 0;
};

using CompletionCallData = StreamCallData<::xllm::proto::CompletionRequest,
                                          ::xllm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<::xllm::proto::ChatRequest, ::xllm::proto::ChatResponse>;

class AnthropicCallData
    : public StreamCallData<::xllm::proto::ChatRequest,
                            ::xllm::proto::AnthropicMessagesResponse> {
 public:
  using Base = StreamCallData<::xllm::proto::ChatRequest,
                              ::xllm::proto::AnthropicMessagesResponse>;
  using Base::Base;
};

}  // namespace xllm_service
