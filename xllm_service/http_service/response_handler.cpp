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

#include "http_service/response_handler.h"

#include <absl/time/clock.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/anthropic_tracer.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/anthropic_stream_encoder.h"
#include "http_service/usage_proto_adapter.h"

namespace xllm_service {
namespace {

// Empty parsed frames are valid while an engine parser buffers a partial
// marker. Never parse raw output as a fallback: that would leak markers or
// parse twice.
std::string validate_chat_frames(const llm::RequestOutput& output,
                                 bool stream,
                                 bool single_choice) {
  if (!output.chat_frames.has_value()) {
    return "Engine output is missing structured chat_output; upgrade xLLM and "
           "configure tool_call_parser/reasoning_parser on the engine";
  }
  if (!stream && output.chat_frames->size() != 1) {
    return "Non-stream chat output requires one complete response frame";
  }
  for (const auto& frame : *output.chat_frames) {
    if (frame.object() !=
        (stream ? "chat.completion.chunk" : "chat.completion")) {
      return "Structured chat output has an unexpected response object";
    }
    for (const auto& choice : frame.choices()) {
      if (!choice.has_index() || (single_choice && choice.index() != 0)) {
        return "Invalid choice index in structured chat output";
      }
      if ((stream && (!choice.has_delta() || choice.has_message())) ||
          (!stream && (!choice.has_message() || choice.has_delta()))) {
        return "Structured chat output has an unexpected message type";
      }
    }
    if (single_choice && frame.choices_size() > 1) {
      return "Anthropic output requires a single choice";
    }
  }
  return "";
}

AnthropicTracer make_anthropic_tracer(
    const std::shared_ptr<AnthropicCallData>& call_data) {
  return AnthropicTracer(
      [call_data](const std::string& formatted) {
        call_data->trace(formatted);
      },
      call_data->request().request_id(),
      call_data->request().service_request_id());
}

}  // namespace

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<ChatCallData> call_data,
    const llm::RequestOutput& output) {
  const std::string error = validate_chat_frames(output,
                                                 /*stream=*/true,
                                                 /*single_choice=*/false);
  if (!error.empty()) {
    call_data->finish_with_error(error);
    return false;
  }
  for (const auto& frame : *output.chat_frames) {
    if (!call_data->write(frame)) {
      return false;
    }
  }
  return output.finished ? call_data->finish() : true;
}

bool ResponseHandler::send_result_to_client(
    std::shared_ptr<ChatCallData> call_data,
    const llm::RequestOutput& output) {
  if (!output.finished) {
    return true;
  }
  const std::string error = validate_chat_frames(output,
                                                 /*stream=*/false,
                                                 /*single_choice=*/false);
  if (!error.empty()) {
    call_data->finish_with_error(error);
    return false;
  }
  return call_data->write_and_finish(output.chat_frames->front());
}

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<CompletionCallData> call_data,
    bool include_usage,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& output) {
  auto& response = call_data->response();
  auto& request_id = output.request_id;

  for (const auto& seq_output : output.outputs) {
    // send chunk with delta message
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);

      // set_logprobs
      if (seq_output.logprobs.has_value() &&
          !seq_output.logprobs.value().empty()) {
        auto* proto_logprobs = choice->mutable_logprobs();
        for (const auto& logprob : seq_output.logprobs.value()) {
          proto_logprobs->add_tokens(logprob.token);
          proto_logprobs->add_token_ids(logprob.token_id);
          proto_logprobs->add_token_logprobs(logprob.logprob);
        }
      }

      if (!call_data->write(response)) {
        return false;
      }
    }

    // send a separate chunk with finish reason
    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call_data->write(response)) {
        return false;
      }
    }
  }

  // send additional chunk for usage statistics
  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    *response.mutable_usage() = to_openai_usage_proto(usage);
    if (!call_data->write(response)) {
      return false;
    }
  }

  if (output.finished) {
    response.Clear();
    // TODO: convert status to grpc status code
    return call_data->finish();
  }
  return true;
}

bool ResponseHandler::send_result_to_client(
    std::shared_ptr<CompletionCallData> call_data,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& req_output) {
  auto& response = call_data->response();
  auto& request_id = req_output.request_id;
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  // add choices into response
  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);

    // set_logprobs
    if (output.logprobs.has_value() && !output.logprobs.value().empty()) {
      auto* proto_logprobs = choice->mutable_logprobs();
      for (const auto& logprob : output.logprobs.value()) {
        proto_logprobs->add_tokens(logprob.token);
        proto_logprobs->add_token_ids(logprob.token_id);
        proto_logprobs->add_token_logprobs(logprob.logprob);
      }
    }

    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    *response.mutable_usage() = to_openai_usage_proto(usage);
  }

  return call_data->write_and_finish(response);
}

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<AnthropicCallData> call_data,
    const llm::RequestOutput& output,
    AnthropicStreamEncoder& encoder) {
  const std::string error = validate_chat_frames(output,
                                                 /*stream=*/true,
                                                 /*single_choice=*/true);
  if (!error.empty()) {
    call_data->finish_with_error(error);
    return false;
  }
  std::vector<std::string> sse_events;
  for (const auto& frame : *output.chat_frames) {
    for (const auto& choice : frame.choices()) {
      const auto& delta = choice.delta();
      auto result =
          encoder.on_reasoning(output, delta.reasoning_content(), &sse_events);
      if (result.ok) {
        result = encoder.on_text(output, delta.content(), &sse_events);
      }
      if (!result.ok) {
        call_data->finish_with_error(result.error);
        return false;
      }
      for (const auto& tool : delta.tool_calls()) {
        result = encoder.on_tool(output,
                                 tool.id(),
                                 tool.function().name(),
                                 tool.function().arguments(),
                                 &sse_events);
        if (!result.ok) {
          call_data->finish_with_error(result.error);
          return false;
        }
      }
    }
  }
  if (output.finished) {
    auto result = encoder.finish(output, &sse_events);
    if (!result.ok) {
      call_data->finish_with_error(result.error);
      return false;
    }
  }
  auto tracer = make_anthropic_tracer(call_data);
  for (const auto& sse : sse_events) {
    tracer.trace("stream_sse", sse);
    if (!call_data->write(sse)) {
      return false;
    }
  }
  return output.finished ? call_data->finish() : true;
}

bool ResponseHandler::send_result_to_client(
    std::shared_ptr<AnthropicCallData> call_data,
    const std::string& model,
    const llm::RequestOutput& output) {
  if (!output.finished) {
    return true;
  }
  const std::string error = validate_chat_frames(output,
                                                 /*stream=*/false,
                                                 /*single_choice=*/true);
  if (!error.empty()) {
    call_data->finish_with_error(error);
    return false;
  }
  const auto& frame = output.chat_frames->front();
  if (frame.choices_size() != 1) {
    call_data->finish_with_error("Anthropic output requires one final choice");
    return false;
  }
  const auto& choice = frame.choices(0);
  const auto& message = choice.message();
  // Adapt structured fields to Anthropic blocks; no model text is parsed here.
  llm::RequestOutput adapted;
  adapted.request_id = output.request_id;
  adapted.usage = output.usage;
  adapted.finished = output.finished;
  llm::SequenceOutput sequence;
  sequence.index = 0;
  sequence.text = message.content();
  if (choice.has_finish_reason()) {
    sequence.finish_reason = choice.finish_reason();
  }
  adapted.outputs.emplace_back(std::move(sequence));
  auto& response = call_data->response();
  auto result =
      fill_anthropic_resp(model, adapted, &response, &message.tool_calls());
  if (!result.ok) {
    call_data->finish_with_error(result.error);
    return false;
  }
  std::optional<std::string> reasoning;
  if (message.has_reasoning_content()) {
    reasoning = message.reasoning_content();
  }
  std::string json;
  std::string json_error;
  if (!anthropic_json(response, reasoning, &json, &json_error)) {
    call_data->finish_with_error(json_error);
    return false;
  }
  make_anthropic_tracer(call_data).trace("non_stream_json_response", json);
  return call_data->write_and_finish(json);
}

void ResponseHandler::bind(std::shared_ptr<ChatCallData> call_data,
                           const std::shared_ptr<Request>& request) {
  request->latest_generate_time = absl::Now();
  request->is_disconnected = [call_data] {
    return call_data->is_disconnected();
  };
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
      return send_delta_to_client(call_data, req_output);
    }
    return send_result_to_client(call_data, req_output);
  };
}

void ResponseHandler::bind(std::shared_ptr<AnthropicCallData> call_data,
                           const std::shared_ptr<Request>& request) {
  request->latest_generate_time = absl::Now();
  auto stream_encoder =
      request->stream ? std::make_shared<AnthropicStreamEncoder>(request->model)
                      : nullptr;
  request->is_disconnected = [call_data] {
    return call_data->is_disconnected();
  };
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
      return send_delta_to_client(call_data, req_output, *stream_encoder);
    }
    return send_result_to_client(call_data, model, req_output);
  };
}

void ResponseHandler::bind(std::shared_ptr<CompletionCallData> call_data,
                           const std::shared_ptr<Request>& request) {
  request->latest_generate_time = absl::Now();

  request->is_disconnected = [call_data] {
    return call_data->is_disconnected();
  };
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
      return send_delta_to_client(
          call_data, include_usage, created_time, model, req_output);
    } else if (!req_output.finished_on_prefill_instance) {
      // for non-stream request, only send final result from decode instance
      return send_result_to_client(call_data, created_time, model, req_output);
    }
    return true;
  };
}

}  // namespace xllm_service
