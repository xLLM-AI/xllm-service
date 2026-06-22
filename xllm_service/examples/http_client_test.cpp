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

#include <brpc/channel.h>
#include <gflags/gflags.h>

#include <future>
#include <iostream>

// use stream=false or true
DEFINE_string(prompt,
              "{\"model\": \"Qwen-7B-Chat\", \
    \"prompt\": \"hello, who are you. \", \
    \"max_tokens\": 10, \
    \"temperature\": 0.7, \
    \"stream\": false}",
              "POST this data to the http server");
DEFINE_string(prompt_streaming,
              "{\"model\": \"Qwen-7B-Chat\", \
    \"prompt\": \"hello, who are you. \", \
    \"max_tokens\": 10, \
    \"temperature\": 0.7, \
    \"stream\": true}",
              "POST this data to the http server");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_int32(timeout_ms, 2000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");
DEFINE_string(protocol, "http", "Client-side protocol");
DEFINE_int32(num_threads, 32, "Number of threads to process requests");
DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");
DEFINE_string(url, "http://127.0.0.1:9999/v1/completions", "Server uri.");

namespace brpc {
DECLARE_bool(http_verbose);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto url = FLAGS_url;

  // A Channel represents a communication line to a Server. Notice that
  // Channel is thread-safe and can be shared by all threads in your program.
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = FLAGS_protocol;
  options.timeout_ms = FLAGS_timeout_ms /*milliseconds*/;
  options.max_retry = FLAGS_max_retry;

  // Initialize the channel, NULL means using default options.
  // options, see `brpc/channel.h'.
  if (channel.Init(url.c_str(), FLAGS_load_balancer.c_str(), &options) != 0) {
    std::cerr << "Fail to initialize channel." << std::endl;
    return -1;
  }

  // Test1: send prompt and return non-streaming
  {
    // We will receive response synchronously, safe to put variables
    // on stack.
    brpc::Controller cntl;

    cntl.http_request().uri() = url;
    if (!FLAGS_prompt.empty()) {
      cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
      cntl.request_attachment().append(FLAGS_prompt);
    }

    // Because `done'(last parameter) is NULL, this function waits until
    // the response comes back or error occurs(including timeout).
    channel.CallMethod(NULL, &cntl, NULL, NULL, NULL);
    if (cntl.Failed()) {
      std::cerr << cntl.ErrorText() << std::endl;
      return -1;
    }

    // If -http_verbose is on, brpc already prints the response to stderr.
    if (!brpc::FLAGS_http_verbose) {
      std::cout << cntl.response_attachment() << std::endl;
    }
  }

  // Test2: send prompt and return streaming
  {
    // We will receive response synchronously, safe to put variables
    // on stack.
    brpc::Controller cntl;

    cntl.http_request().uri() = url;
    if (!FLAGS_prompt_streaming.empty()) {
      cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
      cntl.request_attachment().append(FLAGS_prompt_streaming);
    }

    // Because `done'(last parameter) is NULL, this function waits until
    // the response comes back or error occurs(including timeout).
    channel.CallMethod(NULL, &cntl, NULL, NULL, NULL);
    if (cntl.Failed()) {
      std::cerr << cntl.ErrorText() << std::endl;
      return -1;
    }

    // If -http_verbose is on, brpc already prints the response to stderr.
    if (!brpc::FLAGS_http_verbose) {
      std::cout << cntl.response_attachment() << std::endl;
    }
  }

  // Test3: send prompt and return streaming in NEW THREAD.
  brpc::Channel* channel_ptr = &channel;
  auto future_result = std::async(std::launch::async, [channel_ptr]() {
    // We will receive response synchronously, safe to put variables
    // on stack.
    brpc::Controller cntl;

    cntl.http_request().uri() = FLAGS_url.c_str();  // url;
    if (!FLAGS_prompt_streaming.empty()) {
      cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
      cntl.request_attachment().append(FLAGS_prompt_streaming);
    }

    // Because `done'(last parameter) is NULL, this function waits until
    // the response comes back or error occurs(including timeout).
    channel_ptr->CallMethod(NULL, &cntl, NULL, NULL, NULL);
    if (cntl.Failed()) {
      std::cerr << cntl.ErrorText() << std::endl;
      return -1;
    }

    // If -http_verbose is on, brpc already prints the response to stderr.
    if (!brpc::FLAGS_http_verbose) {
      std::cout << cntl.response_attachment() << std::endl;
    }

    return 0;
  });

  future_result.get();

  // while (true) {
  //   sleep(1);
  // }

  return 0;
}
