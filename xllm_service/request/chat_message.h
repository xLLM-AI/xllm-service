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

#pragma once
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
namespace xllm_service {
class Message final {
 public:
  struct FunctionCall {
    std::string name;
    std::string arguments;
  };

  struct ToolCall {
    std::string id;
    std::string type;
    FunctionCall function;
  };

  struct MMUrl {
    std::string url;
  };

  class MMContent final {
   public:
    explicit MMContent(std::string type) : type(std::move(type)) {}
    MMContent(std::string type, std::string text)
        : type(std::move(type)), text(std::move(text)) {}

    std::string type;

    std::string text;
    std::string thinking;
    std::string signature;
    std::string data;
    std::string id;
    std::string name;
    nlohmann::ordered_json input = nlohmann::ordered_json::object();
    MMUrl image_url;  // image place holder

    MMUrl video_url;  // video place holder
    MMUrl audio_url;  // audio place holder
  };

  using MMContentVec = std::vector<MMContent>;
  using ToolCallVec = std::vector<ToolCall>;
  using Content = std::variant<std::string, MMContentVec>;

  Message() = default;
  Message(std::string role, std::string content)
      : role(std::move(role)), content(std::move(content)) {}

  Message(std::string role, MMContentVec content)
      : role(std::move(role)), content(std::move(content)) {}

  std::string role;
  Content content;
  std::optional<std::string> reasoning_content;
  std::optional<ToolCallVec> tool_calls;
  std::string tool_call_id;
};

}  // namespace xllm_service
