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

#include "http_service/chat_json_parser.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

namespace xllm_service {
namespace {

constexpr double kDefaultTemperature = 1.0;

ChatJsonResult json_error(const std::string& error) {
  return ChatJsonResult{false, "", error};
}

ChatJsonResult normalized_json_result(nlohmann::json&& json,
                                      std::string&& original,
                                      bool modified) {
  if (!modified) {
    return ChatJsonResult{true, std::move(original), ""};
  }
  return ChatJsonResult{true, json.dump(), ""};
}

std::pair<bool, std::string> text_content(const nlohmann::json& content,
                                          std::string* text) {
  size_t total_size = 0;
  for (const auto& item : content) {
    if (!item.is_object()) {
      return {false, "Content array item must be an object."};
    }
    if (!item.contains("type") || item["type"] != "text") {
      return {false,
              "Only text content arrays are supported by this chat endpoint."};
    }
    if (!item.contains("text") || !item["text"].is_string()) {
      return {false, "Missing or invalid 'text' field in content item."};
    }
    total_size += item["text"].get_ref<const std::string&>().size();
  }

  if (!content.empty()) {
    total_size += content.size() - 1;
  }
  text->clear();
  text->reserve(total_size);

  bool first = true;
  for (const auto& item : content) {
    if (!first) {
      *text += '\n';
    }
    *text += item["text"].get_ref<const std::string&>();
    first = false;
  }
  return {true, ""};
}

}  // namespace

ChatJsonResult normalize_chat_json(std::string json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);
    if (!json.is_object()) {
      return ChatJsonResult{true, std::move(json_str), ""};
    }

    bool modified = false;
    if (!json.contains("temperature") || json["temperature"].is_null()) {
      json["temperature"] = kDefaultTemperature;
      modified = true;
    }

    if (json.contains("reasoning_effort") &&
        json["reasoning_effort"].is_string() &&
        !json["reasoning_effort"].get_ref<const std::string&>().empty()) {
      if (!json.contains("chat_template_kwargs") ||
          json["chat_template_kwargs"].is_null()) {
        json["chat_template_kwargs"] = nlohmann::json::object();
        modified = true;
      }
      if (json["chat_template_kwargs"].is_object() &&
          (!json["chat_template_kwargs"].contains("reasoning_effort") ||
           json["chat_template_kwargs"]["reasoning_effort"] !=
               json["reasoning_effort"])) {
        json["chat_template_kwargs"]["reasoning_effort"] =
            json["reasoning_effort"];
        modified = true;
      }
    }

    if (!json.contains("messages") || !json["messages"].is_array()) {
      return normalized_json_result(
          std::move(json), std::move(json_str), modified);
    }

    for (auto& message : json["messages"]) {
      if (!message.is_object()) {
        return json_error("Message in 'messages' array must be an object.");
      }
      if (!message.contains("content") || !message["content"].is_array()) {
        continue;
      }

      std::string text;
      auto [ok, error] = text_content(message["content"], &text);
      if (!ok) {
        return json_error(error);
      }
      message["content"] = std::move(text);
      modified = true;
    }

    return normalized_json_result(
        std::move(json), std::move(json_str), modified);
  } catch (const nlohmann::json::exception& e) {
    return json_error("Invalid JSON format: " + std::string(e.what()));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during chat JSON normalization: " << e.what();
    return json_error("Internal server error during JSON processing.");
  }
}

}  // namespace xllm_service
