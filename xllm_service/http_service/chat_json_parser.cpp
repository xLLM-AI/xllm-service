/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

ChatJsonResult json_error(const std::string& error) {
  return ChatJsonResult{false, "", error};
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
    if (!json.contains("messages") || !json["messages"].is_array()) {
      return ChatJsonResult{true, std::move(json_str), ""};
    }

    bool modified = false;
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

    if (!modified) {
      return ChatJsonResult{true, std::move(json_str), ""};
    }
    return ChatJsonResult{true, json.dump(), ""};
  } catch (const nlohmann::json::exception& e) {
    return json_error("Invalid JSON format: " + std::string(e.what()));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during chat JSON normalization: " << e.what();
    return json_error("Internal server error during JSON processing.");
  }
}

}  // namespace xllm_service
