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

#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm_service {
namespace {

bool get_str_field(const nlohmann::ordered_json& json,
                   const char* key,
                   std::string* value) {
  if (!json.contains(key) || !json[key].is_string()) {
    return false;
  }
  *value = json[key].get<std::string>();
  return true;
}

std::vector<std::string> thinking_texts(
    const nlohmann::ordered_json& messages) {
  std::vector<std::string> texts;
  for (const auto& message : messages) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    for (const auto& item : message["content"]) {
      std::string type;
      std::string thinking;
      if (get_str_field(item, "type", &type) && type == "thinking" &&
          get_str_field(item, "thinking", &thinking) && !thinking.empty()) {
        texts.emplace_back(std::move(thinking));
      }
    }
  }
  return texts;
}

bool has_all_thinking(const std::string& prompt,
                      const nlohmann::ordered_json& messages) {
  for (const auto& thinking : thinking_texts(messages)) {
    if (prompt.find(thinking) == std::string::npos) {
      return false;
    }
  }
  return true;
}

std::string join_parts(const std::vector<std::string>& parts) {
  std::string text;
  for (const auto& part : parts) {
    if (part.empty()) {
      continue;
    }
    if (!text.empty()) {
      text += '\n';
    }
    text += part;
  }
  return text;
}

std::string prompt_content(const nlohmann::ordered_json& content) {
  std::vector<std::string> parts;
  for (const auto& item : content) {
    std::string type;
    if (!get_str_field(item, "type", &type)) {
      continue;
    }
    if (type == "thinking") {
      std::string thinking;
      if (get_str_field(item, "thinking", &thinking) && !thinking.empty()) {
        parts.emplace_back("<think>" + thinking + "</think>");
      }
    } else if (type == "text") {
      std::string text;
      if (get_str_field(item, "text", &text) && !text.empty()) {
        parts.emplace_back(std::move(text));
      }
    }
  }
  return join_parts(parts);
}

bool has_thinking_block(const nlohmann::ordered_json& content) {
  for (const auto& item : content) {
    std::string type;
    if (get_str_field(item, "type", &type) && type == "thinking") {
      return true;
    }
  }
  return false;
}

nlohmann::ordered_json with_thinking_fallback(
    const nlohmann::ordered_json& messages) {
  auto fallback = messages;
  for (auto& message : fallback) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    if (!has_thinking_block(message["content"])) {
      continue;
    }
    auto text = prompt_content(message["content"]);
    if (!text.empty()) {
      message["content"] = std::move(text);
    }
  }
  return fallback;
}

}  // namespace

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        args_.chat_template(), args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm_service::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm_service::JsonTool>& json_tools) const {
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, json_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm_service::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<Message::MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<Message::MMContentVec>(message.content));
    }
    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      for (const auto& tool_call : *message.tool_calls) {
        // Tool-call arguments arrive as a JSON string (OpenAI wire format), but
        // chat templates such as GLM iterate `arguments.items()`, which only
        // works on a JSON object. A non-object value (truncated/malformed JSON,
        // or a JSON array/scalar) makes minja throw "Unknown method: items",
        // which would abort the whole process. Always hand the template an
        // object: parse when possible and fall back to an empty object
        // otherwise, so rendering can never throw on this field.
        nlohmann::ordered_json arguments_json = nlohmann::ordered_json::object();
        if (!tool_call.function.arguments.empty()) {
          try {
            auto parsed = nlohmann::json::parse(tool_call.function.arguments);
            if (parsed.is_object()) {
              arguments_json = std::move(parsed);
            } else {
              LOG(WARNING) << "Tool-call arguments are not a JSON object, "
                              "rendering empty args: "
                           << tool_call.function.arguments;
            }
          } catch (const nlohmann::json::exception& e) {
            LOG(WARNING) << "Failed to parse tool-call arguments, rendering "
                            "empty args: "
                         << e.what();
          }
        }
        tool_calls_json.emplace_back(nlohmann::ordered_json{
            {"id", tool_call.id},
            {"type", tool_call.type},
            {"function",
             {{"name", tool_call.function.name},
              {"arguments", std::move(arguments_json)}}}});
      }
      message_json["tool_calls"] = std::move(tool_calls_json);
    }
    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = message.reasoning_content.value();
    }
    if (!message.tool_call_id.empty()) {
      message_json["tool_call_id"] = message.tool_call_id;
    }

    messages_json.push_back(message_json);
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();
  for (const auto& json_tool : json_tools) {
    nlohmann::ordered_json tool_json;
    tool_json["type"] = json_tool.type;

    nlohmann::ordered_json function_json;
    function_json["name"] = json_tool.function.name;
    function_json["description"] = json_tool.function.description;
    function_json["parameters"] = json_tool.function.parameters;

    tool_json["function"] = function_json;
    tools_json.push_back(tool_json);
  }
  // apply the template
  auto prompt = apply(messages_json, tools_json, chat_template_kwargs);
  if (!prompt.has_value() || has_all_thinking(prompt.value(), messages_json)) {
    return prompt;
  }

  auto fallback_messages = with_thinking_fallback(messages_json);
  auto fallback_prompt =
      apply(fallback_messages, tools_json, chat_template_kwargs);
  if (fallback_prompt.has_value()) {
    return fallback_prompt;
  }
  return prompt;
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools) const {
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  minja::chat_template_inputs input;
  input.messages = messages;
  input.tools = tools;
  input.add_generation_prompt = true;
  input.extra_context = chat_template_kwargs;
  minja::chat_template_options options;

  // minja throws std::runtime_error on any rendering failure (e.g. calling
  // `.items()` on a non-object value). The OpenAI HTTP path does not wrap the
  // scheduler call in a try/catch, so an uncaught exception here would escape
  // the brpc handler and terminate the whole process. Contain it and degrade
  // to an empty result, letting callers report a normal "failed to construct
  // prompt" error instead of crashing.
  try {
    return template_->apply(input, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply chat template: " << e.what();
    return std::nullopt;
  }
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const Message::MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;

    if (item.type == "text") {
      item_json["text"] = item.text;
    } else if (item.type == "thinking") {
      item_json["thinking"] = item.thinking;
      if (!item.signature.empty()) {
        item_json["signature"] = item.signature;
      }
    } else if (item.type == "redacted_thinking") {
      item_json["data"] = item.data;
    } else if (item.type == "tool_use") {
      item_json["id"] = item.id;
      item_json["name"] = item.name;
      item_json["input"] = item.input;
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return std::move(content_json);
}

}  // namespace xllm_service
