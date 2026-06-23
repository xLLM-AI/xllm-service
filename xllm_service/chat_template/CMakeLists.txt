include(cc_library)
include(cc_test)

cc_library (
  NAME
    chat_template
  HDRS
    chat_template.h
    jinja_chat_template.h
  SRCS
    jinja_chat_template.cpp
  DEPS
    :minja
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
)

cc_library (
  NAME
    message_projection
  HDRS
    message_projection.h
    mm_content_text.h
  SRCS
    message_projection.cpp
    mm_content_text.cpp
  INCLUDES
    ${CMAKE_SOURCE_DIR}/third_party/xllm
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/core
  DEPS
    :chat_template
    glog::glog
    nlohmann_json::nlohmann_json
    proto_xllm
)
target_link_libraries(message_projection PUBLIC brpc-static leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf OpenSSL::SSL OpenSSL::Crypto)

# Torch is a header-only dependency here: the upstream xllm::Message header
# includes <torch/torch.h> but the V4 template references no torch symbols, so
# we add only the torch include paths and never link libtorch.
set(TORCH_HEADER_INCLUDE_DIRS "" CACHE STRING "Torch header include dirs")
if(NOT TORCH_HEADER_INCLUDE_DIRS)
  execute_process(
    COMMAND python -c "import torch,os;p=os.path.dirname(torch.__file__);print(os.path.join(p,'include'));print(os.path.join(p,'include','torch','csrc','api','include'))"
    OUTPUT_VARIABLE _torch_include_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_include_res)
  if(NOT _torch_include_res EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch include dirs for DeepSeek V4 cpp chat template")
  endif()
  string(REPLACE "\n" ";" TORCH_HEADER_INCLUDE_DIRS "${_torch_include_out}")
endif()

cc_library (
  NAME
    deepseek_v4_cpp_chat_template
  HDRS
    deepseek_v4_cpp_chat_template.h
  SRCS
    deepseek_v4_cpp_chat_template.cpp
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/core/framework/chat_template/deepseek_v4_cpp_template.cpp
  INCLUDES
    ${CMAKE_SOURCE_DIR}/third_party/xllm
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/core
    ${TORCH_HEADER_INCLUDE_DIRS}
  DEPS
    :chat_template
    :message_projection
    glog::glog
    nlohmann_json::nlohmann_json
    proto_xllm
)
target_link_libraries(deepseek_v4_cpp_chat_template PUBLIC brpc-static leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf OpenSSL::SSL OpenSSL::Crypto)

cc_test (
  NAME
    chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)
