include(cc_library)
include(cc_test)

cc_library (
  NAME
    chat_template
  HDRS
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
  SRCS
    message_projection.cpp
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

cc_test (
  NAME
    chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)
