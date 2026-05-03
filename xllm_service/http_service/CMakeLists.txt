include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
    chat_json_parser
  HDRS
    chat_json_parser.h
  SRCS
    chat_json_parser.cpp
  DEPS
    glog::glog
    nlohmann_json::nlohmann_json
)

cc_library(
  NAME
  xllm_http_service
  HDRS
    service.h
    request_tracer.h
  SRCS
    service.cpp 
    request_tracer.cpp
  DEPS
    :chat_json_parser
    :common
    :request
    :scheduler
    absl::random_random
    absl::synchronization
    glog::glog
    nlohmann_json::nlohmann_json
    proto::proto_http_service
    proto_xllm
)
target_link_libraries(xllm_http_service PRIVATE brpc-static leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf OpenSSL::SSL OpenSSL::Crypto)

cc_binary(
  NAME 
    xllm_http_serving
  SRCS 
    main.cpp
  DEPS
    :xllm_http_service
    gflags::gflags
)

cc_test(
  NAME
    chat_json_parser_test
  SRCS
    chat_json_parser_test.cpp
  DEPS
    :chat_json_parser
    GTest::gtest_main
)
