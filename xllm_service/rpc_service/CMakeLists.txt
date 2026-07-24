include(cc_binary)
include(cc_library)

cc_library(
  NAME
  disagg_generation_adapter
  HDRS
    disagg_generation_adapter.h
  SRCS
    disagg_generation_adapter.cpp
  DEPS
    :common
    proto::proto_rpc_service
)

cc_library(
  NAME
  xllm_rpc_service
  HDRS
    service.h
  SRCS
    service.cpp
  DEPS
    :common
    :disagg_generation_adapter
    :scheduler
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
    tokenizer
    chat_template
)
target_link_libraries(xllm_rpc_service PRIVATE brpc-static)

cc_binary(
  NAME 
    xllm_rpc_serving
  SRCS 
    main.cpp
  DEPS
    :xllm_rpc_service
    gflags::gflags   
)

cc_library(
  NAME
  xllm_rpc_client
  HDRS
    client.h
  SRCS
    client.cpp
  DEPS
    :common
    glog::glog
    proto::proto_rpc_service
)
target_link_libraries(xllm_rpc_client PRIVATE brpc-static)
