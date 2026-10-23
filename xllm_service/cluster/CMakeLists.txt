include(cc_library)
cc_library(
  NAME instance_state
  HDRS instance_state.h instance_types.h
  SRCS instance_state.cpp
  DEPS :common
)
cc_library(
  NAME instance_client
  HDRS instance_client.h
  SRCS instance_client.cpp
  DEPS :common brpc-static proto_xllm
)
cc_library(
  NAME instance_controller
  HDRS instance_controller.h
  SRCS instance_controller.cpp
  DEPS :instance_state :instance_client proto::proto_rpc_service
)
