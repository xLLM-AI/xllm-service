include(cc_library)
add_subdirectory(policies)
cc_library(
  NAME routing_state
  HDRS routing_state.h routing_types.h routing_policy.h
  SRCS routing_state.cpp
  DEPS :instance_state :request :kv_directory
)
cc_library(
  NAME request_manager
  HDRS request_manager.h
  SRCS request_manager.cpp
  DEPS :routing_state :output_dispatcher
)
cc_library(
  NAME router
  HDRS router.h
  SRCS router.cpp
  DEPS :routing_state :request_preprocessor :routing_policies
)
