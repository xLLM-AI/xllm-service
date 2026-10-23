include(cc_binary)
include(cc_library)
include(cc_test)
add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(config)
add_subdirectory(tokenizer)
add_subdirectory(chat_template)
add_subdirectory(kv_conductor)
add_subdirectory(request)
add_subdirectory(cluster)
add_subdirectory(coordinator)
add_subdirectory(router)
add_subdirectory(rpc_service)
add_subdirectory(http_service)
cc_library(
  NAME service_app
  HDRS service_app.h
  SRCS service_app.cpp
  DEPS :router :instance_controller :instance_discovery :service_membership
       :xllm_http_service :xllm_rpc_service :service_config
)
cc_binary(
  NAME xllm_master_serving
  SRCS main.cpp
  DEPS :service_app
)
add_subdirectory(examples)
