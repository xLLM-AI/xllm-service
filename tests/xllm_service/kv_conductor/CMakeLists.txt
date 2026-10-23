include(cc_test)

cc_test(
  NAME
    kv_directory_test
  SRCS
    kv_directory_test.cpp
  DEPS
    :kv_directory
    :routing_policies
    :instance_controller
    :request_manager
    :xllm_http_service
    GTest::gtest_main
)
