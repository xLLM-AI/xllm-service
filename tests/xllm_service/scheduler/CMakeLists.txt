include(cc_test)

cc_test(
  NAME
    xllm_chat_parse_bridge_test
  SRCS
    xllm_chat_parse_bridge_test.cpp
  DEPS
    :xllm_chat_parse_runtime
    GTest::gtest_main
)

cc_test(
  NAME
    response_handler_usage_test
  SRCS
    response_handler_usage_test.cpp
  DEPS
    :scheduler
    GTest::gtest_main
)
