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
