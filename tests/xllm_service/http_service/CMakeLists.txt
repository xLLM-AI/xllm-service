include(cc_test)

cc_test(
  NAME
    chat_json_parser_test
  SRCS
    chat_json_parser_test.cpp
  DEPS
    :chat_json_parser
    GTest::gtest_main
)

cc_test(
  NAME
    anthropic_adapter_test
  SRCS
    anthropic_adapter_test.cpp
  DEPS
    :anthropic_adapter
    GTest::gtest_main
)
