include(cc_test)

cc_test(
  NAME
    tokenizer_test
  SRCS
    tokenizer_test.cpp
  DEPS
    :tokenizer
    GTest::gtest_main
)
