include(cc_test)

cc_test(
  NAME
    message_projection_test
  SRCS
    message_projection_test.cpp
  DEPS
    :message_projection
    GTest::gtest_main
)
