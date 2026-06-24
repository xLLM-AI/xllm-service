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

cc_test(
  NAME
    model_type_test
  SRCS
    model_type_test.cpp
  DEPS
    :model_type
    GTest::gtest_main
)

cc_test(
  NAME
    chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cpp_chat_template_test
  SRCS
    deepseek_v4_cpp_chat_template_test.cpp
  DEPS
    :deepseek_v4_cpp_chat_template
    GTest::gtest_main
)
