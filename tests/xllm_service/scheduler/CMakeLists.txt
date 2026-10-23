include(cc_test)

cc_test(
  NAME
    request_output_test
  SRCS
    request_output_test.cpp
  DEPS
    :request
    GTest::gtest_main
)

cc_test(
  NAME
    request_metrics_test
  SRCS
    request_metrics_test.cpp
  DEPS
    :request
    GTest::gtest_main
)

cc_test(
  NAME
    response_handler_test
  SRCS
    response_handler_test.cpp
  DEPS
    :response_handler
    GTest::gtest_main
)

cc_test(
  NAME
    request_admission_test
  SRCS
    request_admission_test.cpp
  DEPS
    :scheduler
    GTest::gtest_main
)

cc_test(
  NAME session_aware_policy_test
  SRCS session_aware_policy_test.cpp
  DEPS :loadbalance_policy :xllm_http_service GTest::gtest_main
)
