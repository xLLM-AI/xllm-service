include(cc_test)
cc_test(NAME request_output_test SRCS request_output_test.cpp DEPS :request GTest::gtest_main)
cc_test(NAME request_metrics_test SRCS request_metrics_test.cpp DEPS :request GTest::gtest_main)
cc_test(NAME request_admission_test SRCS request_admission_test.cpp DEPS :instance_controller :request_manager :routing_policies GTest::gtest_main)
