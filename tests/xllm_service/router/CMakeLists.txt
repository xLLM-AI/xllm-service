include(cc_test)
cc_test(NAME routing_policy_test SRCS routing_policy_test.cpp DEPS :routing_policies :routing_state GTest::gtest_main)
