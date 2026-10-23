include(cc_test)
cc_test(NAME instance_test SRCS instance_test.cpp
        DEPS :instance_controller :request_manager :routing_policies GTest::gtest_main)
