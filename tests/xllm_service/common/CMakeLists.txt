include(cc_test)

cc_test(
  NAME
    call_data_test
  SRCS
    call_data_test.cpp
  DEPS
    :common
    GTest::gtest_main
)
target_link_libraries(call_data_test
                      PRIVATE brpc-static leveldb::leveldb ZLIB::ZLIB
                              protobuf::libprotobuf OpenSSL::SSL
                              OpenSSL::Crypto)

cc_test(
  NAME
    types_test
  SRCS
    types_test.cpp
  DEPS
    :common
    GTest::gtest_main
)
target_link_libraries(types_test
                      PRIVATE brpc-static leveldb::leveldb ZLIB::ZLIB
                              protobuf::libprotobuf OpenSSL::SSL
                              OpenSSL::Crypto)

cc_test(
  NAME
    anthropic_tracer_test
  SRCS
    anthropic_tracer_test.cpp
  DEPS
    :common
    GTest::gtest_main
)
target_link_libraries(anthropic_tracer_test
                      PRIVATE brpc-static leveldb::leveldb ZLIB::ZLIB
                              protobuf::libprotobuf OpenSSL::SSL
                              OpenSSL::Crypto)
