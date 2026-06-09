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
