include(cc_test)

cc_test(
  NAME
    xllm_rpc_service_test
  SRCS
    rpc_service_test.cpp
  DEPS
    :disagg_generation_adapter
    :xllm_rpc_service
    gflags::gflags
    glog::glog
    GTest::gtest_main
    proto_xllm
)
