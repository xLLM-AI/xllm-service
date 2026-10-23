include(cc_library)
include(cc_test)

cc_library(
  NAME
  etcd_client
  HDRS
    etcd_client.h
  SRCS
    etcd_client.cpp
  DEPS
    :common
    cpprest
    etcd-cpp-api
    glog::glog
    nlohmann_json::nlohmann_json
)
cc_library(
  NAME instance_discovery
  HDRS instance_discovery.h
  SRCS instance_discovery.cpp
  DEPS :etcd_client :instance_state
)
cc_library(
  NAME service_membership
  HDRS service_membership.h
  SRCS service_membership.cpp
  DEPS :etcd_client
)
