include(cc_library)

cc_library(
  NAME
    kv_directory
  HDRS
    kv_directory.h
    kv_index.h
    kv_event_decoder.h
  SRCS
    kv_directory.cpp
    kv_index.cpp
    kv_event_decoder.cpp
  DEPS
    :common
    glog::glog
    nlohmann_json::nlohmann_json
    xxHash
)

if(TARGET libzmq-static)
  target_link_libraries(kv_directory PUBLIC libzmq-static)
else()
  target_link_libraries(kv_directory PUBLIC libzmq)
endif()
