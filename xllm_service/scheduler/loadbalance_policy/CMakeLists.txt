include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
    loadbalance_policy
  HDRS
    loadbalance_policy.h
    round_robin.h
    cache_aware_routing.h
    slo_aware_policy.h
    ds_load_balance.h
    mooncake_prefix_lookup.h
  SRCS
    round_robin.cpp
    cache_aware_routing.cpp
    slo_aware_policy.cpp
    ds_load_balance.cpp
    mooncake_prefix_lookup.cpp
  DEPS
    :chat_template
    :common
    :managers
    :kv_directory
)

if(USE_MOONCAKE_PREFIX_LOOKUP)
  add_library(mooncake_key_lookup STATIC mooncake_key_lookup.cpp)
  target_include_directories(mooncake_key_lookup PRIVATE
    ${COMMON_INCLUDE_DIRS})
  target_link_libraries(mooncake_key_lookup PRIVATE mooncake::metadata)
  target_compile_definitions(loadbalance_policy PRIVATE USE_MOONCAKE_PREFIX_LOOKUP)
  target_link_libraries(loadbalance_policy PUBLIC mooncake_key_lookup)
endif()
