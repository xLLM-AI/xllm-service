include(cc_library)
cc_library(
  NAME routing_policies
  HDRS round_robin.h cache_aware_routing.h slo_aware.h
  SRCS round_robin.cpp cache_aware_routing.cpp slo_aware.cpp
  DEPS :common :kv_directory
)
