include(cc_library)
cc_library(
  NAME service_config
  HDRS service_config.h config_loader.h
  SRCS config_loader.cpp
  DEPS :common
)
