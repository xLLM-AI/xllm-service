include(cc_library)
cc_library(
  NAME request
  HDRS request.h chat_message.h request_metrics.h request_output.h
  DEPS :common
)
cc_library(
  NAME request_preprocessor
  HDRS request_preprocessor.h
  SRCS request_preprocessor.cpp
  DEPS :request :chat_template :deepseek_v4_cpp_chat_template :model_type
)
cc_library(
  NAME output_dispatcher
  HDRS output_dispatcher.h
  SRCS output_dispatcher.cpp
  DEPS :common
)
