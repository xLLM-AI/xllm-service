# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use the pinned Mooncake submodule, matching xLLM's third_party layout.
# Service needs only the SDK's metadata RPC declarations and TCP client.
set(MOONCAKE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Mooncake)
if(NOT EXISTS "${MOONCAKE_SOURCE_DIR}/mooncake-store/include/rpc_service.h")
  message(FATAL_ERROR
          "Initialize third_party/Mooncake: git submodule update --init --recursive third_party/Mooncake")
endif()
if(NOT EXISTS "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/coro_rpc/coro_rpc_client.hpp")
  message(FATAL_ERROR
          "Initialize the Mooncake yalantinglibs submodule: git submodule update --init --recursive third_party/Mooncake")
endif()

# Mooncake's public headers require complete Boost development headers.
# Keep this dependency private to the SDK instead of changing Service's vcpkg
# manifest. Standard CMAKE_INCLUDE_PATH can locate a preinstalled Boost SDK.
find_path(MOONCAKE_BOOST_INCLUDE_DIR boost/lockfree/queue.hpp
          HINTS ${Boost_INCLUDE_DIRS} REQUIRED)
if(NOT EXISTS "${MOONCAKE_BOOST_INCLUDE_DIR}/boost/parameter/template_keyword.hpp")
  message(FATAL_ERROR
          "Mooncake requires complete Boost development headers; set CMAKE_INCLUDE_PATH to their include directory")
endif()
find_package(Threads REQUIRED)

add_library(mooncake_metadata INTERFACE)
add_library(mooncake::metadata ALIAS mooncake_metadata)
target_compile_features(mooncake_metadata INTERFACE cxx_std_20)
target_include_directories(mooncake_metadata INTERFACE
  ${MOONCAKE_BOOST_INCLUDE_DIR}
  ${MOONCAKE_SOURCE_DIR}/mooncake-store/include/cachelib_memory_allocator/fake_include)
target_include_directories(mooncake_metadata SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MOONCAKE_SOURCE_DIR}/mooncake-store/include
  ${MOONCAKE_SOURCE_DIR}/mooncake-common/include
  ${MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/include
  ${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include
  ${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/thirdparty
  ${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/standalone)
target_link_libraries(mooncake_metadata INTERFACE
  glog::glog OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
