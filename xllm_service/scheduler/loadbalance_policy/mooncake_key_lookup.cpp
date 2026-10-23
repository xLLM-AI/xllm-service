/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Mooncake/mooncake-store/include/rpc_service.h>
#include <async_simple/coro/SyncAwait.h>

#include <chrono>
#include <utility>
#include <ylt/coro_rpc/coro_rpc_client.hpp>

#include "scheduler/loadbalance_policy/mooncake_prefix_lookup.h"

namespace xllm_service {
namespace {

// The SDK's metadata RPC uses TCP only. No RealClient, transfer engine,
// registered memory, Store segment, or PUT/GET buffers are allocated here.
class MooncakeKeyLookup final : public StoreKeyLookup {
 public:
  MooncakeKeyLookup(std::string address,
                    std::string tenant_id,
                    int32_t timeout_ms)
      : address_(std::move(address)),
        tenant_id_(std::move(tenant_id)),
        timeout_(timeout_ms) {}

  bool batch_exists(const std::vector<std::string>& keys,
                    std::vector<bool>* exists) override {
    // A per-query client isolates concurrent requests and timed-out sockets.
    // One deadline covers connect and the single batched RPC; no retries.
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    coro_rpc::coro_rpc_client client;
    const auto connected =
        async_simple::coro::syncAwait(client.connect(address_, timeout_));
    if (connected) {
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero()) {
      return false;
    }
    const auto response = async_simple::coro::syncAwait(
        client.call_for<&mooncake::WrappedMasterService::BatchExistKey>(
            remaining, keys, tenant_id_));
    if (!response.has_value() || response->size() != keys.size()) {
      return false;
    }
    exists->clear();
    exists->reserve(keys.size());
    for (const auto& item : *response) {
      if (!item.has_value()) {
        return false;
      }
      exists->emplace_back(*item);
    }
    return true;
  }

 private:
  std::string address_;
  std::string tenant_id_;
  std::chrono::milliseconds timeout_;
};

}  // namespace

std::unique_ptr<StoreKeyLookup> create_mooncake_key_lookup(
    std::string master_address,
    std::string tenant_id,
    int32_t timeout_ms) {
  return std::make_unique<MooncakeKeyLookup>(
      std::move(master_address), std::move(tenant_id), timeout_ms);
}

}  // namespace xllm_service
