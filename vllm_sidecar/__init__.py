# Copyright 2025-2026 The xLLM Authors.
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
# ==============================================================================
"""vLLM sidecar: auto-register a vLLM instance into xllm-service via etcd lease.

A vLLM process speaks only HTTP/OpenAI -- it does not write etcd, hold a lease,
or emit heartbeats. This sidecar runs alongside vLLM and bridges that gap:

  * health-gate  -- only register once vLLM `/health` is up
  * register     -- write an InstanceMetaInfo JSON under an etcd lease
  * keep-alive   -- refresh the lease while vLLM stays healthy
  * deregister   -- revoke the lease on vLLM failure or graceful shutdown,
                    so xllm-service's etcd watcher removes the instance

This replaces the manual `demo/register_vllm.sh` with no C++ changes: the master
already discovers instances by watching the `XLLM:DEFAULT:` etcd prefix.
"""
