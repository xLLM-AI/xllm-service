# vLLM Sidecar — auto-register a vLLM instance into xllm-service

A vLLM server speaks only HTTP/OpenAI. It does not write etcd, hold a lease, or
emit heartbeats — so on its own it cannot join an xllm-service cluster. This
sidecar runs next to a vLLM process and bridges that gap, **with no C++ changes**:
xllm-service already discovers instances by watching the `XLLM:DEFAULT:` etcd
prefix.

It replaces the manual `demo/register_vllm.sh` (a one-shot `etcdctl put`) with a
long-running process that keeps registration in lockstep with vLLM's health.

## Lifecycle

```
            ┌─────────────┐   /health up    ┌────────────────────────────┐
  start ──▶ │ wait health │ ───────────────▶│ grant lease(ttl) + put key │
            └─────────────┘                 └──────────────┬─────────────┘
                  ▲                                         │ every keepalive-interval:
                  │  health recovers                        │   probe /health
                  │  (re-register, new incarnation)         ▼
            ┌─────┴───────┐  N consecutive fails   ┌──────────────────────┐
            │ deregistered│ ◀──────────────────────│ healthy → keepalive  │
            └─────────────┘                        │ unhealthy → count++  │
                  ▲                                 └──────────────────────┘
                  │ SIGTERM/SIGINT → revoke lease (immediate deregister)
```

* **health-gate** — never registers until vLLM `/health` is up.
* **lease, not heartbeat** — the etcd lease *is* the liveness signal. Keepalive
  refreshes the TTL **without rewriting the key**, so the master's watcher sees
  only the initial `PUT` and the final `DELETE` (on revoke or TTL expiry); no
  watch churn.
* **fast + safe deregister** — on N failed probes or a signal, the lease is
  revoked so the key disappears at once; if the sidecar is killed `-9`, the key
  still expires within `--lease-ttl`.
* **clean restart** — each (re)registration uses a fresh `incarnation_id`
  (uuid4), which the master uses to ignore stale deletes for a replaced instance
  and to clean up the previous incarnation.

## How the master picks it up (no C++ changes)

| step | code |
|---|---|
| watch `XLLM:DEFAULT:` etc. | `instance_mgr.cpp:133` `add_watch(prefix, ...)` |
| `PUT` → register | `instance_mgr.cpp:568` `update_instance_metainfo` → `register_instance` |
| lease expiry → `DELETE` → probe → suspect/remove | `instance_mgr.cpp:615+` |
| JSON schema | `common/types.h:248` `parse_from_json` (needs `name`,`rpc_address`,`type`) |
| key prefix / namespace | `instance_mgr.cpp:45` map, `utils.cpp:105` namespace |

> **Constraint:** a single vLLM instance with no decode peer is routable only as
> `type=DEFAULT (0)` — keep `--instance-type=DEFAULT`. This mirrors the M2 finding.

## Install

```bash
pip install -r vllm_sidecar/requirements.txt   # only `requests`
```

## Run

```bash
python -m vllm_sidecar.sidecar \
    --etcd-endpoints 127.0.0.1:2379 \
    --vllm-url       http://127.0.0.1:18000 \
    --register-addr  127.0.0.1:18000          # host:port, NO scheme
```

`--register-addr` defaults to the host:port derived from `--vllm-url`. The master
prepends `http://` itself (`init_brpc_channel`), so pass it bare.

### Flags

| flag | default | notes |
|---|---|---|
| `--etcd-endpoints` | `127.0.0.1:2379` | comma-separated; also `ETCD_ENDPOINTS` |
| `--etcd-namespace` | `""` | **must match** master's `--etcd_namespace` |
| `--etcd-username` / `--etcd-password` | `""` | also `ETCD_USERNAME`/`ETCD_PASSWORD` |
| `--vllm-url` | `http://127.0.0.1:18000` | local vLLM base URL |
| `--register-addr` | derived | host:port the master dials, no scheme |
| `--backend-type` | `vllm` | written into InstanceMetaInfo |
| `--instance-type` | `DEFAULT` | keep DEFAULT for a single instance |
| `--instance-name` | `vllm` | prefix for incarnation id / logs |
| `--lease-ttl` | `6` | seconds (≈ 2× keepalive interval) |
| `--keepalive-interval` | `2` | seconds between probe + lease refresh |
| `--health-fail-threshold` | `3` | failed probes before deregister |

## Verify

```bash
# instance appears (with a lease) and is reachable through xllm-service
etcdctl get --prefix XLLM:DEFAULT:
etcdctl lease list
curl http://127.0.0.1:9998/v1/models           # -> backend model list

# kill the sidecar; key disappears within lease-ttl, master logs removal
kill <sidecar_pid> && sleep 7
etcdctl get --prefix XLLM:DEFAULT:              # empty
```

## Scope

This PR provides **liveness-only** auto-registration via the etcd lease. Load
metrics (queue depth, KV-cache usage), an HTTP `/register`+`/heartbeat` contract
on the master, and `X-Internal-Token` auth are intentionally left to a follow-up
PR — none are needed for single-instance routing today.
