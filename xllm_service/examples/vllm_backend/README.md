# vLLM Backend Demo

This guide runs a local demo where xllm-service forwards OpenAI-compatible
requests to a vLLM backend. It verifies models, non-streaming chat completions,
and streaming chat completions.

## Architecture

```
OpenAI client
    │  HTTP (:9998)
    ▼
xllm-service master
    │  HTTP forwarding for backend_type=vllm
    ▼
vLLM backend (:18000)  ◀── vLLM sidecar registers it in etcd
```

vLLM instances are discovered through etcd keys such as
`XLLM:DEFAULT:127.0.0.1:18000` with `backend_type="vllm"`. The demo uses
`vllm_sidecar` to create and keep this registration under an etcd
lease; if the sidecar exits, the lease expires and the instance is removed.

## Prerequisites

- etcd reachable at `ETCD` (default `127.0.0.1:2379`)
- a built xllm-service binary at `build/xllm_service/xllm_master_serving`
- a vLLM virtual environment at `VLLM_VENV` if this script should start vLLM
- a local model path passed via `MODEL_DIR`

## Start

From the repository root:

```bash
MODEL_DIR=/path/to/model \
MODEL_NAME=demo-model \
VLLM_VENV=$HOME/vllm-venv \
bash xllm_service/examples/vllm_backend/start_demo.sh
```

If vLLM is already running, set `START_VLLM=0` and provide the backend address:

```bash
START_VLLM=0 INSTANCE_ADDR=127.0.0.1:18000 bash xllm_service/examples/vllm_backend/start_demo.sh
```

Useful overrides:

| Variable | Default | Description |
| --- | --- | --- |
| `ETCD` | `127.0.0.1:2379` | etcd endpoint |
| `HTTP_PORT` | `9998` | xllm-service HTTP port |
| `RPC_PORT` | `9999` | xllm-service RPC port |
| `VLLM_PORT` | `18000` | vLLM HTTP port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection for local vLLM |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM GPU memory fraction |
| `LOG_DIR` | `/tmp/xllm-service-demo` | demo logs and pid files |

## Verify

```bash
bash xllm_service/examples/vllm_backend/show.sh
bash xllm_service/examples/vllm_backend/compare.sh
```

The comparison script sends the same request directly to vLLM and through
xllm-service. With deterministic settings, the responses should match.

## Stop

```bash
bash xllm_service/examples/vllm_backend/stop_demo.sh
```

Set `STOP_VLLM=0` to leave an externally managed vLLM process running.

## Scope

This demo covers the minimal vLLM backend path: OpenAI-compatible request
forwarding, model listing, streaming/non-streaming responses, sidecar-based
etcd lease registration, and optional vLLM metrics heartbeat reporting. Cache-aware
routing, disaggregated PD, and mixed-backend clusters are out of scope for this
demo.
