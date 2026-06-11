#!/usr/bin/env bash
# =============================================================================
# xllm-service ↔ vLLM local demo.
# Starts vLLM, xllm-service master, and a vLLM sidecar if needed.
# Override paths/ports with environment variables for your environment.
# =============================================================================
set -uo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-venv}
MODEL_DIR=${MODEL_DIR:-}
MODEL_NAME=${MODEL_NAME:-demo-model}
LOG_DIR=${LOG_DIR:-/tmp/xllm-service-demo}
TMPDIR=${TMPDIR:-/tmp}
VLLM_LOG=${VLLM_LOG:-$LOG_DIR/vllm.log}
VLLM_PID=${VLLM_PID:-$LOG_DIR/vllm.pid}
MASTER_LOG=${MASTER_LOG:-$LOG_DIR/xllm-master.log}
MASTER_PID=${MASTER_PID:-$LOG_DIR/xllm-master.pid}
SIDECAR_LOG=${SIDECAR_LOG:-$LOG_DIR/xllm-sidecar.log}
SIDECAR_PID=${SIDECAR_PID:-$LOG_DIR/xllm-sidecar.pid}

ETCD=${ETCD:-127.0.0.1:2379}
VLLM_PORT=${VLLM_PORT:-18000}
HTTP_PORT=${HTTP_PORT:-9998}
RPC_PORT=${RPC_PORT:-9999}
INSTANCE_ADDR=${INSTANCE_ADDR:-127.0.0.1:${VLLM_PORT}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
START_VLLM=${START_VLLM:-1}

say() { printf '\n\033[1;36m[%s]\033[0m\n' "$*"; }
ok() { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; exit 1; }

mkdir -p "$LOG_DIR" "$TMPDIR"

say "0/4 checking etcd ($ETCD)"
curl -s -m 3 "http://$ETCD/version" >/dev/null 2>&1 \
  && ok "etcd is running" \
  || die "etcd is not reachable; start etcd on $ETCD first"

say "1/4 vLLM backend (:$VLLM_PORT)"
if curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  ok "vLLM is already running"
elif [ "$START_VLLM" != "1" ]; then
  die "vLLM is not running and START_VLLM=$START_VLLM"
else
  [ -n "$MODEL_DIR" ] || die "MODEL_DIR is required, e.g. MODEL_DIR=/path/to/model bash xllm_service/examples/vllm_backend/start_demo.sh"
  [ -x "$VLLM_VENV/bin/vllm" ] || die "vLLM executable not found: $VLLM_VENV/bin/vllm"
  TORCH_LIB=$("$VLLM_VENV"/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
    TMPDIR="$TMPDIR" \
    "$VLLM_VENV"/bin/vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port "$VLLM_PORT" --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      > "$VLLM_LOG" 2>&1 &
  echo $! > "$VLLM_PID"
  printf '  starting vLLM (pid=%s)' "$(cat "$VLLM_PID")"
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 && break
    ps -p "$(cat "$VLLM_PID")" >/dev/null 2>&1 || die "vLLM exited; see $VLLM_LOG"
    printf '.'; sleep 5
  done
  echo
  curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 \
    && ok "vLLM is ready" || die "vLLM readiness timed out; see $VLLM_LOG"
fi

say "2/4 xllm-service master (http :$HTTP_PORT, rpc :$RPC_PORT, backend=vllm)"
if [ -f "$MASTER_PID" ] && ps -p "$(cat "$MASTER_PID")" >/dev/null 2>&1; then
  ok "master is already running (pid=$(cat "$MASTER_PID"))"
else
  nohup "$ROOT"/build/xllm_service/xllm_master_serving \
    --default_backend_type=vllm \
    --etcd_addr="$ETCD" \
    --http_server_port="$HTTP_PORT" \
    --rpc_server_port="$RPC_PORT" \
    > "$MASTER_LOG" 2>&1 &
  echo $! > "$MASTER_PID"
  sleep 4
  ps -p "$(cat "$MASTER_PID")" >/dev/null 2>&1 \
    && ok "master started (pid=$(cat "$MASTER_PID"))" \
    || die "master failed to start; see $MASTER_LOG"
fi

say "3/4 registering vLLM instance ($INSTANCE_ADDR)"
if [ -d "$ROOT/vllm_sidecar" ]; then
  if [ -f "$SIDECAR_PID" ] && ps -p "$(cat "$SIDECAR_PID")" >/dev/null 2>&1; then
    ok "sidecar is already running (pid=$(cat "$SIDECAR_PID"))"
  else
    nohup env PYTHONPATH="$ROOT" python3 -m vllm_sidecar.sidecar \
      --etcd-endpoints "$ETCD" \
      --vllm-url "http://127.0.0.1:$VLLM_PORT" \
      --register-addr "$INSTANCE_ADDR" \
      --xllm-service-url "http://127.0.0.1:$HTTP_PORT" \
      > "$SIDECAR_LOG" 2>&1 &
    echo $! > "$SIDECAR_PID"
    ok "sidecar started (pid=$(cat "$SIDECAR_PID"))"
  fi
else
  bash "$ROOT/xllm_service/examples/vllm_backend/register_vllm.sh" "$INSTANCE_ADDR"
fi

sleep 2
for _ in $(seq 1 10); do
  curl -s -m 3 "http://127.0.0.1:$HTTP_PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -s -m 3 "http://127.0.0.1:$HTTP_PORT/v1/models" >/dev/null 2>&1 \
  && ok "xllm-service HTTP entrypoint is ready (:$HTTP_PORT)" \
  || die "HTTP entrypoint is not ready; see $MASTER_LOG"

say "4/4 environment ready"
cat <<EOF
  bash $ROOT/xllm_service/examples/vllm_backend/show.sh     # step-by-step connectivity demo
  bash $ROOT/xllm_service/examples/vllm_backend/compare.sh  # direct vLLM vs xllm-service forwarding
  bash $ROOT/xllm_service/examples/vllm_backend/stop_demo.sh     # stop master/sidecar and optional vLLM

  xllm-service: http://127.0.0.1:$HTTP_PORT
  vLLM backend: http://127.0.0.1:$VLLM_PORT
EOF
