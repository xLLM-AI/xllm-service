#!/usr/bin/env bash
set -uo pipefail

LOG_DIR=${LOG_DIR:-/tmp/xllm-service-demo}
MASTER_PID=${MASTER_PID:-$LOG_DIR/xllm-master.pid}
SIDECAR_PID=${SIDECAR_PID:-$LOG_DIR/xllm-sidecar.pid}
VLLM_PID=${VLLM_PID:-$LOG_DIR/vllm.pid}
ETCD=${ETCD:-127.0.0.1:2379}
STOP_VLLM=${STOP_VLLM:-1}

[ -f "$SIDECAR_PID" ] && kill "$(cat "$SIDECAR_PID")" 2>/dev/null && echo "  ✓ stopped sidecar ($(cat "$SIDECAR_PID"))" || echo "  - sidecar not running"
sleep 1
[ -f "$MASTER_PID" ] && kill "$(cat "$MASTER_PID")" 2>/dev/null && echo "  ✓ stopped master ($(cat "$MASTER_PID"))" || echo "  - master not running"
if [ "$STOP_VLLM" = "1" ]; then
  [ -f "$VLLM_PID" ] && kill "$(cat "$VLLM_PID")" 2>/dev/null && echo "  ✓ stopped vLLM ($(cat "$VLLM_PID"))" || echo "  - vLLM not running"
fi

etcdctl --endpoints="$ETCD" del --prefix XLLM:DEFAULT:  >/dev/null 2>&1 || true
etcdctl --endpoints="$ETCD" del --prefix XLLM:LOADMETRICS: >/dev/null 2>&1 || true
echo "  ✓ cleaned demo etcd keys if etcdctl is available"
