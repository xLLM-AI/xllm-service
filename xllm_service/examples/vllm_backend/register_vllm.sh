#!/usr/bin/env bash
# =============================================================================
# Register a vLLM instance into the xllm-service cluster.
#
# Usage: register_vllm.sh [addr]    addr defaults to 127.0.0.1:18000
#
# How it works: xllm-service discovers instances by watching the etcd prefix
# XLLM:DEFAULT:. This writes a single InstanceMetaInfo JSON (backend_type=vllm,
# type=0=DEFAULT); the master's watcher then runs register_instance -> builds an
# HTTP channel -> the instance comes online.
#
# type must be 0 (DEFAULT): with a single instance and no decode peer, the
# scheduler only admits DEFAULT-type instances.
# (A vLLM sidecar will later do this automatically instead of writing etcd by hand.)
# =============================================================================
set -euo pipefail

ADDR="${1:-127.0.0.1:18000}"
ETCD="${ETCD:-127.0.0.1:2379}"
KEY="XLLM:DEFAULT:${ADDR}"

JSON=$(cat <<EOF
{"name":"${ADDR}","rpc_address":"${ADDR}","type":0,"backend_type":"vllm","incarnation_id":"vllm-1","register_ts_ms":1}
EOF
)

etcdctl --endpoints="$ETCD" put "$KEY" "$JSON" >/dev/null
echo "  ✓ registered: $KEY"
echo "    $JSON"
