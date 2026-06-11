#!/usr/bin/env bash
# =============================================================================
# xllm-service ↔ vLLM connectivity demo. Press Enter between steps.
# Set PAUSE=0 to disable the pauses and run straight through.
# =============================================================================
set -uo pipefail
HTTP=127.0.0.1:9998        # xllm-service unified entrypoint
ETCD=127.0.0.1:2379
MASTER_LOG=/tmp/xllm-master.log
PAUSE="${PAUSE:-1}"

hr()    { printf '\033[1;34m──────────────────────────────────────────────────────────\033[0m\n'; }
title() { hr; printf '\033[1;36m▶ %s\033[0m\n' "$*"; hr; }
run()   { printf '\033[1;33m$ %s\033[0m\n' "$*"; eval "$*"; echo; }
pause() { [ "$PAUSE" = "1" ] && read -rp $'\033[2m  [Enter to continue]\033[0m' _ || true; }

clear
title "Scene 1 ── Which backend instances are in the cluster?"
echo "xllm-service discovers backends through etcd. Let's see what is registered:"
run "etcdctl --endpoints=$ETCD get --prefix XLLM:DEFAULT: | tail -1 | python3 -m json.tool"
echo "↑ One instance with backend_type=vllm. How the master admitted it:"
run "grep -E 'Register a new|instances available' $MASTER_LOG | tail -2"
pause

title "Scene 2 ── Models seen through the unified entrypoint (GET /v1/models)"
echo "The client talks only to xllm-service($HTTP); it does not need to know the backend is vLLM:"
run "curl -s http://$HTTP/v1/models | python3 -m json.tool | head -8"
pause

title "Scene 3 ── Non-streaming chat (POST /v1/chat/completions)"
run "curl -s http://$HTTP/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{\"model\":\"qwen2.5-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"Introduce yourself in one sentence.\"}],\"max_tokens\":64}' \\
  | python3 -c \"import sys,json;r=json.load(sys.stdin);print('answer:',r['choices'][0]['message']['content']);print('usage:',r['usage'])\""
pause

title "Scene 4 ── Streaming chat (SSE, token by token)"
echo "OpenAI-style SSE is forwarded transparently, token by token, ending with data: [DONE]:"
run "curl -sN http://$HTTP/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{\"model\":\"qwen2.5-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 5.\"}],\"max_tokens\":40,\"stream\":true}'"
hr
echo "Demo complete. For technical details / transparency proof, run: bash xllm_service/examples/vllm_backend/compare.sh"
