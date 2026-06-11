#!/usr/bin/env bash
# Compare direct vLLM output with xllm-service forwarding output.
set -uo pipefail

VLLM=${VLLM:-127.0.0.1:18000}
HTTP=${HTTP:-127.0.0.1:9998}
MODEL_NAME=${MODEL_NAME:-demo-model}
VLLM_LOG=${VLLM_LOG:-/tmp/xllm-service-demo/vllm.log}

PAYLOAD=$(printf '{"model":"%s","messages":[{"role":"user","content":"Introduce the city of Hangzhou in one sentence."}],"max_tokens":60,"temperature":0,"seed":42}' "$MODEL_NAME")
get_content() { python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"; }

printf '\n[1/3] identical request body\n%s\n' "$PAYLOAD"
printf '\n[2/3] direct vLLM vs xllm-service\n'
DIRECT=$(curl -s "http://$VLLM/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" | get_content)
VIA=$(curl -s "http://$HTTP/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" | get_content)
printf '  direct vLLM       : %s\n' "$DIRECT"
printf '  via xllm-service  : %s\n' "$VIA"
[ "$DIRECT" = "$VIA" ] && echo '  ✓ outputs match' || echo '  ! outputs differ'

printf '\n[3/3] recent vLLM access log (if available)\n'
[ -f "$VLLM_LOG" ] && grep -E 'POST /v1/chat/completions' "$VLLM_LOG" | tail -3 || echo "  log not found: $VLLM_LOG"
