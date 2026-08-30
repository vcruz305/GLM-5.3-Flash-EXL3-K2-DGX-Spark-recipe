#!/usr/bin/env bash
# Minimal reproducer for the GLM-5.3 K-pool tail out-of-bounds write.
#
# One request. A 76-token prompt that drives a 32,768-token generation, because
# the constraints are close to unsatisfiable and the model loops in thinking.
# The fault is driven by TOTAL SEQUENCE LENGTH reached through decoding, not by
# prompt length, which is why long-prompt probes only find the edges of it.
#
# Expected on an affected build: HTTP 500, and in the server log a CUDA illegal
# memory access from vllm/models/glm5next/nvidia/ops/kpool_compress.py.
#
# Whether it faults or corrupts silently depends on where the tail layer's view
# sits in the shared KV pool. Lower-offset layers write onto other layers' data
# and survive; the highest-offset layer escapes the allocation and dies. So a
# clean run here is not proof the build is unaffected.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8888}"
MODEL="${MODEL:-GLM-5.3-Flash-EXL3}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

read -r -d '' PROMPT <<'TXT' || true
Write a short proposal for a new research project that investigates how language evolves over time. I want to make it challenging, so:
1. Do not include any commas in your response.
2. Do not include the letter "c" anywhere in your response.
3. Your response should contain at least 250 words.
TXT

echo "target:     $BASE_URL"
echo "model:      $MODEL"
echo "max_tokens: $MAX_TOKENS"
echo

curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null || {
  echo "server is not answering /health" >&2
  exit 1
}

payload=$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": True},
}))')

start=$(date +%s)
http=$(curl -s -o /tmp/kpool_repro_body.json -w '%{http_code}' \
  --max-time 3600 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "$BASE_URL/v1/chat/completions" || true)
elapsed=$(( $(date +%s) - start ))

echo "HTTP $http after ${elapsed}s"
if [[ "$http" == "200" ]]; then
  python3 - <<'PY'
import json
d = json.load(open("/tmp/kpool_repro_body.json", encoding="utf-8"))
u = d.get("usage", {})
print("completion_tokens:", u.get("completion_tokens"))
print("finish_reason:    ", d["choices"][0].get("finish_reason"))
print()
print("Completed without faulting. On an affected build this means the writes")
print("landed inside the pool rather than outside it. Check the server log for")
print("corruption indicators rather than concluding the build is clean.")
PY
else
  echo
  head -c 400 /tmp/kpool_repro_body.json 2>/dev/null || true
  echo
  echo "Reproduced. Check the server log for the faulting kernel:"
  echo "  grep -n 'illegal memory access' <server.log>"
  echo "  expect kpool_compress.py in the frames above it"
  echo
  echo "For exact attribution rather than the next launch to notice, rerun the"
  echo "server with CUDA_LAUNCH_BLOCKING=1 and --enforce-eager."
fi
