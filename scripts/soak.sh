#!/usr/bin/env bash
# Long-generation soak gate. Run this before publishing a runtime artifact.
#
# The K-pool tail bug is driven by GENERATED tokens, not prompt length or
# context setting. Short-generation tests (128-token benches, 400-token
# ladders, /v1 smokes, 8-token context probes) do not exercise it, which is
# how a broken runtime got published in the first place.
#
# Requires the server to be running with:
#   GLM_KPOOL_TAIL_BOUNDS=1     (scripts/patch_kpool_tail_detector.py)
#
# Exit 0 only if zero overruns were observed across the whole soak.
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8888}"
MODEL="${MODEL:-GLM-5.3-Flash-EXL3}"
SERVER_LOG="${SERVER_LOG:-}"
TARGET_TOKENS="${TARGET_TOKENS:-20000}"
PER_REQUEST="${PER_REQUEST:-4096}"

if [[ -z "$SERVER_LOG" ]]; then
  echo "set SERVER_LOG=/path/to/server.log so the gate can read the counter" >&2
  exit 2
fi
if [[ ! -r "$SERVER_LOG" ]]; then
  echo "cannot read SERVER_LOG: $SERVER_LOG" >&2
  exit 2
fi
if ! curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
  echo "server is not answering $BASE_URL/health" >&2
  exit 2
fi
if ! grep -q "KPOOL_TAIL_BOUNDS" "$SERVER_LOG" 2>/dev/null; then
  echo "WARNING: no KPOOL_TAIL_BOUNDS lines in $SERVER_LOG yet." >&2
  echo "Confirm the detector is installed and GLM_KPOOL_TAIL_BOUNDS=1 is set," >&2
  echo "otherwise this gate silently passes everything." >&2
fi

before=$(grep -c "KPOOL_TAIL_OVERRUN" "$SERVER_LOG" 2>/dev/null || echo 0)

# Prompts chosen to run long. The near-unsatisfiable constraints make the model
# loop in thinking, which is what drives sequence position high enough to matter.
PROMPTS=(
"Write a short proposal for a new research project that investigates how language evolves over time. I want to make it challenging, so:
1. Do not include any commas in your response.
2. Do not include the letter \"c\" anywhere in your response.
3. Your response should contain at least 250 words."
"Write a 600 word essay about the history of bridges. Do not use the letter e at any point. Do not use commas."
"Enumerate 200 distinct facts about the ocean. Every fact must be exactly one sentence and must not contain the letter a."
"Explain quantum entanglement in at least 800 words without using any word longer than six letters."
"Write a detailed technical postmortem of a fictional outage, at least 1000 words, with no proper nouns and no numbers."
)

generated=0
req=0
echo "soak: target ${TARGET_TOKENS} generated tokens, ${PER_REQUEST} per request"
while (( generated < TARGET_TOKENS )); do
  p="${PROMPTS[$(( req % ${#PROMPTS[@]} ))]}"
  req=$(( req + 1 ))
  payload=$(MODEL="$MODEL" P="$p" MT="$PER_REQUEST" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["P"]}],
    "max_tokens": int(os.environ["MT"]),
    "temperature": 1.0, "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": True},
}))')
  body=$(mktemp)
  code=$(curl -s -o "$body" -w '%{http_code}' --max-time 3600 \
    -H 'Content-Type: application/json' -d "$payload" \
    "$BASE_URL/v1/chat/completions" || echo 000)
  if [[ "$code" != "200" ]]; then
    echo "  request $req: HTTP $code — engine likely died"
    head -c 200 "$body"; echo
    rm -f "$body"
    echo
    echo "SOAK RESULT: FAIL (engine fault after ~${generated} generated tokens)"
    exit 1
  fi
  n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['usage']['completion_tokens'])" "$body" 2>/dev/null || echo 0)
  rm -f "$body"
  generated=$(( generated + n ))
  echo "  request $req: +${n} tokens (total ${generated})"
done

sleep 2
after=$(grep -c "KPOOL_TAIL_OVERRUN" "$SERVER_LOG" 2>/dev/null || echo 0)
new=$(( after - before ))

echo
grep "KPOOL_TAIL_BOUNDS calls=" "$SERVER_LOG" 2>/dev/null | tail -1
echo "generated tokens: ${generated}"
echo "new overruns during soak: ${new}"
echo

if (( new > 0 )); then
  echo "SOAK RESULT: FAIL — ${new} out-of-bounds tail writes"
  echo "Do not publish this runtime. See docs/KPOOL_TAIL_BUG.md."
  grep "KPOOL_TAIL_OVERRUN" "$SERVER_LOG" | tail -3
  exit 1
fi

echo "SOAK RESULT: PASS — zero out-of-bounds tail writes over ${generated} generated tokens"
