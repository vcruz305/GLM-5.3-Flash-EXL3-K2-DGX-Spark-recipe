#!/usr/bin/env bash
# sixcat 0.5.1+ against the live vLLM /v1. HTTP only — not stdio, not an agent facade.
# Stock glm-5.x sends 6 stop strings; vLLM's OpenAI schema allows 4. Trim in the
# sixcat checkout (see README) and set preclose_think=false or thinking never reaches the template.
set -euo pipefail

ROOT="${SIXCAT_ROOT:-${HOME}/src/sixcat-eval}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-GLM-5.3-Flash-EXL3}"
OUT_DIR="${OUT_DIR:-$(pwd)/results}"
STAMP="${STAMP:-glm53-flash-exl3-k2-mtp2}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$ROOT" ]]; then
  echo "Clone sixcat first: git clone --branch v0.5.1 https://github.com/vcruz305/sixcat-eval.git $ROOT"
  exit 1
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv "${ROOT}/.venv"
  PY="${ROOT}/.venv/bin/python"
  "$PY" -m pip install -U pip
  "$PY" -m pip install -e "$ROOT"
fi

exec "$PY" -m sixcat \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --policy vendor \
  --policy-family glm-5.x \
  --thinking on \
  --limit 20 \
  --max-minutes 0 \
  --request-timeout 3600 \
  --ctx "${CTX:-65536}" \
  --concurrency 1 \
  --transport openai \
  --out "$OUT_DIR/${STAMP}.json" \
  --log "$OUT_DIR/${STAMP}.jsonl" \
  --no-resume
