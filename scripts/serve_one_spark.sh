#!/usr/bin/env bash
# Serve GLM-5.3-Flash EXL3 K2 from a local vLLM + ExLlamaV3 installation on one DGX Spark (GB10).
# Winner on this pack: native MTP k=2. Do not mix MTP with a DFlash sidecar.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${HOME}/models/GLM-5.3-Flash-EXL3-K2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MTP_TOKENS="${MTP_TOKENS:-2}"          # 0 = no spec
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "missing $MODEL_DIR/config.json — run scripts/download_weights.sh"
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1 && ! python3 -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not on PATH / not importable. Create a venv and install vLLM first (see README)."
  exit 1
fi

python3 - <<PY
import json, sys
from pathlib import Path
cfg = json.loads(Path("${MODEL_DIR}/config.json").read_text())
q = cfg.get("quantization_config") or {}
print("arch", cfg.get("architectures"))
print("quant_method", q.get("quant_method"), "bits", q.get("bits"), "codebook", q.get("codebook"))
if str(q.get("quant_method", "")).lower() != "exl3":
    sys.exit("config.json is not an EXL3 pack")
if int(q.get("bits", -1)) != 2:
    print("WARNING: this recipe was measured at bits=2; config has bits", q.get("bits"))
PY

export MTP_TOKENS
export EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --quantization exl3
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype fp8
  --enable-prefix-caching
  --no-enable-flashinfer-autotune
  --skip-mm-profiling
  --limit-mm-per-prompt '{"image":4,"video":1}'
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
)

if [[ "${MTP_TOKENS}" != "0" ]]; then
  SPEC=$(python3 -c "import json,os; print(json.dumps({'method':'mtp','num_speculative_tokens':int(os.environ['MTP_TOKENS'])},separators=(',',':')))")
  ARGS+=(--speculative-config "$SPEC")
  ARGS+=(--cudagraph-capture-sizes 1 2 3 4 6 8 12)
fi

echo "EXL3_FUSED_MOE=$EXL3_FUSED_MOE MTP_TOKENS=$MTP_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
