#!/usr/bin/env bash
# Serve GLM-5.3-Flash EXL3 K2 from a local vLLM + ExLlamaV3 installation on one DGX Spark (GB10).
# Select exactly one speculative method: none, MTP, or DFlash2.
set -euo pipefail

# vLLM's has_flashinfer() returns False when nvcc is not on PATH (FlashInfer
# JIT-compiles its kernels on this box; there are no pre-downloaded cubins), and
# the only sparse-MLA backend for SM121 is then rejected at engine init:
#   ValueError: No valid attention backend found for cuda with ... use_sparse=True
#   Reasons: {FLASHINFER_MLA_SPARSE_SM120: [... requires FlashInfer's sparse MLA decode API]}
# Put the CUDA 13 toolkit on PATH before vLLM looks for it.
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; echo "nvcc was not on PATH; added $d"; break; fi
  done
fi
# FlashInfer's JIT runs ninja, which pip installs into the venv's bin/. Serve
# through the venv's vllm and put that bin/ on PATH so ninja resolves.
if ! command -v ninja >/dev/null 2>&1; then
  for d in "${VENV:-$HOME/venvs/glm53-exl3-local}/bin" "$(dirname "$(command -v vllm 2>/dev/null || echo /nonexistent/x)")"; do
    if [[ -x "$d/ninja" ]]; then export PATH="$d:$PATH"; echo "ninja was not on PATH; added $d"; break; fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install the CUDA 13 toolkit or: export PATH=/usr/local/cuda-13.0/bin:\$PATH" >&2
  echo "Without it vLLM rejects FLASHINFER_MLA_SPARSE_SM120 and fails with 'No valid attention backend found'." >&2
  exit 1
fi

MODEL_DIR="${MODEL_DIR:-${HOME}/models/GLM-5.3-Flash-EXL3-K2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# 0.87 is the measured 8k default. The 64k measurements (sixcat, the 64k
# ladders) were taken at 0.91, and at 0.87 the KV pool is materially smaller.
# Pick the measured value for the context unless the caller overrides it.
if [[ -z "${GPU_MEM_UTIL:-}" ]]; then
  if (( MAX_MODEL_LEN >= 65536 )); then GPU_MEM_UTIL=0.91; else GPU_MEM_UTIL=0.87; fi
fi
# The sparse indexer's per-chunk logits allocations grow with the prefix as
# chunked prefill advances. On GB10 unified memory cudaMalloc never fails, so
# the caching allocator never reuses freed blocks and ratchets reserved memory
# until host RAM is gone -- prefill livelocks past roughly 200k tokens with
# the engine silent and /health still returning 200. expandable_segments lets
# the allocator reuse that space. (Replay-validated alternative:
# VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64, which shrinks and equalizes sub-chunk
# size so blocks get reused; not the default here.)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# KV_CACHE_MEMORY_BYTES pins the KV pool to an exact byte count instead of the
# utilization-derived pool (--gpu-memory-utilization). 3221225472 (3 GiB) is
# the single-request 262k-token configuration used in the 2026-09-01 258k
# verification (349,525 fp8 KV tokens, enough for one 262,144-token
# request). The utilization-derived pool (leave this unset) is for
# multi-request serving. KV_CACHE_MEMORY is kept as a backwards-compatible
# alias; vLLM wants a bare integer byte count, not "3GiB".
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-${KV_CACHE_MEMORY:-}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
SPEC_METHOD="${SPEC_METHOD:-mtp}"
MTP_TOKENS="${MTP_TOKENS:-2}"
DFLASH_DIR="${DFLASH_DIR:-${HOME}/models/GLM-5.3-Flash-DFlash2}"
DFLASH_TOKENS="${DFLASH_TOKENS:-3}"
DFLASH_QUANTIZATION="${DFLASH_QUANTIZATION:-}"
DFLASH_SAMPLE_METHOD="${DFLASH_SAMPLE_METHOD:-probabilistic}"
DFLASH_KV_DTYPE="${DFLASH_KV_DTYPE:-auto}"
# TRITON_ATTN is the only draft backend that works. With FLASH_ATTN the
# target's sparse-MLA K-pool indexer faults at the first cache page
# transition and takes the engine down. See docs/MEASUREMENTS.md.
DFLASH_ATTN_BACKEND="${DFLASH_ATTN_BACKEND:-TRITON_ATTN}"
GLM_DFLASH_MANAGER_BLOCK_SIZE="${GLM_DFLASH_MANAGER_BLOCK_SIZE:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_DIR}/chat_template.jinja}"
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "missing $MODEL_DIR/config.json — run scripts/download_weights.sh"
  exit 1
fi
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "missing $CHAT_TEMPLATE — run scripts/patch_chat_template_thinking.py" >&2
  exit 1
fi
case "$SPEC_METHOD" in
  none|mtp|dflash) ;;
  *) echo "SPEC_METHOD must be none, mtp, or dflash" >&2; exit 2 ;;
esac
if [[ "$SPEC_METHOD" == "dflash" && ! -f "$DFLASH_DIR/config.json" ]]; then
  echo "missing DFlash2 checkpoint: $DFLASH_DIR/config.json" >&2
  exit 1
fi
case "$DFLASH_SAMPLE_METHOD" in
  greedy|probabilistic) ;;
  *) echo "DFLASH_SAMPLE_METHOD must be greedy or probabilistic" >&2; exit 2 ;;
esac
case "$ENABLE_PREFIX_CACHING" in
  0|1) ;;
  *) echo "ENABLE_PREFIX_CACHING must be 0 or 1" >&2; exit 2 ;;
esac
case "$ENFORCE_EAGER" in
  0|1) ;;
  *) echo "ENFORCE_EAGER must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "$GLM_DFLASH_MANAGER_BLOCK_SIZE" ]]; then
  if (( GLM_DFLASH_MANAGER_BLOCK_SIZE < 128 \
     || GLM_DFLASH_MANAGER_BLOCK_SIZE % 16 != 0 )); then
    echo "GLM_DFLASH_MANAGER_BLOCK_SIZE must be >=128 and divisible by 16" >&2
    exit 2
  fi
  # The mixed GLM+DFlash cache planner treats this as an upper bound and
  # resolves the largest valid divisor of the target manager block.
  export GLM_DFLASH_MANAGER_BLOCK_SIZE
fi

if ! command -v vllm >/dev/null 2>&1 && ! python3 -c "import vllm" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
vLLM is not importable in this environment.

Do NOT run `pip install vllm`. Stock vLLM cannot serve this pack: it has
neither the EXL3 quantization method nor the Glm5Next architecture, and no
command-line flag turns them on. Both come from this recipe's runtime.

Install the prebuilt runtime instead (minutes, no compiler):

    bash scripts/install_prebuilt.sh

Then confirm before going further:

    python scripts/preflight.py
MSG
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

export MTP_TOKENS DFLASH_TOKENS DFLASH_DIR DFLASH_QUANTIZATION
export DFLASH_SAMPLE_METHOD DFLASH_KV_DTYPE DFLASH_ATTN_BACKEND
export EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --quantization exl3
  --load-format "$LOAD_FORMAT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype fp8
  --no-enable-flashinfer-autotune
  --skip-mm-profiling
  --limit-mm-per-prompt '{"image":4,"video":1}'
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --chat-template "$CHAT_TEMPLATE"
)

if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
else
  ARGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi

if [[ "$SPEC_METHOD" == "mtp" ]]; then
  SPEC=$(python3 -c "import json,os; print(json.dumps({'method':'mtp','num_speculative_tokens':int(os.environ['MTP_TOKENS'])},separators=(',',':')))")
  ARGS+=(--speculative-config "$SPEC")
  # Exact target verification sizes are k+1. Keep 3/4/5 so the MTP k=2/3/4
  # ladder does not silently pad a decode step to a larger captured graph.
  ARGS+=(--cudagraph-capture-sizes 1 2 3 4 5 6 8 12)
elif [[ "$SPEC_METHOD" == "dflash" ]]; then
  SPEC=$(python3 -c 'import json,os; d={"method":"dflash","model":os.environ["DFLASH_DIR"],"num_speculative_tokens":int(os.environ["DFLASH_TOKENS"]),"kv_cache_dtype":os.environ["DFLASH_KV_DTYPE"],"attention_backend":os.environ["DFLASH_ATTN_BACKEND"],"draft_sample_method":os.environ["DFLASH_SAMPLE_METHOD"],"rejection_sample_method":"standard","draft_tensor_parallel_size":1}; q=os.environ.get("DFLASH_QUANTIZATION"); d.update({"quantization":q} if q else {}); print(json.dumps(d,separators=(",",":")))')
  ARGS+=(--speculative-config "$SPEC")
  ARGS+=(--cudagraph-capture-sizes 1 2 4 8 16)
fi

echo "EXL3_FUSED_MOE=$EXL3_FUSED_MOE SPEC_METHOD=$SPEC_METHOD MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
