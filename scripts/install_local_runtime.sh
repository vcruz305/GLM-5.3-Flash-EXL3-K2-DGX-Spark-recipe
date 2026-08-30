#!/usr/bin/env bash
# Build the tested vLLM + ExLlamaV3 runtime directly on an ARM64 DGX Spark.
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${HOME}/src/glm53-exl3-runtime}"
VENV="${VENV:-${HOME}/venvs/glm53-exl3-local}"
VLLM_SRC="${VLLM_SRC:-${RUNTIME_ROOT}/vllm-glm53}"
EXLLAMAV3_SRC="${EXLLAMAV3_SRC:-${RUNTIME_ROOT}/exllamav3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
VLLM_REV="${VLLM_REV:-878631b6079d2cf9fb80830ef9cb41b43aded098}"
EXLLAMAV3_REV="${EXLLAMAV3_REV:-17bc3923259ffd48aab742edd261a0ca45d55459}"
MAX_JOBS="${MAX_JOBS:-12}"
NVCC_THREADS="${NVCC_THREADS:-1}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer is for the ARM64 DGX Spark; got $(uname -m)." >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT" "$(dirname "$VENV")"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

PYTHON="${VENV}/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel cmake ninja packaging

# The CUDA 13 PyTorch wheel is intentionally a prerequisite: its distribution
# channel changes independently of this recipe, and silently installing a CPU
# or older-CUDA wheel here would make a 22-minute vLLM build unusable.
"$PYTHON" - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Install the NVIDIA/CUDA 13 PyTorch wheel in this venv first "
        "(tested: torch 2.13.0+cu130, torchvision 0.28.0+cu130)."
    ) from exc
if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
    raise SystemExit(f"CUDA 13 PyTorch required; found torch={torch.__version__} cuda={torch.version.cuda}")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

checkout() {
  local url="$1" path="$2" rev="$3"
  if [[ ! -d "${path}/.git" ]]; then
    git clone --filter=blob:none "$url" "$path"
  fi
  if [[ "$(git -C "$path" rev-parse HEAD)" != "$rev" ]]; then
    if [[ -n "$(git -C "$path" status --short)" ]]; then
      echo "Refusing to switch a dirty checkout: $path" >&2
      exit 1
    fi
    git -C "$path" fetch origin "$rev"
    git -C "$path" checkout --detach "$rev"
  fi
}

checkout https://github.com/ZJY0516/vllm.git "$VLLM_SRC" "$VLLM_REV"
checkout https://github.com/turboderp-org/exllamav3.git "$EXLLAMAV3_SRC" "$EXLLAMAV3_REV"

"$PYTHON" "$RECIPE_ROOT/scripts/patch_exllamav3_aarch64.py" \
  "$EXLLAMAV3_SRC/exllamav3/exllamav3_ext"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${VENV}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export MAX_JOBS NVCC_THREADS

"$PYTHON" -m pip install --no-build-isolation "$EXLLAMAV3_SRC"

"$PYTHON" "$RECIPE_ROOT/scripts/patch_glm53_sm121_nope.py" --source "$VLLM_SRC"
"$PYTHON" "$RECIPE_ROOT/scripts/patch_glm53_eagle3.py" --source "$VLLM_SRC"
"$PYTHON" "$RECIPE_ROOT/scripts/patch_dflash2_selective_quant.py" --source "$VLLM_SRC"
# Opt-in K-pool tail bounds detector. Inert unless GLM_KPOOL_TAIL_BOUNDS=1
# is set on the server. Shipping it means anyone can answer "is my build
# affected?" with a number instead of waiting for a crash.
"$PYTHON" "$RECIPE_ROOT/scripts/patch_kpool_tail_detector.py" --source "$VLLM_SRC"
"$PYTHON" -m pip install --no-build-isolation --editable "$VLLM_SRC"
"$PYTHON" -m pip install --pre --upgrade "flashinfer-python==0.6.18rc10"
"$PYTHON" -m pip install --editable "$RECIPE_ROOT/runtime/exl3_plugin"

"$PYTHON" "$RECIPE_ROOT/scripts/verify_runtime.py"
echo "Local runtime ready: $VENV"
