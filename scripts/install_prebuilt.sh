#!/usr/bin/env bash
# Install the tested runtime from prebuilt wheels. Minutes, not hours.
#
# This is the default install path. scripts/install_local_runtime.sh builds the
# same runtime from source and is only needed if you are changing the patches
# or running on a Python/CUDA combination these wheels were not built for.
set -euo pipefail

VENV="${VENV:-${HOME}/venvs/glm53-exl3-local}"
WHEEL_DIR="${WHEEL_DIR:-${HOME}/glm53-runtime-wheels}"
WHEEL_REPO="${WHEEL_REPO:-vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

# These wheels contain compiled CUDA extensions. They are not portable across
# architecture or Python minor version, so refuse early rather than fail deep
# inside an import.
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "These wheels are aarch64 (DGX Spark / GB10). Got $(uname -m)." >&2
  echo "On another architecture, build from source: scripts/install_local_runtime.sh" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Creating venv at ${VENV}"
  python3 -m venv "$VENV"
fi
PYTHON="${VENV}/bin/python"

PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYVER" != "3.12" ]]; then
  echo "These wheels are cp312. This venv is Python ${PYVER}." >&2
  echo "Recreate the venv with python3.12, or build from source." >&2
  exit 1
fi

"$PYTHON" -m pip install --quiet --upgrade pip

# PyTorch stays a separate install: its distribution channel changes
# independently of this recipe, and the CUDA 13 build is a hard requirement.
if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
  echo "Installing CUDA 13 PyTorch"
  "$PYTHON" -m pip install --index-url "$TORCH_INDEX" \
    "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"
fi

if [[ ! -d "$WHEEL_DIR" ]] || ! compgen -G "${WHEEL_DIR}/*.whl" >/dev/null; then
  echo "Fetching prebuilt wheels into ${WHEEL_DIR}"
  if ! command -v hf >/dev/null 2>&1; then
    "$PYTHON" -m pip install --quiet "huggingface_hub[cli]"
    export PATH="${VENV}/bin:${PATH}"
  fi
  hf download "$WHEEL_REPO" --local-dir "$WHEEL_DIR"
fi

echo "Installing runtime wheels"
"$PYTHON" -m pip install "${WHEEL_DIR}"/*.whl

# FlashInfer is a normal published wheel; no patching needed.
"$PYTHON" -m pip install --pre --upgrade "flashinfer-python==0.6.18rc10"

# Canonical routed-expert EXL3 plugin with native Blackwell sm_121 kernels
"$PYTHON" -m pip install "vllm-exl3>=0.3.1"

echo
"$PYTHON" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preflight.py" || {
  echo "Preflight failed after install. Report this with the output above." >&2
  exit 1
}
echo
echo "Runtime ready: ${VENV}"
echo "Next: bash scripts/download_weights.sh"
