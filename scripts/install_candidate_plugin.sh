#!/usr/bin/env bash
# Install the unreleased vllm-exl3 candidate used by TP1 policy experiments.
# Run scripts/install_prebuilt.sh first so the vLLM fork, PyTorch, ExLlamaV3,
# CUDA toolchain dependencies, and venv are already present.
set -euo pipefail

VENV="${VENV:-${HOME}/venvs/glm53-exl3-local}"
PYTHON="${VENV}/bin/python"
VLLM_EXL3_REPO="${VLLM_EXL3_REPO:-https://github.com/vcruz305/vllm-exl3.git}"
# Exact source boundary qualified by this recipe. Override only intentionally.
VLLM_EXL3_REF="${VLLM_EXL3_REF:-28041c423a81fe033e7128d8888e3762fb914910}"
REF_RECEIPT="${VENV}/.vllm-exl3-candidate-ref"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing ${PYTHON}; run bash scripts/install_prebuilt.sh first" >&2
  exit 1
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "candidate native kernels are being qualified on aarch64 / DGX Spark; got $(uname -m)" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; break; fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; add the CUDA 13 toolkit to PATH before building the candidate" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "exllamav3") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("candidate prerequisites missing: " + ", ".join(missing))
PY

echo "Installing vllm-exl3 candidate from exact ref: ${VLLM_EXL3_REF}"
"$PYTHON" -m pip install --no-build-isolation --force-reinstall \
  "git+${VLLM_EXL3_REPO}@${VLLM_EXL3_REF}"
printf '%s\n' "$VLLM_EXL3_REF" > "$REF_RECEIPT"

"$PYTHON" - <<PY
import importlib.metadata, json
import vllm_exl3
print("vllm-exl3 version:", importlib.metadata.version("vllm-exl3"))
print("candidate source ref: ${VLLM_EXL3_REF}")
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
PY

echo
echo "Candidate installed. Re-run: python scripts/preflight.py"
echo "Then benchmark with: python scripts/bench_tp1_policy.py --label candidate-default"
