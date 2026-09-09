#!/usr/bin/env bash
# Install the unreleased vllm-exl3 candidate used by TP1 policy experiments.
# Use an existing compatible runtime where possible. For a fresh environment,
# establish the recipe runtime first. Preserve a rollback artifact before use.
set -euo pipefail

VENV="${VENV:-${HOME}/venvs/glm53-exl3-local}"
PYTHON="${VENV}/bin/python"
VLLM_EXL3_REPO="${VLLM_EXL3_REPO:-https://github.com/vcruz305/vllm-exl3.git}"
# CPU/source/packaging CI passed at this exact ref; GPU qualification is pending.
VLLM_EXL3_REF="${VLLM_EXL3_REF:-d3cfd394920360d69f820d2dc96f8292a9e10283}"
REF_RECEIPT="${VENV}/.vllm-exl3-candidate-ref"

if [[ ! "$VLLM_EXL3_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "VLLM_EXL3_REF must be a full 40-character Git commit SHA" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing ${PYTHON}; establish the compatible recipe runtime first" >&2
  exit 1
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "candidate native kernels are being qualified on aarch64 / DGX Spark; got $(uname -m)" >&2
  exit 1
fi
export PATH="${VENV}/bin:${PATH}"
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; break; fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; add the CUDA 13 toolkit to PATH before building the candidate" >&2
  exit 1
fi
if [[ "${VLLM_EXL3_NO_CUDA:-0}" == "1" ]]; then
  echo "VLLM_EXL3_NO_CUDA=1 would skip the native build; unset it for qualification" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("the prebuilt GB10 recipe requires Python 3.12")
missing = [name for name in ("torch", "exllamav3") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("candidate prerequisites missing: " + ", ".join(missing))
PY

echo "Installing vllm-exl3 candidate from exact ref: ${VLLM_EXL3_REF}"
"$PYTHON" -m pip install --no-build-isolation --no-deps --force-reinstall \
  "git+${VLLM_EXL3_REPO}@${VLLM_EXL3_REF}"

"$PYTHON" - "$VLLM_EXL3_REF" <<'PY'
import importlib.metadata
import json
import sys
import torch
import vllm_exl3
import vllm_exl3_c
if getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", 0) < 2:
    raise SystemExit("native MoE ABI 2 or newer is required; inspect/rebuild the extension")
vllm_exl3.register()
print("vllm-exl3 version:", importlib.metadata.version("vllm-exl3"))
print("candidate source ref:", sys.argv[1])
print("native extension:", vllm_exl3_c.__file__)
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
PY
printf '%s\n' "$VLLM_EXL3_REF" > "$REF_RECEIPT"

echo
echo "Candidate installed; this is not a GPU performance qualification."
echo "Re-run preflight and restart only the test server before benchmarking."
echo "Follow docs/TP1_POLICY_AB.md; the legacy schema-1 harness needs validation."
