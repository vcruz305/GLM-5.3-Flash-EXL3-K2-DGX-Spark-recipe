#!/usr/bin/env bash
# Download vcruz305/GLM-5.3-Flash-EXL3-K2. Resume with --local-dir only.
# There is no --resume-download flag on current `hf` CLI.
set -euo pipefail

MODEL_ID="${MODEL_ID:-vcruz305/GLM-5.3-Flash-EXL3-K2}"
DEST="${DEST:-${HOME}/models/GLM-5.3-Flash-EXL3-K2}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"
EXPECTED_BYTES="${EXPECTED_BYTES:-97728721536}"

mkdir -p "$DEST"

if ! command -v hf >/dev/null 2>&1; then
  echo "Install Hugging Face CLI: python3 -m pip install -U huggingface_hub"
  exit 1
fi

base="$(basename "$DEST")"
if [[ "$base" != "GLM-5.3-Flash-EXL3-K2" && "$base" != "GLM-5.3-Flash-EXL3-K2K3-mix" ]]; then
  echo "DEST basename should be GLM-5.3-Flash-EXL3-K2 or GLM-5.3-Flash-EXL3-K2K3-mix, got: $base"
  exit 1
fi

echo "Downloading $MODEL_ID -> $DEST"
hf download "$MODEL_ID" --local-dir "$DEST"

export DEST EXPECTED_SHARDS EXPECTED_BYTES
python3 - <<'PY'
import glob, os, sys
dest = os.environ["DEST"]
shards = sorted(glob.glob(os.path.join(dest, "*.safetensors")))
if not shards:
    shards = sorted(glob.glob(os.path.join(dest, "**", "*.safetensors"), recursive=True))
total = sum(os.path.getsize(p) for p in shards)
exp_n = int(os.environ["EXPECTED_SHARDS"])
exp_b = int(os.environ["EXPECTED_BYTES"])
print(f"safetensors files: {len(shards)}")
print(f"safetensors bytes: {total}")
print(f"expected shards:  {exp_n}")
print(f"expected bytes:   {exp_b}")
if len(shards) < exp_n:
    sys.exit(f"incomplete pack: {len(shards)} shards")
if total != exp_b:
    print("WARNING: byte total does not match the measured 91.017 GiB pack")
print("download check done")
PY
