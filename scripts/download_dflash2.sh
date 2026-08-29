#!/usr/bin/env bash
# Optional BF16 DFlash2 sidecar used by the speculative-decoding ladder.
set -euo pipefail

MODEL_ID="${MODEL_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DEST="${DEST:-${HOME}/models/GLM-5.3-Flash-DFlash2}"
REVISION="${REVISION:-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}"
EXPECTED_WEIGHT_SHA256="${EXPECTED_WEIGHT_SHA256:-b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Install Hugging Face CLI: python3 -m pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DEST"
hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$DEST"

python3 - "$DEST" "$REVISION" "$EXPECTED_WEIGHT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
expected_weight_sha256 = sys.argv[3]
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
architectures = config.get("architectures") or []
if "DFlash2DraftModel" not in architectures:
    raise SystemExit(f"unexpected draft architecture: {architectures}")
shards = list(root.glob("*.safetensors"))
if len(shards) != 1 or shards[0].name != "model.safetensors":
    raise SystemExit(f"unexpected checkpoint files: {[p.name for p in shards]}")
digest = hashlib.sha256()
with shards[0].open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_weight_sha256 = digest.hexdigest()
if actual_weight_sha256 != expected_weight_sha256:
    raise SystemExit(
        f"DFlash2 weight hash mismatch: expected {expected_weight_sha256}, "
        f"found {actual_weight_sha256}"
    )
print("revision", revision)
print("architecture", architectures)
print("safetensors", len(shards), "bytes", sum(p.stat().st_size for p in shards))
print("model.safetensors sha256", actual_weight_sha256)
PY
