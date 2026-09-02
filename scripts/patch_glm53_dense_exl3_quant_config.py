#!/usr/bin/env python3
"""Let the GLM-5.3 fork's attention projections see the quant config when the
pack declares dense EXL3 tensors for them (vllm-exl3 ``non_routed_exl3``).

Two exact-match, idempotent edits under ``vllm/models/glm5next/nvidia/``:
kda.py nulls ``vllm_config.quant_config`` around the KDA attention constructor
and model.py passes ``quant_config=None`` to the MLA attention (both because
fp8 checkpoints ship no scales for these projections). With the guard, packs
without ``non_routed_exl3`` behave exactly as before.
Usage: python patch_glm53_dense_exl3_quant_config.py [path/to/site-packages/vllm]
"""
import shutil, sys
from pathlib import Path

KDA_OLD = (
    "        saved_quant_config = vllm_config.quant_config\n"
    "        try:\n"
    "            vllm_config.quant_config = None\n"
)
KDA_NEW = (
    "        saved_quant_config = vllm_config.quant_config\n"
    "        try:\n"
    "            # Keep the quant config when the pack declares dense EXL3 tensors\n"
    "            # for these projections (vllm-exl3 non_routed_exl3); unmatched\n"
    "            # linears still resolve to the unquantized method.\n"
    "            if not getattr(saved_quant_config, \"non_routed_exl3\", None):\n"
    "                vllm_config.quant_config = None\n"
)
MLA_OLD = (
    "                quant_config=None,  # MLA projections are BF16 in checkpoint\n"
)
MLA_NEW = (
    "                # MLA projections are BF16 unless the pack declares dense EXL3\n"
    "                quant_config=(\n"
    "                    quant_config\n"
    "                    if getattr(quant_config, \"non_routed_exl3\", None)\n"
    "                    else None\n"
    "                ),\n"
)


def patch(path, old, new):
    s = path.read_text()
    if s.count(new) == 1:
        return "already patched"
    if s.count(old) != 1:
        raise SystemExit(f"{path}: anchor count {s.count(old)} != 1")
    orig = path.with_suffix(path.suffix + ".orig-p2")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(s.replace(old, new))
    return "patched"


root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if root is None:
    import vllm
    root = Path(vllm.__file__).resolve().parent
d = root / "models" / "glm5next" / "nvidia"
print("kda.py:", patch(d / "kda.py", KDA_OLD, KDA_NEW))
print("model.py:", patch(d / "model.py", MLA_OLD, MLA_NEW))
import ast
for f in ("kda.py", "model.py"):
    ast.parse((d / f).read_text())
print("P2_FORK_PATCH_OK")
