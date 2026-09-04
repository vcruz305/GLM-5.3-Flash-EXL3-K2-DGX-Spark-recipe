#!/usr/bin/env python3
"""Fix the >163k prefill wedge by raising the EXL3 fused-MoE per-expert row cap.

Past ~163,840 prompt tokens the router concentrates more than 128 rows onto a
single expert within a 2,048-token prefill chunk. The fused exl3_moe kernel caps
at TEMP_ROWS_FUSED = 128 rows per expert, so a "fat" expert falls back to the
slow per-expert LinearEXL3 reconstruct (apply_exl3_python_loop), which takes
minutes per chunk and reads as an engine hang -- a latency cliff, not a deadlock.

Raising the cap to 2048 means no expert in a 2,048-token batch can be fat, so the
fused kernel always handles every expert directly. Idempotent and exact-match:
aborts if the source differs rather than corrupting it.

Usage: python patch_moe_fat_expert_rows.py [path/to/glm53_exl3_plugin/exl3.py]
(defaults to the installed glm53_exl3_plugin/exl3.py).
"""
import sys


def resolve_default():
    import os
    try:
        import vllm_exl3
        return os.path.join(os.path.dirname(vllm_exl3.__file__), "exl3.py")
    except ImportError:
        import glm53_exl3_plugin
        return os.path.join(os.path.dirname(glm53_exl3_plugin.__file__), "exl3.py")


path = sys.argv[1] if len(sys.argv) > 1 else resolve_default()
s = open(path).read()
if "TEMP_ROWS_FUSED = 2048" in s:
    print(f"already patched: {path} (TEMP_ROWS_FUSED = 2048)")
    sys.exit(0)
n = s.count("TEMP_ROWS_FUSED = 128")
assert n == 1, f"expected exactly 1 'TEMP_ROWS_FUSED = 128' in {path}, found {n}"
s = s.replace("TEMP_ROWS_FUSED = 128", "TEMP_ROWS_FUSED = 2048", 1)
open(path, "w").write(s)
print(f"patched {path}: TEMP_ROWS_FUSED 128 -> 2048 (fixes the >163k prefill wedge)")
