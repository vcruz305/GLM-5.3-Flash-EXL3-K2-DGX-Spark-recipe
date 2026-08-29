#!/usr/bin/env python3
"""If vLLM's EXL3 loader rejects bits=2, extend the allowlist in-place.

K2 is 2-bit MCG trellis on routed experts. Some EXL3 loaders only allow 3–6.
This is a one-line allowlist edit, not a kernel rewrite.

Usage:
  python scripts/patch_exl3_bits2.py
  python scripts/patch_exl3_bits2.py /path/to/vllm/model_executor/layers/quantization/exl3.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


OLD = "if self.bits not in (3, 4, 5, 6):"
NEW = "if self.bits not in (2, 3, 4, 5, 6):"


def find_default() -> Path | None:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    except Exception:
        return None
    candidates = [
        root / "model_executor" / "layers" / "quantization" / "exl3.py",
        root / "vllm" / "model_executor" / "layers" / "quantization" / "exl3.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.path or find_default()
    if path is None or not path.is_file():
        print("Could not find vLLM exl3.py. Pass the path explicitly.", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"already allows bits=2: {path}")
        return 0
    if OLD not in text:
        print(f"allowlist pattern not found in {path}", file=sys.stderr)
        print("Open the file and ensure bits=2 is accepted.", file=sys.stderr)
        return 2
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched bits allowlist to (2, 3, 4, 5, 6): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
