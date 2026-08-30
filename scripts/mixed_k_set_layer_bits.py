#!/usr/bin/env python3
"""Declare per-layer K in a mixed-K checkpoint's config.

The EXL3 plugin allocates trellis parameters from `bits`, so layers stored at a
different K fail the load shape check unless `quantization_config.layer_bits`
names them. This writes that map into the given checkpoint's config.json and
quantization_config.json. It refuses to touch a directory passed as --base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="merged mixed-K directory")
    ap.add_argument("--bits", type=int, required=True, help="K of the delta layers")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--base", help="base dir; refused as --output for safety")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    if args.base and Path(args.base).resolve() == out:
        raise SystemExit("refusing to modify the base checkpoint")
    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    layer_bits = {str(i): args.bits for i in layers}

    touched = []
    for name in ("config.json", "quantization_config.json"):
        p = out / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        qc = d.get("quantization_config") if name == "config.json" else d
        if qc is None:
            raise SystemExit(f"{p}: no quantization_config")
        if str(qc.get("quant_method", "")).lower() != "exl3":
            raise SystemExit(f"{p}: quant_method is not exl3")
        qc["layer_bits"] = layer_bits
        qc["mixed_k_note"] = (
            f"base bits={qc.get('bits')}; layers {layers} are K{args.bits}. "
            "Requires the glm53_exl3_vllm_plugin with per-layer K support."
        )
        p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        touched.append(name)
    if not touched:
        raise SystemExit(f"no config files found in {out}")

    rep = out / "mixed_k_merge_report.json"
    if rep.is_file():
        r = json.loads(rep.read_text(encoding="utf-8"))
        r["layer_bits"] = layer_bits
        rep.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        touched.append(rep.name)
    print("layer_bits written:", layer_bits, "->", ", ".join(touched))


if __name__ == "__main__":
    main()
