#!/usr/bin/env python3
"""Keep DFlash2 attention native when applying a draft quantization config.

DFlash2 builds its context KV projection by directly slicing the dense QKV
weight. vLLM's generic draft quantization config otherwise reaches QKV/O as
well as the MLP and auxiliary FC. This patch limits DFlash2 online weight
quantization to the auxiliary FC and five MLPs while leaving attention BF16.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count == 1:
        path.write_text(source.replace(old, new), encoding="utf-8")
        return
    if count == 0 and new in source:
        return
    raise RuntimeError(f"{path}: expected one patch target, found {count}: {old!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="vLLM source checkout")
    args = parser.parse_args()
    target = args.source / "vllm/model_executor/models/qwen3_dflash.py"

    replace_once(
        target,
        "        dflash_config = getattr(config, \"dflash_config\", None) or {}\n"
        "        add_swa_attention_sink_bias = dflash_config.get(\n",
        "        dflash_config = getattr(config, \"dflash_config\", None) or {}\n"
        "        # DFlash2's fused context-KV builder slices qkv_proj.weight\n"
        "        # directly. Keep QKV/O native and apply the draft quantization\n"
        "        # config only to the MLPs and the model-level auxiliary FC.\n"
        "        attention_quant_config = (\n"
        "            None if \"conv_kernel_size\" in dflash_config else quant_config\n"
        "        )\n"
        "        add_swa_attention_sink_bias = dflash_config.get(\n",
    )
    replace_once(
        target,
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
        "            rope_parameters=config.rope_parameters,\n",
        "            cache_config=cache_config,\n"
        "            quant_config=attention_quant_config,\n"
        "            rope_parameters=config.rope_parameters,\n",
    )

    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print("DFlash2 selective draft-quantization source patch verified")


if __name__ == "__main__":
    main()
