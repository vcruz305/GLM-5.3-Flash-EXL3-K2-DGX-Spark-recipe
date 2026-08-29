#!/usr/bin/env python3
"""Fail fast unless the local Spark runtime has every required component."""
from __future__ import annotations

from importlib.metadata import version

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise SystemExit(f"expected GB10 SM121, got capability {capability}")

    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    if "exl3" not in QUANTIZATION_METHODS:
        raise SystemExit("EXL3 plugin did not register with vLLM")

    import exllamav3_ext
    import vllm

    if not hasattr(exllamav3_ext, "exl3_moe"):
        raise SystemExit("ExLlamaV3 extension has no fused exl3_moe entry point")

    print("device", torch.cuda.get_device_name(), "capability", capability)
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("vllm", vllm.__version__)
    print("exllamav3", version("exllamav3"))
    print("flashinfer-python", version("flashinfer-python"))
    print("exl3_moe concurrency", exllamav3_ext.exl3_moe_max_concurrency(0))


if __name__ == "__main__":
    main()
