#!/usr/bin/env python3
"""Answer 'can this environment serve the pack?' in under a second.

Run this BEFORE downloading 91 GiB or installing anything. Every check prints
the exact next command, because the failure this exists to prevent is an
operator or agent spending hours on a build and only then discovering that
stock vLLM has neither EXL3 nor Glm5Next.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

WHEEL_INDEX = "https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm"
INSTALL_CMD = "bash scripts/install_prebuilt.sh"


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []
        self.fatal = False

    def check(self, name: str, ok: bool, detail: str, fatal: bool = True) -> bool:
        self.rows.append((name, ok, detail))
        if not ok and fatal:
            self.fatal = True
        return ok

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        for name, ok, detail in self.rows:
            print(f"  [{'OK ' if ok else 'X  '}] {name.ljust(width)}  {detail}")
        return 1 if self.fatal else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "models" / "GLM-5.3-Flash-EXL3-K2",
        help="checked only if it already exists; preflight never requires the weights",
    )
    args = parser.parse_args()
    r = Result()

    print("GLM-5.3-Flash EXL3 K2 preflight\n")

    machine = platform.machine()
    r.check("arch aarch64", machine == "aarch64", machine)

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    r.check("python 3.12", sys.version_info[:2] == (3, 12), py)
    ninja = shutil.which("ninja") or (
        os.path.join(sys.prefix, "bin", "ninja")
        if os.path.exists(os.path.join(sys.prefix, "bin", "ninja")) else None
    )
    r.check(
        "ninja available",
        ninja is not None,
        ninja or "not found: FlashInfer JIT runs ninja; pip install ninja into the venv "
        "or put the venv's bin/ on PATH",
    )
    nvcc = shutil.which("nvcc")
    r.check(
        "nvcc on PATH",
        nvcc is not None,
        nvcc or "not found: vLLM's has_flashinfer() then rejects FLASHINFER_MLA_SPARSE_SM120 "
        "and serve fails with 'No valid attention backend found'. "
        "export PATH=/usr/local/cuda-13.0/bin:$PATH",
    )

    try:
        import torch
    except ImportError:
        r.check("torch importable", False, "not installed")
        torch = None
    else:
        cuda = torch.version.cuda or "none"
        r.check("torch CUDA 13", cuda.startswith("13."), f"{torch.__version__} cuda={cuda}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            r.check(
                "GB10 SM121",
                cap == (12, 1),
                f"{torch.cuda.get_device_name()} capability {cap[0]}.{cap[1]}",
            )
        else:
            r.check("CUDA available", False, "torch.cuda.is_available() is False")

    # The two checks that the three-hour failure mode is actually about.
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None:
        r.check("vllm importable", False, "not installed")
    else:
        origin = vllm_spec.origin or ""
        glm5 = importlib.util.find_spec("vllm.models.glm5next") is not None
        r.check(
            "vllm has Glm5Next",
            glm5,
            "present" if glm5 else "MISSING -> this is stock vLLM, not the recipe build",
        )
        try:
            from vllm.plugins import load_general_plugins

            load_general_plugins()
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

            exl3 = "exl3" in QUANTIZATION_METHODS
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            exl3 = False
            r.check("exl3 quantization registered", False, f"plugin load failed: {exc!r}")
        else:
            r.check(
                "exl3 quantization registered",
                exl3,
                "present" if exl3 else "MISSING -> the EXL3 plugin is not installed (pip install 'vllm-exl3>=0.3.1')",
            )
        r.check("vllm location", True, os.path.dirname(origin), fatal=False)

    exl3_spec = importlib.util.find_spec("vllm_exl3") or importlib.util.find_spec("glm53_exl3_plugin")
    r.check(
        "vllm-exl3 plugin",
        exl3_spec is not None,
        "installed" if exl3_spec else "MISSING -> pip install 'vllm-exl3>=0.3.1'",
    )

    try:
        import exllamav3_ext
    except ImportError:
        r.check("exllamav3_ext", False, "not installed")
    else:
        fused = hasattr(exllamav3_ext, "exl3_moe")
        r.check("fused exl3_moe kernel", fused, "present" if fused else "MISSING")

    model_dir: Path = args.model_dir
    if model_dir.exists():
        shards = len(list(model_dir.glob("*.safetensors")))
        total = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
        ok = shards == 120 and total == 97_728_721_536
        r.check(
            "pack complete",
            ok,
            f"{shards}/120 shards, {total:,} bytes"
            + ("" if ok else " (expected 120 and 97,728,721,536)"),
            fatal=True,
        )
    else:
        r.check("pack present", True, f"{model_dir} not downloaded yet", fatal=False)

    code = r.report()
    print()
    if code:
        print("PREFLIGHT FAILED. Do not start a build or a download yet.\n")
        print("Almost always the fix is the prebuilt runtime, which takes minutes:\n")
        print(f"    {INSTALL_CMD}\n")
        print(
            "Do NOT run `pip install vllm`. Stock vLLM has neither the EXL3\n"
            "quantization method nor the Glm5Next architecture, and no flag\n"
            "turns them on. Both come from this recipe's runtime.\n"
            f"Wheels: {WHEEL_INDEX}"
        )
    else:
        print("PREFLIGHT PASSED. This environment can serve the pack.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
