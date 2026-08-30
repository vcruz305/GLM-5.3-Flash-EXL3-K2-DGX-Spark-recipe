#!/usr/bin/env python3
"""Capture PRE-final-norm hidden states from a vLLM-served checkpoint on the
fidelity suite's token windows, for scoring with score_kld.py.

Runs vLLM offline in eager mode with a forward pre-hook on the language
model's final RMSNorm, one context per forward (2048 prompt tokens, 1 generated
token discarded), and writes hidden_XXXX.safetensors [2047, 4096] bf16 in the
suite's reference layout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("EXL3_FUSED_MOE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
# apply_model() ships Python functions to the engine core. Keep the engine in
# this process so the hook and the captured tensors never cross a process
# boundary (and never need pickle); the second flag is the fallback if the
# runner ignores the first.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _ensure_nvcc() -> None:
    """vLLM's has_flashinfer() needs nvcc on PATH or it rejects the only sparse-MLA
    backend for SM121 ("No valid attention backend found"). The engine core is a
    subprocess and inherits this PATH."""
    import glob
    import shutil
    import sys

    # FlashInfer's JIT also needs ninja, which pip put in this venv's bin/.
    venv_bin = os.path.join(sys.prefix, "bin")
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("nvcc"):
        return
    for cand in sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True):
        os.environ["PATH"] = os.path.dirname(cand) + os.pathsep + os.environ.get("PATH", "")
        print(f"nvcc was not on PATH; added {os.path.dirname(cand)}", flush=True)
        return
    raise SystemExit(
        "nvcc not on PATH and no /usr/local/cuda*/bin/nvcc found; vLLM would reject "
        "FLASHINFER_MLA_SPARSE_SM120. export PATH=/usr/local/cuda-13.0/bin:$PATH"
    )


_ensure_nvcc()

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _install_hook(model):
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    norm = inner.norm
    model._kld_capture = []

    def pre(mod, args):
        model._kld_capture.append(args[0].detach().to(torch.bfloat16).cpu())

    model._kld_hook = norm.register_forward_pre_hook(pre)
    return "hook installed on " + type(norm).__name__


def _drain(model):
    out = model._kld_capture
    model._kld_capture = []
    return out


def _layer_k(model):
    """Per-layer EXL3 K as the loaded plugin will dispatch it (RoutedExperts._exl3_k)."""
    ks = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "_exl3_k") and ".layers." in name:
            ks[int(name.split(".layers.")[1].split(".")[0])] = int(mod._exl3_k)
    return ks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=512, help="contexts; shard0 has 512")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokdir = Path(args.suite) / "suite/tokens"
    llm = LLM(
        model=args.model,
        quantization="exl3",
        kv_cache_dtype="fp8",
        max_model_len=4096,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        skip_mm_profiling=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    print(llm.apply_model(_install_hook)[0], flush=True)

    ks = llm.apply_model(_layer_k)[0]
    print("EXL3_LAYER_K", json.dumps({str(k): v for k, v in sorted(ks.items())}), flush=True)
    print("EXL3_LAYER_K_SUMMARY", json.dumps({f"K{v}": sorted(k for k, kk in ks.items() if kk == v) for v in sorted(set(ks.values()))}), flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    done = 0
    for i in range(args.start, args.start + args.limit):
        dst = out / f"hidden_{i:04d}.safetensors"
        if dst.is_file():
            continue
        ids = json.loads((tokdir / f"context-{i:04d}.json").read_text())
        assert len(ids) == 2048, (i, len(ids))
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        caps = llm.apply_model(_drain)[0]
        # The prompt forward may be chunked; concatenate in order and keep the
        # 2047 scored positions (predictions for tokens 1..2047).
        h = torch.cat(caps, 0)[:2048]
        assert h.shape[0] == 2048, h.shape
        save_file({"hidden_states": h[:2047].contiguous()}, str(dst))
        done += 1
        if done % 16 == 0:
            print(f"captured {done} contexts", flush=True)
    print(f"CAPTURE_DONE {done} new, dir={out}")


if __name__ == "__main__":
    main()
