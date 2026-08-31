#!/usr/bin/env python3
"""Standalone micro-repro for the deep-context KDA wedge.

Calls chunk_kda_with_fused_gate exactly as the GLM-5.3 KDA layer does, at a
given T, on random data. Run each T in a subprocess with a timeout: OK means
the call returned; HANG means the kill-guard fired.

Usage: kda_micro.py <T>            (child: run one length)
       kda_micro.py --matrix a,b,c (parent: subprocess per T, 150 s guard)
"""
from __future__ import annotations

import subprocess
import sys
import time


def run_one(T: int) -> None:
    import torch

    sys.path.insert(0, "/home/markus/src/vllm-glm53")
    from vllm.third_party.flash_linear_attention.ops.kda import (
        chunk_kda_with_fused_gate,
    )

    torch.manual_seed(0)
    dev = "cuda"
    B, H, D = 1, 64, 128
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    raw_g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=dev)
    A_log = torch.zeros(H, dtype=torch.float32, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    t0 = time.perf_counter()
    o, state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        A_log=A_log,
        g_bias=None,
        initial_state=None,
        output_final_state=True,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(
        f"T={T} OK in {dt:.1f}s out={tuple(o.shape)} finite={bool(torch.isfinite(o.float()).all())}",
        flush=True,
    )


def main() -> None:
    if sys.argv[1] == "--matrix":
        ts = [int(x) for x in sys.argv[2].split(",")]
        for T in ts:
            t0 = time.perf_counter()
            p = subprocess.Popen(
                [sys.executable, __file__, str(T)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                out, _ = p.communicate(timeout=150)
                print(out.strip(), flush=True)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                print(
                    f"T={T} HANG (killed after {time.perf_counter()-t0:.0f}s)",
                    flush=True,
                )
        print("MATRIX_DONE", flush=True)
    else:
        run_one(int(sys.argv[1]))


if __name__ == "__main__":
    main()
