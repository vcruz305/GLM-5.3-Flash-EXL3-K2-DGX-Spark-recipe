#!/usr/bin/env python3
"""Score a hidden-state capture against the fidelity suite's BF16 reference.

Replays both operands through the ONE shared head (final RMSNorm + lm_head) and
computes exact full-vocab KL(reference || candidate) per position in fp32, the
same construction as malaiwah/GLM-5.3-Flash-fidelity-suite-v1. Reports token
mean KLD, context macro mean with a 95% CI, median/p99/p999, and top-1
agreement.

Inputs per context i: <suite>/reference-bf16-shard0/hidden_XXXX.safetensors and
<capture>/hidden_XXXX.safetensors, both holding tensor "hidden_states" of shape
[2047, 4096] bf16: the PRE-final-norm residual stream at positions 0..2046.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * w.float()


@torch.no_grad()
def kld_one(h_ref, h_cand, w_norm, w_head, eps, dev, chunk=256):
    """Per-position KL(ref||cand) and top-1 agreement in fp32, chunked."""
    kls, agree = [], []
    head = w_head.to(dev, torch.bfloat16)
    wn = w_norm.to(dev)
    for s in range(0, h_ref.shape[0], chunk):
        a = rmsnorm(h_ref[s : s + chunk].to(dev), wn, eps).to(torch.bfloat16) @ head.T
        b = rmsnorm(h_cand[s : s + chunk].to(dev), wn, eps).to(torch.bfloat16) @ head.T
        la = torch.log_softmax(a.float(), -1)
        lb = torch.log_softmax(b.float(), -1)
        kls.append((la.exp() * (la - lb)).sum(-1).cpu())
        agree.append((la.argmax(-1) == lb.argmax(-1)).float().cpu())
    return torch.cat(kls), torch.cat(agree)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, help="local dir of the fidelity suite")
    ap.add_argument("--capture", required=True, help="dir of hidden_XXXX.safetensors")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    suite = Path(args.suite)
    cap = Path(args.capture)
    w_head = load_file(str(suite / "head/head.safetensors"))["weight"]
    w_norm = load_file(str(suite / "head/final_norm.safetensors"))["weight"]
    refs = sorted((suite / "reference-bf16-shard0").glob("hidden_*.safetensors"))
    if args.limit:
        refs = refs[: args.limit]

    per_ctx, all_kl, all_top1 = [], [], []
    for r in refs:
        c = cap / r.name
        if not c.is_file():
            continue
        h_ref = load_file(str(r))["hidden_states"]
        h_cand = load_file(str(c))["hidden_states"]
        n = min(h_ref.shape[0], h_cand.shape[0])
        kl, top1 = kld_one(h_ref[:n], h_cand[:n], w_norm, w_head, args.eps, args.device)
        per_ctx.append(
            {"context": r.stem, "n": n, "mean_kld": kl.mean().item(), "top1": top1.mean().item()}
        )
        all_kl.append(kl)
        all_top1.append(top1)
        print(
            f"{r.stem}: n={n} mean_kld={kl.mean().item():.6f} top1={top1.mean().item():.4f}",
            flush=True,
        )

    if not all_kl:
        raise SystemExit("no matching capture files")
    kl = torch.cat(all_kl)
    top1 = torch.cat(all_top1)
    ctx_means = torch.tensor([c["mean_kld"] for c in per_ctx])
    macro = ctx_means.mean().item()
    ci = (
        1.96 * ctx_means.std(unbiased=True).item() / math.sqrt(len(ctx_means))
        if len(ctx_means) > 1
        else float("nan")
    )
    report = {
        "label": args.label,
        "contexts": len(per_ctx),
        "positions": int(kl.numel()),
        "token_mean_kld": kl.mean().item(),
        "context_macro_mean_kld": macro,
        "ci95": [macro - ci, macro + ci],
        "median_kld": kl.median().item(),
        "p99_kld": kl.quantile(0.99).item(),
        "p999_kld": kl.quantile(0.999).item(),
        "top1_agreement": top1.mean().item(),
        "per_context": per_ctx,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_context"}, indent=1))


if __name__ == "__main__":
    main()
