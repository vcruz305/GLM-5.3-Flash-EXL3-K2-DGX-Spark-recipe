#!/usr/bin/env python3
"""Benchmark a running OpenAI-compatible GLM TP1 server and write a policy receipt.

This intentionally uses only the Python standard library for HTTP/SSE.  Run it
against the same loaded checkpoint while changing one EXL3 policy variable at a
time.  It records local runtime diagnostics so benchmark numbers cannot become
detached from the actual dispatch configuration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time
from urllib.request import Request, urlopen

DEFAULT_PROMPT = (
    "Write a compact but technically precise explanation of how tensor parallelism, "
    "speculative decoding, and routed MoE expert execution interact during LLM inference. "
    "Use several paragraphs and include one concrete performance tradeoff."
)
TRACKED_ENV = (
    "VLLM_EXL3_MOE_KERNEL",
    "VLLM_EXL3_NATIVE_MOE_MAX_ROWS",
    "VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2",
    "VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3",
    "VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K4",
    "VLLM_EXL3_NATIVE_MOE_MEASURED_CAP",
    "VLLM_EXL3_FUSED_TEMP_ROWS",
    "VLLM_EXL3_FAT_THRESHOLD",
    "VLLM_EXL3_GROUPED_PREFILL",
    "VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS",
    "EXL3_FUSED_MOE",
    "SPEC_METHOD",
    "MTP_TOKENS",
    "MAX_MODEL_LEN",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "GPU_MEM_UTIL",
    "KV_CACHE_MEMORY_BYTES",
)


def local_runtime() -> dict[str, object]:
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        info["vllm_exl3_version"] = importlib.metadata.version("vllm-exl3")
    except importlib.metadata.PackageNotFoundError:
        info["vllm_exl3_version"] = None
    try:
        from vllm_exl3 import runtime_diagnostics

        info["vllm_exl3_policy"] = runtime_diagnostics()
    except Exception as exc:  # receipt should survive a remote-only client
        info["vllm_exl3_policy_error"] = repr(exc)
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name()
            info["cuda_capability"] = list(torch.cuda.get_device_capability())
    except Exception as exc:
        info["torch_error"] = repr(exc)
    info["environment"] = {name: os.environ[name] for name in TRACKED_ENV if name in os.environ}
    return info


def stream_once(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    finished_at = started
    usage: dict[str, object] | None = None
    chunks = 0
    content_parts: list[str] = []

    with urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    now = time.perf_counter()
                    if first_content_at is None:
                        first_content_at = now
                    chunks += 1
                    content_parts.append(text)
            finished_at = time.perf_counter()
    finished_at = time.perf_counter()

    if first_content_at is None:
        raise RuntimeError("stream completed without a content token")
    wall = finished_at - started
    decode_wall = max(finished_at - first_content_at, 1e-9)
    completion_tokens = None
    prompt_tokens = None
    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
    decode_tps = (float(completion_tokens) / decode_wall) if completion_tokens is not None else None
    return {
        "ttft_s": first_content_at - started,
        "wall_s": wall,
        "decode_wall_s": decode_wall,
        "decode_tok_s": decode_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stream_content_chunks": chunks,
        "response_chars": sum(len(x) for x in content_parts),
        "response_prefix": "".join(content_parts)[:160],
    }


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"runs": len(runs)}
    for source, target in (("ttft_s", "median_ttft_s"), ("wall_s", "median_wall_s"), ("decode_tok_s", "median_decode_tok_s")):
        vals = [float(r[source]) for r in runs if r.get(source) is not None]
        if vals:
            summary[target] = statistics.median(vals)
            summary[target.replace("median_", "min_")] = min(vals)
            summary[target.replace("median_", "max_")] = max(vals)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    p.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--label", default="tp1-policy")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.max_tokens < 1:
        p.error("--runs/--max-tokens must be positive and --warmup non-negative")

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}")
        stream_once(args.url, args.model, prompt, args.max_tokens, args.timeout)

    runs: list[dict[str, object]] = []
    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}")
        result = stream_once(args.url, args.model, prompt, args.max_tokens, args.timeout)
        runs.append(result)
        print(
            f"  TTFT={result['ttft_s']:.3f}s wall={result['wall_s']:.3f}s "
            f"decode={result['decode_tok_s'] if result['decode_tok_s'] is not None else 'usage-unavailable'} tok/s"
        )

    now = datetime.now(timezone.utc)
    receipt = {
        "schema": 1,
        "created_at": now.isoformat(),
        "label": args.label,
        "request": {
            "url": args.url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "prompt_chars": len(prompt),
            "temperature": 0,
        },
        "runtime": local_runtime(),
        "summary": summarize(runs),
        "measurements": runs,
    }
    output = args.output
    if output is None:
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / "policy" / f"{stamp}-{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"receipt: {output}")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
