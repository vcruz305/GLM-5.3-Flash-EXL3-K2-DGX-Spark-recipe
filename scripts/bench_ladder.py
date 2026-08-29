#!/usr/bin/env python3
"""Repeatable streamed /v1 benchmark for the one-Spark optimization ladder.

Uses only the Python standard library. Each measured request is written as one
JSONL row so a partial ladder remains useful if a later server boot fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = {
    "prose": (
        "Write a self-contained technical explanation of how unified CPU/GPU "
        "memory changes large-language-model inference on a single workstation. "
        "Use complete paragraphs, include benefits and bottlenecks, and do not "
        "use a table. Continue until the response is at least 400 tokens."
    ),
    "structured": (
        "Output the integers from 1 through 220 in order. Put exactly one space "
        "between integers. Do not omit, repeat, explain, or add any other text."
    ),
    "code": (
        "Implement a production-quality Python function that parses a stream of "
        "Server-Sent Events from an iterable of bytes, yields decoded JSON data "
        "objects, handles comments and multi-line data fields, and stops on "
        "[DONE]. Include type hints, a docstring, and focused unit tests."
    ),
    "math": (
        "Solve this carefully and show the derivation: a decoder produces one "
        "verified token per base-model step without speculation. A speculative "
        "method proposes k tokens; position i is accepted only if all earlier "
        "positions were accepted. Derive expected verified tokens per step from "
        "conditional acceptance probabilities, then evaluate k=4 for "
        "p=[0.82,0.61,0.40,0.20]. Discuss when proposal overhead erases the gain."
    ),
}

METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def optional_request_json(url: str, timeout: int = 10) -> Any | None:
    try:
        return request_json(url, timeout=timeout)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def metrics_snapshot(metrics_url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        match = METRIC_LINE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if "spec" not in name.lower():
            continue
        key = name + (match.group("labels") or "")
        out[key] = out.get(key, 0.0) + float(match.group("value"))
    return out


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        name: round(value - before.get(name, 0.0), 6)
        for name, value in sorted(after.items())
        if value - before.get(name, 0.0)
    }


def metrics_counter_reset(before: dict[str, float], after: dict[str, float]) -> bool:
    return any(name in after and after[name] < value for name, value in before.items())


def speculative_summary(delta: dict[str, float]) -> dict[str, float]:
    """Reduce vLLM's speculative counters to comparable request-level ratios."""

    def total(metric: str) -> float:
        return sum(value for name, value in delta.items() if name.startswith(metric))

    drafts = total("vllm:spec_decode_num_drafts_total")
    proposed = total("vllm:spec_decode_num_draft_tokens_total")
    accepted = total("vllm:spec_decode_num_accepted_tokens_total")
    if not drafts or not proposed:
        return {}
    return {
        "draft_steps": round(drafts, 6),
        "proposed_tokens": round(proposed, 6),
        "accepted_tokens": round(accepted, 6),
        "accept_ratio": round(accepted / proposed, 6),
        "accepted_per_step": round(accepted / drafts, 6),
        "verified_tokens_per_step": round(1.0 + accepted / drafts, 6),
    }


def stream_chat(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    chat_template: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if chat_template is not None:
        payload["chat_template"] = chat_template
    return stream_openai(url, payload, timeout, chat_mode=True)


def stream_completion(
    url: str,
    model: str,
    rendered_prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": rendered_prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return stream_openai(url, payload, timeout, chat_mode=False)


def stream_openai(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    *,
    chat_mode: bool,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    parts: list[str] = []

    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                if chat_mode:
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                else:
                    content = choice.get("text") or ""
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    parts.append(content)

    finished = time.perf_counter()
    output = "".join(parts)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft_s = (first_content_at or finished) - started
    decode_s = max(finished - (first_content_at or finished), 1e-9)
    decode_tok_s = (
        (completion_tokens - 1) / decode_s if completion_tokens > 1 else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": round(ttft_s * 1000, 3),
        "wall_s": round(finished - started, 6),
        "decode_s": round(decode_s, 6),
        "decode_tok_s": round(decode_tok_s, 4) if decode_tok_s is not None else None,
        "wall_tok_s": round(completion_tokens / (finished - started), 4)
        if completion_tokens
        else None,
        "finish_reason": finish_reason,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_head": output[:120].replace("\n", "\\n"),
        "output_tail": output[-120:].replace("\n", "\\n"),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--label", required=True, help="Unique server/config rung label")
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=sorted(PROMPTS),
        default=list(PROMPTS),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--chat-template-file",
        type=Path,
        help="Render this Jinja template client-side for an exact cross-runtime A/B",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer directory used with --chat-template-file",
    )
    parser.add_argument("--out", type=Path, default=Path("ladder.jsonl"))
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup must be >= 0")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be >= 2 for steady-state decode TPS")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")

    base = args.base_url.rstrip("/")
    models = request_json(base + "/models")
    available = [row.get("id") for row in models.get("data", [])]
    if args.model not in available:
        raise SystemExit(f"model {args.model!r} not in /models: {available}")
    metrics_url = base.removesuffix("/v1") + "/metrics"
    server_identity = {
        "models": models.get("data", []),
        "version": optional_request_json(base.removesuffix("/v1") + "/version"),
    }
    chat_template = (
        args.chat_template_file.read_text(encoding="utf-8")
        if args.chat_template_file is not None
        else None
    )
    chat_template_sha256 = (
        hashlib.sha256(chat_template.encode()).hexdigest()
        if chat_template is not None
        else None
    )
    rendered_prompts: dict[str, str] = {}
    if chat_template is not None:
        if args.tokenizer is None:
            parser.error("--tokenizer is required with --chat-template-file")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for prompt_name, prompt in PROMPTS.items():
            rendered_prompts[prompt_name] = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prompt_name in args.prompts:
        prompt = PROMPTS[prompt_name]
        for ordinal in range(args.warmup + args.runs):
            warmup = ordinal < args.warmup
            before = metrics_snapshot(metrics_url)
            if chat_template is None:
                result = stream_chat(
                    base + "/chat/completions",
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.timeout,
                )
            else:
                result = stream_completion(
                    base + "/completions",
                    args.model,
                    rendered_prompts[prompt_name],
                    args.max_tokens,
                    args.timeout,
                )
            after = metrics_snapshot(metrics_url)
            counter_reset = metrics_counter_reset(before, after)
            spec_delta = {} if counter_reset else metric_delta(before, after)
            row = {
                "ts_unix": round(time.time(), 3),
                "label": args.label,
                "prompt": prompt_name,
                "run": ordinal + 1 if warmup else ordinal - args.warmup + 1,
                "warmup": warmup,
                "chat_template_sha256": chat_template_sha256,
                "server_identity": server_identity,
                **result,
                "metrics_counter_reset": counter_reset,
                "spec_metrics_delta": spec_delta,
                "spec_summary": speculative_summary(spec_delta),
            }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if not warmup:
                rows.append(row)

    print("\nSUMMARY")
    for prompt_name in args.prompts:
        prompt_rows = [row for row in rows if row["prompt"] == prompt_name]
        speeds = [
            float(row["decode_tok_s"])
            for row in prompt_rows
            if row["decode_tok_s"] is not None
        ]
        ttfts = [float(row["ttft_ms"]) for row in prompt_rows]
        hashes = {row["output_sha256"] for row in prompt_rows}
        summary = {
            "label": args.label,
            "prompt": prompt_name,
            "n": len(prompt_rows),
            "decode_tok_s_median": round(statistics.median(speeds), 4) if speeds else None,
            "decode_tok_s_min": round(min(speeds), 4) if speeds else None,
            "decode_tok_s_max": round(max(speeds), 4) if speeds else None,
            "ttft_ms_median": round(statistics.median(ttfts), 3) if ttfts else None,
            "ttft_ms_p95": round(percentile(ttfts, 0.95) or 0, 3) if ttfts else None,
            "unique_output_hashes": len(hashes),
        }
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
