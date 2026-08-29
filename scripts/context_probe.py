#!/usr/bin/env python3
"""Exercise an allocated context with a real token-ID prompt over OpenAI HTTP."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--token-id", type=int, default=198)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.max_tokens < 1:
        parser.error("token counts must be positive")
    if args.token_id < 0:
        parser.error("--token-id must be non-negative")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")

    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(base_url + "/models", timeout=30) as response:
        models = json.loads(response.read())
    model_rows = [
        row for row in models.get("data", []) if row.get("id") == args.model
    ]
    if len(model_rows) != 1:
        raise SystemExit(f"model {args.model!r} not uniquely advertised: {model_rows}")
    model_row = model_rows[0]
    advertised_max = model_row.get("max_model_len")
    requested_total = args.prompt_tokens + args.max_tokens
    if advertised_max is not None and requested_total > int(advertised_max):
        raise SystemExit(
            f"requested {requested_total} total tokens exceeds advertised "
            f"max_model_len {advertised_max}"
        )

    payload = {
        "model": args.model,
        "prompt": [args.token_id] * args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url + "/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read())
    wall_s = time.perf_counter() - started
    with urllib.request.urlopen(base_url.removesuffix("/v1") + "/health", timeout=30) as response:
        health_after = response.status
    usage = result.get("usage") or {}
    receipt = {
        "model": args.model,
        "server_model": model_row,
        "requested_prompt_tokens": args.prompt_tokens,
        "requested_total_tokens": requested_total,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "wall_s": round(wall_s, 3),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "health_after": health_after,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
