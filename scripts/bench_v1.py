#!/usr/bin/env python3
"""Streamed /v1 chat TPS. Thinking off. Same shape as the measured 128-token runs."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    p.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--prompt",
        default="Reply with a short technical paragraph about GB10 unified memory.",
    )
    args = p.parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    completion = 0
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if ttft is None:
                ttft = time.perf_counter() - t0
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                chunks += 1
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens"):
                completion = int(usage["completion_tokens"])
    wall = time.perf_counter() - t0
    decode_s = max((wall - (ttft or 0)), 1e-6)
    n = completion or args.max_tokens
    out = {
        "ttft_s": round(ttft or wall, 3),
        "wall_s": round(wall, 3),
        "completion_tokens": n,
        "decode_tok_s": round((n - 1) / decode_s, 3) if n > 1 else None,
        "chunks": chunks,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
