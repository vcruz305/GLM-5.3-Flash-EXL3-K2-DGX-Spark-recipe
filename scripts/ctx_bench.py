#!/usr/bin/env python3
"""Long-context ladder against a running server: real text, a needle, and
prefill / decode throughput per point.

For each target prompt size the bench assembles natural text from the fidelity
suite's token windows, plants a needle sentence at 10% depth, and sends two
chat requests with thinking off:

  1. "summarize" with max_tokens=400: time to first token gives the prefill
     rate (prompt_tokens / TTFT), the rest gives the decode rate.
  2. "what is the vault code?" with max_tokens=32: needle recall.

Serve with prefix caching OFF for clean prefill numbers; the JSON records both
TTFTs so a cache hit on the second request is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

NEEDLE = "\n\nNote for the reader: the vault code is 7391-ALPHA. Remember it.\n\n"
Q_NEEDLE = "\n\nQuestion: what is the vault code mentioned in the text above? Reply with the code only."
Q_SUMMARY = "\n\nWrite a detailed summary of the text above."


def build_corpus(tok, tokdir: Path, need: int) -> str:
    parts, count, i = [], 0, 0
    while count < need:
        ids = json.loads((tokdir / f"context-{i:04d}.json").read_text())
        parts.append(tok.decode(ids, skip_special_tokens=True))
        count += len(ids)
        i += 1
    return "\n\n".join(parts)


def size_text(tok, text: str, target: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > target:
        ids = ids[:target]
    return tok.decode(ids, skip_special_tokens=True)


def stream_chat(base_url: str, model: str, content: str, max_tokens: int, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    text = []
    usage = None
    with requests.post(f"{base_url}/chat/completions", json=body, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    total = time.perf_counter() - t0
    return {"ttft_s": ttft, "total_s": total, "text": "".join(text), "usage": usage}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="tokenizer source")
    ap.add_argument("--suite", required=True, help="fidelity suite root (suite/tokens/*.json)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--points", default="8192,32768,65536,131072,196608,258048")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--label", default="ctx")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    points = [int(p) for p in args.points.split(",")]
    tokdir = Path(args.suite) / "suite/tokens"
    corpus = build_corpus(tok, tokdir, max(points) + 4096)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    summary = []
    with out.open("a") as fh:
        for target in points:
            body_tokens = target - 400  # room for needle, question, and chat template
            text = size_text(tok, corpus, body_tokens)
            cut = int(len(text) * 0.10)
            text = text[:cut] + NEEDLE + text[cut:]
            rec = {"label": args.label, "target": target}
            s = stream_chat(args.base_url, args.model, text + Q_SUMMARY, args.max_tokens, args.timeout)
            if "error" in s or not s.get("usage") or s.get("ttft_s") is None:
                rec["error"] = s.get("error", "no usage/ttft in stream")
                failures += 1
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                summary.append(f"{target//1024}k: ERROR {rec['error'][:60]}")
                print(json.dumps(rec), flush=True)
                continue
            pt = s["usage"]["prompt_tokens"]
            ct = s["usage"]["completion_tokens"]
            dec_s = max(s["total_s"] - s["ttft_s"], 1e-6)
            rec.update({
                "prompt_tokens": pt,
                "ttft_s": round(s["ttft_s"], 3),
                "prefill_tps": round(pt / s["ttft_s"], 1),
                "gen_tokens": ct,
                "decode_s": round(dec_s, 3),
                "decode_tps": round(ct / dec_s, 2) if ct > 1 else None,
                "total_s": round(s["total_s"], 3),
            })
            n = stream_chat(args.base_url, args.model, text + Q_NEEDLE, 32, args.timeout)
            if "error" in n:
                rec["needle_error"] = n["error"]
                failures += 1
            else:
                rec["needle_ttft_s"] = round(n["ttft_s"], 3) if n.get("ttft_s") else None
                rec["needle_answer"] = n["text"].strip()[:80]
                rec["needle_ok"] = "7391-ALPHA" in n["text"]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(json.dumps(rec), flush=True)
            summary.append(
                f"{pt//1024}k: prefill {rec['prefill_tps']:,.0f} tok/s (ttft {rec['ttft_s']:.1f}s) "
                f"decode {rec['decode_tps']} tok/s over {ct} needle={'OK' if rec.get('needle_ok') else 'MISS'}"
            )
    print("SUMMARY " + " | ".join(summary), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
