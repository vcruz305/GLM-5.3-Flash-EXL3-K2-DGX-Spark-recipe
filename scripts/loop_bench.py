#!/usr/bin/env python3
"""Loop / degeneration battery against an OpenAI-compatible vLLM server,
shaped like an agent harness (Hermes-style): a long system prompt, eight tool
definitions, multi-turn tool results, thinking on, vendor sampling.

Every response is scored for
  - length_hit : finish_reason == "length" (ran to the token budget)
  - ngram_loop : an 8-word phrase repeated >= 6 times, or a 20..200 char chunk
                 repeated 5+ times back to back
  - tool_loop  : the same tool call (name + arguments) emitted >= 3 times
  - empty      : no content and no tool call

"loop" = ngram_loop or tool_loop. Prints one JSON line per response and a
final SUMMARY line. Run with --concurrency N to reproduce multi-request
serving (the reporter's max-num-seqs 4).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys
import time

import requests

SYSTEM = """You are Hermes, an autonomous software engineering and research agent running inside a developer's workstation. You have access to tools that read and write files, run shell commands, search the web and fetch pages. Work step by step: inspect before you change, prefer small verifiable edits, run tests after edits, and report what you did in plain language.

Rules:
1. Never invent file contents or command output. If you need information, call a tool.
2. When a task is complete, say so and stop. Do not repeat yourself.
3. Keep answers concise; put code in fenced blocks with the language tag.
4. If a tool returns an error, explain it and propose the next step.
5. Do not run destructive commands (rm -rf, git push --force, database drops) without saying so first.
6. For research questions, cite the source URL you fetched.
7. Use the python tool for arithmetic instead of guessing.
8. Prefer the project's existing conventions over your own.

Environment: Ubuntu 24.04, Python 3.12, git, node 22, docker available. Working directory is /workspace/project. The user is an experienced engineer and wants direct answers."""

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file from the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a text file in the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the workspace and return stdout, stderr and the exit code.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory (non-recursive).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web; returns titles, URLs and snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_url", "description": "Fetch a URL and return its text content.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "python", "description": "Execute Python code and return stdout.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "git_diff", "description": "Return the working tree diff of the workspace repository.", "parameters": {"type": "object", "properties": {"staged": {"type": "boolean"}}}}},
]

BUGGY = '''```python
def moving_average(xs, k):
    out = []
    for i in range(len(xs)):
        window = xs[i:i+k]
        out.append(sum(window) / k)
    return out

def dedupe(items):
    seen = set()
    for it in items:
        if it in seen:
            items.remove(it)
        seen.add(it)
    return items

class Cache:
    store = {}
    def get(self, k, default=None):
        return self.store.get(k, default)
    def put(self, k, v):
        self.store[k] = v
```'''

DOC = """Project notes: The ingest service accepts CSV uploads up to 250 MB, validates the header against the schema registry, and writes row batches of 5,000 to the staging table. Failed rows go to a quarantine table with the validation error. The scheduler runs the promote job every 15 minutes; it moves staged batches whose checksum matches the manifest into the warehouse partitions keyed by upload date. Retries are capped at 3 with exponential backoff starting at 30 seconds. Metrics are exported on :9102 (rows_ingested_total, rows_quarantined_total, promote_latency_seconds). Known issues: the header validator is case-sensitive; uploads with a UTF-8 BOM fail schema lookup; the promote job holds a table lock for the full batch which blocks reads during large promotes. Planned work: case-insensitive header matching, BOM stripping, and chunked promotes with row-level locks. On-call runbook: if rows_quarantined_total spikes, check the schema registry for a version bump first; if promote_latency_seconds exceeds 120 s, look for a stuck lock with pg_locks and cancel the oldest promote."""

# Synthetic tool results, keyed by tool name, so multi-turn flows continue.
TOOL_RESULTS = {
    "read_file": {"config.yaml": "server:\n  host: 0.0.0.0\n  port: 8443\n  workers: 4\nlogging:\n  level: info\n  file: /var/log/app.log\ndatabase:\n  url: postgres://app@db:5432/app\n"},
    "run_command": {"git status": " M src/api/routes.py\n M src/api/models.py\n?? src/api/tests/test_routes.py\n", "pytest": "============ 47 passed, 1 failed in 12.31s ============\nFAILED tests/test_routes.py::test_create_user - AssertionError: expected 201, got 500\n", "python hello.py": "hello\n"},
    "list_dir": {"src": "api/\ncore/\ncli.py\n__init__.py\nutils.py\n"},
    "web_search": [{"title": "vLLM v0.11.0 release", "url": "https://github.com/vllm-project/vllm/releases/tag/v0.11.0", "snippet": "vLLM v0.11.0: new V2 model runner, hybrid model support, DeepSeek V4 sparse attention."}],
    "fetch_url": "vLLM v0.11.0 highlights: V2 model runner default, Glm5Next hybrid support, sparse MLA on Blackwell, FlashInfer 0.6.",
    "python": "42\n",
    "git_diff": "diff --git a/src/api/routes.py b/src/api/routes.py\n@@ -40,7 +40,7 @@ def create_user(payload):\n-    user = User(**payload)\n+    user = User(**payload, created_at=now())\n",
    "write_file": "ok",
}


def tool_result(name: str, args: dict) -> str:
    r = TOOL_RESULTS.get(name, "ok")
    if isinstance(r, dict):
        for k, v in r.items():
            if k in json.dumps(args):
                return v
        return next(iter(r.values()))
    if isinstance(r, list):
        return json.dumps(r)
    return r


def prompts() -> list[dict]:
    P = []
    chat = [
        ("code-cli", "Write a Python CLI (argparse) that walks a directory, finds files over a size threshold, and prints them sorted by size with human-readable units. Include type hints and a __main__ guard."),
        ("code-bug", "Find and fix every bug in this code. Explain each fix in one line.\n\n" + BUGGY),
        ("code-tests", "Write pytest unit tests for a function `parse_duration(s: str) -> int` that accepts strings like '90s', '5m', '2h30m', '1d' and returns seconds, raising ValueError on bad input. Cover edge cases."),
        ("code-lru", "Implement an LRU cache with per-entry TTL in Python without third-party packages. Show the class and a short usage example."),
        ("code-bash", "Write a bash script that rotates logs in /var/log/app: compress files older than 1 day, delete compressed files older than 14 days, and print a summary. Must be safe under `set -euo pipefail`."),
        ("code-sql", "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), write SQL for the top 5 customers by revenue per country in 2025, with ties broken by name."),
        ("reason-math", "A tank fills through pipe A in 6 hours and drains through pipe B in 9 hours. Both are open for 2 hours, then B is closed. How long until the tank is full? Show the steps."),
        ("reason-cap", "Explain the CAP theorem with one concrete example each of a CP and an AP system, and say what 'partition tolerance' means operationally."),
        ("reason-compare", "Compare Rust and Go for writing a cross-platform CLI that ships as a single binary. Give a recommendation for a 3-person team with Python background."),
        ("plan-itinerary", "Plan a 3-day work trip to Tokyo for an engineer: two client meetings in Shinjuku, one in Shinagawa, arriving Sunday evening, leaving Wednesday night. Keep it realistic and brief."),
        ("constrained-list", "List 30 different animals, one per line, numbered, no repeats, no commentary."),
        ("constrained-haiku", "Write ten numbered haikus about the sea. Each must be three lines. No two haikus may share a first line."),
        ("summarize-doc", "Summarize the following notes into: (1) what the service does, (2) known issues, (3) the on-call checklist. Be brief.\n\n" + DOC),
        ("explain-kld", "Explain what KL divergence measures when comparing a quantized model to its full-precision teacher, and why the median and the p99 can tell different stories."),
        ("json-only", "Return a JSON object describing three fictional employees with fields name, role, start_date (ISO), skills (array). JSON only, no prose."),
        ("rewrite", "Rewrite this paragraph for a status update to executives, under 80 words: 'so basically the migration blew up because the connection pool was maxed and nobody noticed the alert was muted, we rolled back and it's fine now but we need to fix the alerting before we try again next week.'"),
        ("multi-turn", None),  # filled below
        ("stop-test", "Say exactly the word DONE and nothing else."),
    ]
    for pid, text in chat:
        if pid == "multi-turn":
            P.append({"id": pid, "tools": False, "messages": [
                {"role": "user", "content": "I'm building a rate limiter for an API. Token bucket or sliding window?"},
                {"role": "assistant", "content": "Token bucket if you want to allow short bursts up to a cap; sliding window log if you need strict limits per window. For most public APIs token bucket with a refill rate equal to the sustained limit is the usual choice."},
                {"role": "user", "content": "Token bucket then. Write it in Go, safe for concurrent use, with a Allow() method and a small test."},
            ]})
        else:
            P.append({"id": pid, "tools": False, "messages": [{"role": "user", "content": text}]})
    tool_tasks = [
        ("tool-git", "Check the git status of the repo and summarize the uncommitted changes."),
        ("tool-config", "Read config.yaml and tell me which port the server listens on and where it logs."),
        ("tool-search", "Search the web for the latest vLLM release and report the version and its headline features."),
        ("tool-tests", "Run the test suite. If anything fails, read the diff and propose a fix."),
        ("tool-layout", "List the files in src/ and describe the project layout in two sentences."),
        ("tool-hello", "Create hello.py that prints 'hello', run it, and confirm the output."),
    ]
    for pid, text in tool_tasks:
        P.append({"id": pid, "tools": True, "messages": [{"role": "user", "content": text}]})
    return P


def score_text(text: str) -> dict:
    words = text.split()
    ngram_max = 0
    if len(words) >= 8:
        c = collections.Counter(tuple(words[i:i + 8]) for i in range(len(words) - 7))
        ngram_max = max(c.values())
    chunk_rep = bool(re.search(r"(.{20,200}?)(?:\s*\1){4,}", text, flags=re.S))
    return {"ngram8_max": ngram_max, "chunk_repeat": chunk_rep,
            "ngram_loop": ngram_max >= 6 or chunk_rep}


def one_request(base_url, model, messages, tools, args) -> dict:
    body = {
        "model": model, "messages": messages, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/chat/completions", json=body, timeout=args.timeout)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}", "wall_s": dt}
    d = r.json()
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    calls = msg.get("tool_calls") or []
    return {"finish": ch.get("finish_reason"), "content": content, "reasoning": reasoning,
            "tool_calls": calls, "usage": d.get("usage", {}), "wall_s": dt}


def run_item(item, args) -> list[dict]:
    out = []
    messages = [{"role": "system", "content": SYSTEM}] + item["messages"]
    seen_calls = collections.Counter()
    for rnd in range(1, (args.rounds if item["tools"] else 1) + 1):
        res = one_request(args.base_url, args.model, messages, item["tools"], args)
        rec = {"id": item["id"], "round": rnd}
        if "error" in res:
            rec.update({"error": res["error"], "wall_s": round(res["wall_s"], 1)})
            out.append(rec)
            break
        text = res["reasoning"] + "\n" + res["content"]
        sc = score_text(text)
        call_keys = [c["function"]["name"] + ":" + json.dumps(c["function"].get("arguments", ""), sort_keys=True) for c in res["tool_calls"]]
        for k in call_keys:
            seen_calls[k] += 1
        within = collections.Counter(call_keys)
        tool_loop = bool(within and max(within.values()) >= 3) or bool(seen_calls and max(seen_calls.values()) >= 3)
        ct = res["usage"].get("completion_tokens", 0)
        rec.update({
            "finish": res["finish"], "completion_tokens": ct, "wall_s": round(res["wall_s"], 1),
            "tps": round(ct / res["wall_s"], 2) if res["wall_s"] > 0 else None,
            "length_hit": res["finish"] == "length", "tool_calls": len(res["tool_calls"]),
            "tool_loop": tool_loop, "empty": not res["content"].strip() and not res["tool_calls"],
            **sc,
        })
        rec["loop"] = rec["ngram_loop"] or rec["tool_loop"]
        rec["excerpt"] = (res["content"] or res["reasoning"])[:400]
        if rec["loop"]:
            rec["loop_tail"] = text[-600:]
        out.append(rec)
        if not res["tool_calls"]:
            break
        messages.append({"role": "assistant", "content": res["content"] or None, "tool_calls": res["tool_calls"]})
        for c in res["tool_calls"]:
            try:
                a = json.loads(c["function"].get("arguments") or "{}")
            except Exception:
                a = {}
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result(c["function"]["name"], a)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--label", default="loops")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--rounds", type=int, default=3, help="max tool rounds per tool task")
    ap.add_argument("--repeat", type=int, default=1, help="run the battery N times")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    items = prompts() * args.repeat
    t0 = time.perf_counter()
    recs: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex, open(args.out, "a") as fh:
        for rs in ex.map(lambda it: run_item(it, args), items):
            for r in rs:
                r["label"] = args.label
                fh.write(json.dumps(r) + "\n"); fh.flush()
                print(json.dumps({k: v for k, v in r.items() if k not in ("excerpt", "loop_tail")}), flush=True)
                recs.append(r)
    wall = time.perf_counter() - t0
    ok = [r for r in recs if "error" not in r]
    errs = len(recs) - len(ok)
    toks = sum(r["completion_tokens"] for r in ok)
    loops = [r for r in ok if r["loop"]]
    lens = [r for r in ok if r["length_hit"]]
    tl = [r for r in ok if r["tool_loop"]]
    empt = [r for r in ok if r["empty"]]
    ct = sorted(r["completion_tokens"] for r in ok)
    med = ct[len(ct) // 2] if ct else 0
    print(
        f"SUMMARY {args.label}: responses={len(recs)} errors={errs} loops={len(loops)} "
        f"({', '.join(r['id'] + '#' + str(r['round']) for r in loops) or 'none'}) "
        f"length_hits={len(lens)} ({', '.join(r['id'] for r in lens) or 'none'}) tool_loops={len(tl)} empty={len(empt)} "
        f"median_tokens={med} total_tokens={toks} wall={wall:.0f}s agg_tps={toks / wall:.2f} "
        f"concurrency={args.concurrency} thinking={args.thinking} T={args.temperature}",
        flush=True,
    )
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
