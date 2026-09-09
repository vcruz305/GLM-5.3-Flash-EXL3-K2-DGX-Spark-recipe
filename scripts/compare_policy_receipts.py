#!/usr/bin/env python3
"""Compare two JSON receipts written by bench_tp1_policy.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise SystemExit(f"{path}: unsupported receipt schema {data.get('schema')!r}")
    return data


def number(data: dict[str, Any], key: str) -> float | None:
    value = (data.get("summary") or {}).get(key)
    return None if value is None else float(value)


def pct(candidate: float, baseline: float, higher_is_better: bool) -> float:
    if baseline == 0:
        raise ZeroDivisionError("baseline metric is zero")
    raw = (candidate / baseline - 1.0) * 100.0
    return raw if higher_is_better else -raw


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--json", action="store_true", help="emit machine-readable comparison")
    args = p.parse_args()

    a = load(args.baseline)
    b = load(args.candidate)
    metrics = [
        ("median_decode_tok_s", True, "decode tok/s"),
        ("median_ttft_s", False, "TTFT s"),
        ("median_wall_s", False, "wall s"),
    ]
    comparison: dict[str, Any] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_label": a.get("label"),
        "candidate_label": b.get("label"),
        "metrics": {},
        "baseline_policy": (a.get("runtime") or {}).get("vllm_exl3_policy"),
        "candidate_policy": (b.get("runtime") or {}).get("vllm_exl3_policy"),
    }

    for key, higher, label in metrics:
        av, bv = number(a, key), number(b, key)
        entry: dict[str, Any] = {"baseline": av, "candidate": bv}
        if av is not None and bv is not None and av != 0:
            entry["improvement_pct"] = pct(bv, av, higher)
        comparison["metrics"][key] = entry

    if args.json:
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0

    print(f"baseline : {a.get('label')} ({args.baseline})")
    print(f"candidate: {b.get('label')} ({args.candidate})")
    print()
    print(f"{'metric':<16} {'baseline':>12} {'candidate':>12} {'improvement':>13}")
    print("-" * 57)
    for key, _higher, label in metrics:
        e = comparison["metrics"][key]
        improvement = e.get("improvement_pct")
        change = "n/a" if improvement is None else f"{improvement:+.2f}%"
        print(f"{label:<16} {fmt(e['baseline']):>12} {fmt(e['candidate']):>12} {change:>13}")

    print("\nRecorded EXL3 policies:")
    print("baseline:")
    print(json.dumps(comparison["baseline_policy"], indent=2, sort_keys=True))
    print("candidate:")
    print(json.dumps(comparison["candidate_policy"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
