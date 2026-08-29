#!/usr/bin/env python3
"""Summarize measured rows emitted by bench_ladder.py."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("warmup", False) and row.get("decode_tok_s") is not None:
                groups[(row["label"], row["prompt"])].append(row)

    labels: dict[str, list[float]] = defaultdict(list)
    for (label, prompt), rows in sorted(groups.items()):
        speeds = [float(row["decode_tok_s"]) for row in rows]
        med = statistics.median(speeds)
        labels[label].append(med)
        summaries = [row.get("spec_summary") or {} for row in rows]
        accept = [float(item["accept_ratio"]) for item in summaries if "accept_ratio" in item]
        verified = [
            float(item["verified_tokens_per_step"])
            for item in summaries
            if "verified_tokens_per_step" in item
        ]
        print(
            json.dumps(
                {
                    "label": label,
                    "prompt": prompt,
                    "n": len(rows),
                    "decode_tok_s_median": round(med, 4),
                    "accept_ratio_median": round(statistics.median(accept), 4)
                    if accept
                    else None,
                    "verified_tokens_per_step_median": round(
                        statistics.median(verified), 4
                    )
                    if verified
                    else None,
                },
                sort_keys=True,
            )
        )

    print("\nLABEL MEANS")
    for label, medians in sorted(labels.items()):
        print(f"{statistics.mean(medians):.4f}\t{label}")


if __name__ == "__main__":
    main()
