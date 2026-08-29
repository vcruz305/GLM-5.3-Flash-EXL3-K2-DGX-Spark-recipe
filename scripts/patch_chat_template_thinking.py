#!/usr/bin/env python3
"""Make the model's stock chat template honor enable_thinking.

The Hub template always opens a reasoning block, even when callers pass
``chat_template_kwargs={"enable_thinking": false}``.  This applies the three
small, auditable changes used by the benchmark recipe without replacing the
rest of the model-provided template.
"""
from __future__ import annotations

import argparse
from pathlib import Path


PATCHES = (
    (
        "[gMASK]<sop>\n"
        "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}\n",
        "[gMASK]<sop>\n"
        "{%- if thinking is defined or enable_thinking is defined -%}\n"
        "{%- set thinking_enabled = (thinking if thinking is defined else false) or (enable_thinking if enable_thinking is defined else false) -%}\n"
        "{%- else -%}\n"
        "{%- set thinking_enabled = true -%}\n"
        "{%- endif -%}\n"
        "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}\n",
    ),
    (
        "{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}",
        "{%- if thinking_enabled and effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}",
    ),
    (
        "    <|assistant|>{{- '<think>' -}}\n{%- endif -%}",
        "    <|assistant|>{{- '<think>' if thinking_enabled else '<think></think>' -}}\n{%- endif -%}",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "template",
        nargs="?",
        type=Path,
        default=Path.home() / "models/GLM-5.3-Flash-EXL3-K2/chat_template.jinja",
    )
    args = parser.parse_args()
    path = args.template.resolve()
    if not path.is_file():
        parser.error(f"template not found: {path}")

    source = path.read_text(encoding="utf-8")
    changed = False
    for old, new in PATCHES:
        if new in source:
            continue
        if source.count(old) != 1:
            raise SystemExit(f"expected exactly one patch target in {path}: {old!r}")
        source = source.replace(old, new, 1)
        changed = True

    if changed:
        path.write_text(source, encoding="utf-8")
        print(f"patched thinking toggle: {path}")
    else:
        print(f"thinking toggle already patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
