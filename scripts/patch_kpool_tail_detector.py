#!/usr/bin/env python3
"""Install an opt-in detector for K-pool tail out-of-bounds writes.

Answers "is my build affected?" with a number instead of luck. Off by default;
enable with ``GLM_KPOOL_TAIL_BOUNDS=1`` on the server process.

Why this exists: every affected build performs the bad writes, and whether one
escapes its allocation and kills the engine depends on where each tail layer's
view sits in the shared KV pool. Contained writes silently corrupt a
neighbouring layer's sparse-attention index. So a run completing proves
nothing, and only a counter does.

Both write paths are instrumented:

  kpool_seed_tail_cache                        prefill seed
  kpool_decode_update_and_maybe_write_cache_batched   decode update

The decode path is the one real workloads hit, because the trigger is generated
tokens rather than prompt length.

Cost when disabled: one module-global boolean test per call. Cost when enabled:
a device sync per call, so do not benchmark with it on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PREAMBLE_ANCHOR = "logger = init_logger(__name__)\n"

PREAMBLE = '''logger = init_logger(__name__)

# --- K-pool tail bounds detector (opt-in: GLM_KPOOL_TAIL_BOUNDS=1) -----------
# See docs/KPOOL_TAIL_BUG.md. Counts writes whose destination block falls
# outside the tail cache. A clean run is not evidence of an unaffected build;
# a zero counter over a long generation is.
_KPOOL_TAIL_BOUNDS = _os_environ_get_kpool_bounds()
_KPOOL_TAIL_STATS = {
    "seed_calls": 0,
    "seed_over": 0,
    "decode_calls": 0,
    "decode_over": 0,
    "worst_block": -1,
    "tail_blocks": -1,
    "reported": 0,
}


def _kpool_tail_report(force: bool = False) -> None:
    s = _KPOOL_TAIL_STATS
    total = s["seed_over"] + s["decode_over"]
    calls = s["seed_calls"] + s["decode_calls"]
    if not force and calls % 1000 != 0:
        return
    logger.warning(
        "KPOOL_TAIL_BOUNDS calls=%d overruns=%d (seed %d/%d, decode %d/%d) "
        "worst_block=%d tail_blocks=%d",
        calls,
        total,
        s["seed_over"],
        s["seed_calls"],
        s["decode_over"],
        s["decode_calls"],
        s["worst_block"],
        s["tail_blocks"],
    )


def _kpool_tail_check(kind: str, blocks, tail_kv_cache) -> None:
    """`blocks` is an int64 tensor of destination block ids actually written."""
    s = _KPOOL_TAIL_STATS
    s[kind + "_calls"] += 1
    cap = tail_kv_cache.shape[0]
    s["tail_blocks"] = cap
    if blocks.numel():
        mx = int(blocks.max().item())
        if mx > s["worst_block"]:
            s["worst_block"] = mx
        if mx >= cap:
            s[kind + "_over"] += 1
            if s["reported"] < 20:
                s["reported"] += 1
                logger.warning(
                    "KPOOL_TAIL_OVERRUN path=%s block=%d tail_blocks=%d "
                    "overrun_by=%d",
                    kind,
                    mx,
                    cap,
                    mx - cap + 1,
                )
    _kpool_tail_report()
# --- end detector ------------------------------------------------------------
'''

HELPER = '''import os as _kpool_os


def _os_environ_get_kpool_bounds() -> bool:
    return _kpool_os.environ.get("GLM_KPOOL_TAIL_BOUNDS") == "1"


'''

SEED_ANCHOR = """    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
"""

SEED_PATCHED = """    n = tslot.shape[0]
    if n == 0:
        return
    if _KPOOL_TAIL_BOUNDS:
        import torch as _t

        # Mirror the kernel's own predicate: a token is stored only when the
        # token KPOOL ahead lands in a different tail block. Bounding every
        # slot would flag skipped tokens that are legitimately out of range.
        _s = tslot.to(_t.int64)
        _blk = _t.div(_s, kpool, rounding_mode="floor")
        _ahead = _t.full_like(_s, -1)
        if _s.numel() > kpool:
            _ahead[:-kpool] = _s[kpool:]
        _same = (_ahead >= 0) & (
            _t.div(_ahead, kpool, rounding_mode="floor") == _blk
        )
        _kpool_tail_check("seed", _blk[(_s >= 0) & ~_same], tail_kv_cache)
    _kpool_tail_seed_kernel[(n,)](
"""

DECODE_ANCHOR = """    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
"""

DECODE_PATCHED = """    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    if _KPOOL_TAIL_BOUNDS:
        import torch as _t

        # Decode writes every valid tail slot; no ahead-predicate here.
        _s = tail_slot_mapping.to(_t.int64).reshape(-1)
        _kpool_tail_check(
            "decode",
            _t.div(_s[_s >= 0], pool_size, rounding_mode="floor"),
            tail_kv_cache,
        )
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="vLLM source root")
    args = ap.parse_args()

    path = Path(args.source) / "vllm/models/glm5next/nvidia/ops/kpool_compress.py"
    text = path.read_text(encoding="utf-8")

    if "_KPOOL_TAIL_BOUNDS" in text:
        print(f"detector already installed: {path}")
        return

    for name, anchor in (
        ("preamble", PREAMBLE_ANCHOR),
        ("seed", SEED_ANCHOR),
        ("decode", DECODE_ANCHOR),
    ):
        if text.count(anchor) != 1:
            raise SystemExit(
                f"{path}: expected one {name} anchor, found {text.count(anchor)}"
            )

    text = text.replace(PREAMBLE_ANCHOR, HELPER + PREAMBLE, 1)
    text = text.replace(SEED_ANCHOR, SEED_PATCHED, 1)
    text = text.replace(DECODE_ANCHOR, DECODE_PATCHED, 1)
    path.write_text(text, encoding="utf-8")

    print(f"detector installed: {path}")
    print("enable with GLM_KPOOL_TAIL_BOUNDS=1 on the server process")


if __name__ == "__main__":
    main()
