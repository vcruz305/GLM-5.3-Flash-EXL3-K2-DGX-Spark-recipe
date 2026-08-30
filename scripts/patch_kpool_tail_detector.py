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

Cost when disabled: one module-global boolean test per call.

When enabled the counters are accumulated on the device, because the decode path
runs inside CUDA graph capture where any host sync raises "operation not
permitted when stream is capturing". Read-back happens only outside capture and
at most every 200 calls. Still, do not benchmark with it on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PREAMBLE_ANCHOR = "INDEX_HEAD_DIM = 128\n"

PREAMBLE = '''INDEX_HEAD_DIM = 128

# --- K-pool tail bounds detector (opt-in: GLM_KPOOL_TAIL_BOUNDS=1) -----------
# See docs/KPOOL_TAIL_BUG.md. Counts writes whose destination block falls
# outside the tail cache. A clean run is not evidence of an unaffected build;
# a zero counter over a long generation is.
#
# The decode path runs inside CUDA graph capture, where any host sync
# (.item(), .cpu(), print of a tensor) raises
# "operation not permitted when stream is capturing". So counters live on the
# device and are only read back when the stream is NOT capturing.
import os as _kpool_os

_KPOOL_TAIL_BOUNDS = _kpool_os.environ.get("GLM_KPOOL_TAIL_BOUNDS") == "1"
# [seed_calls, seed_over, decode_calls, decode_over, worst_block]
_KPOOL_TAIL_C = None
_KPOOL_TAIL_LAST = -1
_KPOOL_TAIL_CAP = -1   # Python-side: assigning an int into a CUDA tensor is a
                       # host->device copy, illegal during graph capture.


def _kpool_tail_check(kind, max_block, tail_kv_cache):
    """Accumulate on-device.

    `max_block` is a 0-dim int64 tensor: the largest destination block this call
    will write, or -1 if it writes none. It must be produced with fixed-shape
    ops only. Boolean mask indexing (x[mask]) has a data-dependent output shape
    and forces a device sync, which is illegal inside CUDA graph capture.
    """
    global _KPOOL_TAIL_C, _KPOOL_TAIL_LAST, _KPOOL_TAIL_CAP
    dev = tail_kv_cache.device
    if _KPOOL_TAIL_C is None or _KPOOL_TAIL_C.device != dev:
        # Never allocate inside capture: a tensor created there lives in the
        # graph's private pool and touching it afterwards is an illegal access.
        if torch.cuda.is_current_stream_capturing():
            return
        _KPOOL_TAIL_C = torch.zeros(5, dtype=torch.int64, device=dev)
    c = _KPOOL_TAIL_C
    i = 0 if kind == "seed" else 2
    cap = int(tail_kv_cache.shape[0])          # shape read: no sync
    _KPOOL_TAIL_CAP = cap
    # Every op below is a device kernel with a Python scalar operand; none of
    # them materialises a host tensor, so all are legal under capture.
    c[i] += 1
    c[4] = torch.maximum(c[4], max_block)
    c[i + 1] += (max_block >= cap).to(torch.int64)

    # Read back only outside capture, and only occasionally.
    if torch.cuda.is_current_stream_capturing():
        return
    calls = int(c[0].item()) + int(c[2].item())
    if calls - _KPOOL_TAIL_LAST < 200:
        return
    _KPOOL_TAIL_LAST = calls
    v = c.tolist()
    over = v[1] + v[3]
    print(
        "KPOOL_TAIL_BOUNDS calls=%d overruns=%d (seed %d/%d, decode %d/%d) "
        "worst_block=%d tail_blocks=%d"
        % (calls, over, v[1], v[0], v[3], v[2], v[4], _KPOOL_TAIL_CAP),
        flush=True,
    )
    if over:
        print(
            "KPOOL_TAIL_OVERRUN total=%d worst_block=%d tail_blocks=%d"
            % (over, v[4], _KPOOL_TAIL_CAP),
            flush=True,
        )
# --- end detector ------------------------------------------------------------
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
        # Mirror the kernel's own predicate: a token is stored only when the
        # token KPOOL ahead lands in a different tail block. Bounding every
        # slot would flag skipped tokens that are legitimately out of range.
        _s = tslot.to(torch.int64)
        _blk = torch.div(_s, kpool, rounding_mode="floor")
        _ahead = torch.full_like(_s, -1)
        if _s.numel() > kpool:
            _ahead[:-kpool] = _s[kpool:]
        _same = (_ahead >= 0) & (
            torch.div(_ahead, kpool, rounding_mode="floor") == _blk
        )
        # torch.where keeps the shape fixed; a boolean index would not.
        _written = (_s >= 0) & ~_same
        _kpool_tail_check(
            "seed",
            torch.where(_written, _blk, torch.full_like(_blk, -1)).max(),
            tail_kv_cache,
        )
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
        # Decode writes every valid tail slot; no ahead-predicate here.
        _s = tail_slot_mapping.to(torch.int64).reshape(-1)
        _b = torch.div(_s, pool_size, rounding_mode="floor")
        _kpool_tail_check(
            "decode",
            torch.where(_s >= 0, _b, torch.full_like(_b, -1)).max(),
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

    text = text.replace(PREAMBLE_ANCHOR, PREAMBLE, 1)
    text = text.replace(SEED_ANCHOR, SEED_PATCHED, 1)
    text = text.replace(DECODE_ANCHOR, DECODE_PATCHED, 1)
    path.write_text(text, encoding="utf-8")

    print(f"detector installed: {path}")
    print("enable with GLM_KPOOL_TAIL_BOUNDS=1 on the server process")


if __name__ == "__main__":
    main()
