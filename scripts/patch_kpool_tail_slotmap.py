#!/usr/bin/env python3
"""Fix the K-pool tail out-of-bounds slot mapping.

`KpoolTailSpec` is a one-block circular scratch cache: both
`max_admission_blocks_per_request` and `max_num_blocks_per_req` return 1, so its
block table row is a single entry. Its slot mapping is nevertheless computed by
the generic paged path, whose kernel does:

    block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK
                    + local_block_offsets // block_size
    block_numbers = tl.load(block_table_ptr + row_offset + block_indices,
                            mask=mask & is_local, other=0)

The mask guards token validity. Nothing bounds `block_indices` against the row
width. For the tail group the row is one entry wide, so every token at position
>= block_size reads past it and the mapping is filled with adjacent memory. Both
kpool kernels then write to those addresses.

The fix clamps the block index to the row:

    block_indices = tl.minimum(block_indices, block_table_stride - 1)

For the tail group `block_table_stride == 1`, so the index pins to 0 and the
slot becomes `block_table[req, 0] * block_size + pos % block_size`. That is
exactly the addressing `_kpool_tail_seed_kernel` documents:
`tslot = block * KPOOL + pos % KPOOL`, where block is the request's single tail
block. The one-block circular contract falls out of the clamp.

For every other group the clamp is identity: a request never legitimately needs
more blocks than its row holds, so `block_indices < block_table_stride` already.
Reading past the row is an out-of-bounds read in any group, not just this one.

Upstream would likely prefer this expressed as a third `SlotMappingMode`
alongside `TOKEN_TO_KV_SLOT` and `NONE`, selected for `KpoolTailSpec` in
`gpu_model_runner`. That is a larger change touching the enum, the compile key
and the mode plumbing. This clamp is the minimal form with the same behaviour
and no new compile-key dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ANCHOR = """            block_indices = (
                virtual_block_indices * BLOCKS_PER_KV_BLOCK
                + local_block_offsets // block_size
            )
            block_numbers = tl.load(
                block_table_ptr + row_offset + block_indices,
                mask=mask & is_local,
                other=0,
            ).to(tl.int64)
"""

PATCHED = """            block_indices = (
                virtual_block_indices * BLOCKS_PER_KV_BLOCK
                + local_block_offsets // block_size
            )
            # Never index past the request's block-table row. A row is
            # max_num_blocks_per_req wide; nothing above bounds block_indices
            # against it, and the mask only guards token validity.
            #
            # KpoolTailSpec is a one-block circular scratch cache whose row is a
            # single entry, so without this clamp every token at position >=
            # block_size reads adjacent memory and the slot mapping is filled
            # with garbage that the kpool kernels then write through.
            #
            # Clamping pins that group to entry 0, which yields
            #   block_table[req, 0] * block_size + pos % block_size
            # the addressing _kpool_tail_seed_kernel documents. For every other
            # group block_indices is already inside the row, so this is
            # identity.
            block_indices = tl.minimum(block_indices, block_table_stride - 1)
            block_numbers = tl.load(
                block_table_ptr + row_offset + block_indices,
                mask=mask & is_local,
                other=0,
            ).to(tl.int64)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="vLLM source root")
    ap.add_argument(
        "--revert", action="store_true", help="undo the clamp instead of applying it"
    )
    args = ap.parse_args()

    path = Path(args.source) / "vllm/v1/worker/block_table.py"
    text = path.read_text(encoding="utf-8")

    if args.revert:
        if PATCHED not in text:
            print(f"not patched: {path}")
            return
        path.write_text(text.replace(PATCHED, ANCHOR, 1), encoding="utf-8")
        print(f"reverted: {path}")
        return

    if "block_table_stride - 1" in text:
        print(f"already patched: {path}")
        return
    if text.count(ANCHOR) != 1:
        raise SystemExit(
            f"expected exactly one anchor in {path}, found {text.count(ANCHOR)}. "
            "The slot-mapping kernel has changed; re-derive the patch."
        )

    path.write_text(text.replace(ANCHOR, PATCHED, 1), encoding="utf-8")
    print(f"patched: {path}")
    print("clamped block_indices to the block-table row width")


if __name__ == "__main__":
    main()
