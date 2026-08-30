#!/usr/bin/env python3
"""Fix the K-pool tail out-of-bounds slot mapping on hybrid models.

Root cause, measured on the Spark with a probe at the tail metadata builder:
``common_attn_metadata.positions`` is None on every call, so the correction
that already exists in ``compute_kpool_tail_slot_mapping`` (indexer.py) is
skipped and the tail group falls through to the generic paged mapping. That
mapping indexes a one-entry block-table row by ``pos // block_size``, produces
garbage block ids, and both kpool kernels write through them.

Why positions are None: GLM-5.3 is a hybrid (KDA + sparse MLA) model, so its
attention metadata is built by ``model_states/mamba_hybrid.py``. That path
calls ``build_attn_metadata(...)`` WITHOUT ``positions=``, while the plain
transformer path in ``model_states/default.py`` passes
``positions=input_batch.positions``. The tail builder is therefore never handed
positions on any hybrid model. This is present in the ZJY0516/vllm pin used by
this recipe (878631b6) and in the vllm-openai:glm53-flash image (487ecf187).

The fix mirrors default.py: pass the real positions through the hybrid path.
One line, no new plumbing, no synthesized values.

Applies to vllm/v1/worker/gpu/model_states/mamba_hybrid.py, which the recipe
patch stack does not touch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ANCHOR = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""

PATCHED = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # Hybrid models never passed positions here, unlike default.py.
            # The K-pool tail builder needs them: without positions it skips
            # compute_kpool_tail_slot_mapping and uses the generic paged
            # mapping against a one-entry block-table row, which writes the
            # tail cache out of bounds. See docs/KPOOL_TAIL_BUG.md.
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""

REL = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"

# Second edit. Once positions are present, compute_kpool_tail_slot_mapping runs
# every step and returned a fresh clone. CUDA graph capture records that
# transient address; replay then reads a buffer that has since been freed or
# reused, which is the illegal memory access seen with graphs on and not under
# --enforce-eager. The caller's slot_mapping is the tail group's own persistent
# buffer, so writing in place is the correct semantics as well as the safe one.
REL2 = "vllm/v1/attention/backends/mla/indexer.py"
ANCHOR2 = """    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
"""
PATCHED2 = """    # In place: slot_mapping is the tail group's persistent buffer. A fresh
    # clone here is captured by CUDA graphs at a transient address and read
    # back stale on replay (illegal memory access). See docs/KPOOL_TAIL_BUG.md.
    out = slot_mapping
    if num_actual_tokens == 0:
        return out
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="vLLM source root")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    path = Path(args.source) / REL
    text = path.read_text(encoding="utf-8")

    path2 = Path(args.source) / REL2
    text2 = path2.read_text(encoding="utf-8")

    if args.revert:
        if PATCHED in text:
            path.write_text(text.replace(PATCHED, ANCHOR, 1), encoding="utf-8")
            print(f"reverted: {path}")
        if PATCHED2 in text2:
            path2.write_text(text2.replace(PATCHED2, ANCHOR2, 1), encoding="utf-8")
            print(f"reverted: {path2}")
        return

    if "Hybrid models never passed positions here" in text:
        print(f"already patched: {path}")
    else:
        if text.count(ANCHOR) != 1:
            raise SystemExit(
                f"{path}: expected exactly one anchor, found {text.count(ANCHOR)}. "
                "The hybrid model-state call has changed; re-derive the patch."
            )
        path.write_text(text.replace(ANCHOR, PATCHED, 1), encoding="utf-8")
        print(f"patched: {path}")
        print("mamba_hybrid.py now passes positions=input_batch.positions")

    if "In place: slot_mapping is the tail group" in text2:
        print(f"already patched: {path2}")
        return
    if text2.count(ANCHOR2) != 1:
        raise SystemExit(f"{path2}: expected exactly one anchor, found {text2.count(ANCHOR2)}")
    path2.write_text(text2.replace(ANCHOR2, PATCHED2, 1), encoding="utf-8")
    print(f"patched: {path2}")
    print("compute_kpool_tail_slot_mapping now writes the persistent buffer in place")


if __name__ == "__main__":
    main()
