#!/usr/bin/env python3
"""Replicate upstream 57073552's BLOCK_STRIDE_ROWS fix in the SM120/121 sparse
decode path: compute the physical block row stride and pass it to
triton_convert_req_index_to_global_index, mirroring flashinfer_mla_sparse.py.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "flat_kv_row_view" in src:
    print("sm120 already fixed")
    sys.exit(0)

OLD_IMPORT = "    triton_convert_req_index_to_global_index,"
NEW_IMPORT = "    flat_kv_row_view,\n    triton_convert_req_index_to_global_index,"
assert src.count(OLD_IMPORT) == 1, "import anchor"
src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)

ANCHOR = "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n"
INSERT = (
    ANCHOR
    + "\n        _, block_stride_rows = flat_kv_row_view(\n"
    + "            kv_c_and_k_pe_cache, attn_metadata.block_size\n"
    + "        )\n"
)
assert src.count(ANCHOR) == 1, "topk anchor"
src = src.replace(ANCHOR, INSERT, 1)

OLD_CALL = "                BLOCK_SIZE=attn_metadata.block_size,\n                NUM_TOPK_TOKENS=topk_indices.shape[1],"
NEW_CALL = "                BLOCK_SIZE=attn_metadata.block_size,\n                BLOCK_STRIDE_ROWS=block_stride_rows,\n                NUM_TOPK_TOKENS=topk_indices.shape[1],"
n = src.count(OLD_CALL)
assert n >= 1, "call anchor"
src = src.replace(OLD_CALL, NEW_CALL)
open(path, "w", encoding="utf-8", newline="\n").write(src)
print(f"sm120 fixed: import + stride compute + {n} call site(s)")
