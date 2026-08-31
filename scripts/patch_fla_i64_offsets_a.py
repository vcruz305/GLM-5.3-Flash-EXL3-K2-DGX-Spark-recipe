#!/usr/bin/env python3
"""Fix the i32 offset overflow in the FLA chunked delta-rule H kernel, v2.

Cast the varlen offsets to int64 inside the base-pointer products (the idiom
the kernel already uses for i_t), keeping make_block_ptr offsets i32.
boh * H*V*K overflows int32 for boh > 2047 (any varlen call with T > 131,072);
proven standalone: T=163,840 raises CUDA illegal memory access.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "boh.to(tl.int64) * H" in src:
    print("already fixed")
    sys.exit(0)
pairs = [
    ("((boh * H + i_h)", "((boh.to(tl.int64) * H + i_h)"),
    ("((bos * H + i_h)", "((bos.to(tl.int64) * H + i_h)"),
]
changed = 0
for old, new in pairs:
    n = src.count(old)
    assert n >= 1, f"anchor missing: {old!r}"
    src = src.replace(old, new)
    changed += n
open(path, "w", encoding="utf-8", newline="\n").write(src)
print(f"cast {changed} base-pointer products to int64 in {path}")
