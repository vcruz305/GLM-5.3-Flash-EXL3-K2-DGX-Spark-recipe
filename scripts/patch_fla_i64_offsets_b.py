#!/usr/bin/env python3
"""FLA i64 offset fixes, round 3: the o-kernel's chunk-state base and the
missed bos*Hg advance. Together with flafix_v2 (chunk_delta_h boh/bos), this
covers every chunk-index-scale (x H*V*K) product in the vendored library.
"""
import sys

root = sys.argv[1]

kda = root + "/kda.py"
src = open(kda, encoding="utf-8").read()
OLD = "            h + (i_tg * H + i_h) * K * V,"
NEW = "            h + (i_tg.to(tl.int64) * H + i_h) * K * V,"
if NEW in src:
    print("kda.py already fixed")
else:
    assert src.count(OLD) == 1, src.count(OLD)
    open(kda, "w", encoding="utf-8", newline="\n").write(src.replace(OLD, NEW, 1))
    print("kda.py: o-kernel h base cast to int64")

cdh = root + "/chunk_delta_h.py"
src = open(cdh, encoding="utf-8").read()
OLD2 = "    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)"
NEW2 = "    k += ((bos.to(tl.int64) * Hg + i_h // (H // Hg)) * K).to(tl.int64)"
if NEW2 in src:
    print("chunk_delta_h.py k-advance already fixed")
elif OLD2 in src:
    open(cdh, "w", encoding="utf-8", newline="\n").write(src.replace(OLD2, NEW2, 1))
    print("chunk_delta_h.py: k advance cast to int64")
else:
    print("chunk_delta_h.py: k-advance anchor not found (skipped)")
