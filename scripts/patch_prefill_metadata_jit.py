#!/usr/bin/env python3
"""Port vLLM PR #54345 (de-specialize BuildPrefillChunkMetadataKernel scalars)
into the installed indexer.py, plus add KPOOL_TAIL to the warmup backend set.
Exact-match with assertions: aborts loudly if the tree differs, never corrupts."""
import ast
import sys

IDX = sys.argv[1]   # indexer.py
WU = sys.argv[2]    # sparse_mla_triton_warmup.py

s = open(IDX).read()
edits = []

# 1) CompileKey: drop the 5 specialized scalar fields
old = ("        query_slice_start: int\n"
       "        query_slice_stop: int\n"
       "        DCP_RANK: int\n"
       "        DCP_WORLD: int\n"
       "        DCP_INTERLEAVE: int\n"
       "        BLOCK_SIZE: int\n")
new = "        BLOCK_SIZE: int\n"
edits.append(("CompileKey fields", old, new))

# 2) kernel decorator: add do_not_specialize
old = ("    @staticmethod\n"
       "    @triton.jit\n"
       "    def kernel(")
new = ("    @staticmethod\n"
       "    @triton.jit(\n"
       "        do_not_specialize=[\n"
       '            "query_slice_start",\n'
       '            "query_slice_stop",\n'
       '            "DCP_RANK",\n'
       '            "DCP_WORLD",\n'
       '            "DCP_INTERLEAVE",\n'
       "        ]\n"
       "    )\n"
       "    def kernel(")
edits.append(("kernel decorator", old, new))

# 3) dispatch(): drop the 5 params
old = ("        query_slice_start: int,\n"
       "        query_slice_stop: int,\n"
       "        DCP_RANK: int,\n"
       "        DCP_WORLD: int,\n"
       "        DCP_INTERLEAVE: int,\n"
       "        BLOCK_SIZE: int,\n"
       "        COMPRESS_RATIO: int,\n"
       "        input_variant: TritonPointerInputVariant,\n"
       "    ) -> CompileKey:\n"
       "        return self.CompileKey(\n"
       "            query_slice_start=query_slice_start,\n"
       "            query_slice_stop=query_slice_stop,\n"
       "            DCP_RANK=DCP_RANK,\n"
       "            DCP_WORLD=DCP_WORLD,\n"
       "            DCP_INTERLEAVE=DCP_INTERLEAVE,\n"
       "            BLOCK_SIZE=BLOCK_SIZE,\n")
new = ("        BLOCK_SIZE: int,\n"
       "        COMPRESS_RATIO: int,\n"
       "        input_variant: TritonPointerInputVariant,\n"
       "    ) -> CompileKey:\n"
       "        return self.CompileKey(\n"
       "            BLOCK_SIZE=BLOCK_SIZE,\n")
edits.append(("dispatch params", old, new))

# 4) compile(): replace the 5 compile_key scalars with literals
old = ("            compile_key.query_slice_start,\n"
       "            compile_key.query_slice_stop,\n"
       "            compile_key.DCP_RANK,\n"
       "            compile_key.DCP_WORLD,\n"
       "            compile_key.DCP_INTERLEAVE,\n")
new = ("            0,\n"
       "            1,\n"
       "            0,\n"
       "            1,\n"
       "            1,\n")
edits.append(("compile literals", old, new))

# 5) get_warmup_keys(): drop the scalar args in the dispatch trace call
old = ("            query_slice_start=WarmupIntRange(0, 2),\n"
       "            query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),\n"
       "            DCP_RANK=dcp_rank,\n"
       "            DCP_WORLD=dcp_world,\n"
       "            DCP_INTERLEAVE=dcp_interleave,\n"
       "            BLOCK_SIZE=self.BLOCK_SIZE,\n")
new = "            BLOCK_SIZE=self.BLOCK_SIZE,\n"
edits.append(("warmup_keys dispatch args", old, new))

for label, o, n in edits:
    c = s.count(o)
    if c != 1:
        print("ABORT: %r matched %d times (expected 1)" % (label, c))
        sys.exit(2)
    s = s.replace(o, n, 1)

ast.parse(s)   # verify it still parses
open(IDX, "w").write(s)
print("indexer.py: 5 hunks applied, parses OK")

# 6) warmup set: add KPOOL_TAIL (+ dsv4 / b12x names)
w = open(WU).read()
old = 'frozenset({"DEEPSEEK_V32_INDEXER"})'
new = 'frozenset({"DEEPSEEK_V32_INDEXER", "KPOOL_TAIL", "DEEPSEEK_V4_INDEXER", "B12X_MLA_SPARSE"})'
if w.count(old) < 1:
    print("ABORT: warmup backend set not found")
    sys.exit(3)
w = w.replace(old, new)
ast.parse(w)
open(WU, "w").write(w)
print("warmup module: KPOOL_TAIL added, parses OK")
