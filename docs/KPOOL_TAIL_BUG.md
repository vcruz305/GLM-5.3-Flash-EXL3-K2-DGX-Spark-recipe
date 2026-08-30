# GLM-5.3 K-pool tail: out-of-bounds slot mapping

**Status: fixed.** `scripts/patch_kpool_tail_positions.py`, shipped in the
`spark-vllm` wheels from 2026-08-30. Validated at 65,536 context with CUDA
graphs on: the `ifeval:1300` prompt generated 4,096 and 8,192 tokens to
completion with the engine alive, and under `--enforce-eager` the device-side
detector counted 19,575 decode-path tail updates with zero out of bounds.

Root cause for the CUDA illegal memory accesses seen on `Glm5NextForConditionalGeneration`
in vLLM. Affects any quantization; it is model-attention plumbing, not EXL3.

## Summary

`KpoolTailSpec` declares a **one-block circular scratch cache**, one block per
request. Its slot mapping is nevertheless computed by the generic paged path,
which indexes `block_table[req, pos // block_size]`. That row is **one entry
wide**, so every token at position >= `block_size` reads past it and the mapping
is filled with whatever memory follows. The kernels then write to those
addresses without bounds-checking the block index.

## The two sides of the mismatch

`vllm/v1/kv_cache_interface.py`:

```python
class KpoolTailSpec(SlidingWindowSpec):
    """One-block circular scratch cache for a kpool indexer's raw tail."""

    def max_admission_blocks_per_request(self, ...) -> int:
        return 1

    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return 1
```

`vllm/v1/worker/block_table.py`:

```python
self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block
self.block_table = self._make_buffer(
    self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
)          # -> shape (max_num_reqs, 1) for the tail group

def compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
    ...
    assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT
    _COMPUTE_SLOT_MAPPING_KERNEL(
        ..., positions, self.block_table.gpu, self.block_table.gpu.stride(0),
        self.block_size, self.slot_mapping.gpu, ...)
```

`SlotMappingMode` offers only `TOKEN_TO_KV_SLOT` and `NONE`. Mamba groups opt
out with `NONE`. The tail group does not, so it gets standard paged addressing
against a one-entry row.

The consuming kernel documents the addressing it actually wants, in
`vllm/models/glm5next/nvidia/ops/kpool_compress.py`:

> ``tslot = block * KPOOL + pos % KPOOL``; the destination is
> ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``

`block` there is the request's single tail block. It is not `pos // block_size`.

## Why every observed symptom follows

| Observation | Explanation |
|---|---|
| Destination blocks of 271, 1631, 12927, 15207, 34303 against a ~186-block cache, at fixed capacity | not a systematic wrong stride; it is garbage read past a one-entry row |
| Long **generations** trigger it more reliably than long prompts | `pos` keeps climbing through decode, so `pos // block_size` walks further past the row |
| Faults are intermittent across runs and builds | the tail view's offset in the shared pool decides whether a write lands on another layer or outside the allocation |
| `--max-num-seqs 1` still fails | the table is then `(1, 1)`; any position >= 4 is already past the whole buffer |
| Short prompts are safe | positions below `block_size` index entry 0, which is correct |

Two kernels consume the mapping and neither bounds-checks the block index:

- `_kpool_tail_seed_kernel` (prefill seed), guards only `t < 0`
- `_kpool_decode_update_batched_kernel` (decode update)

## Reproducer

One request, no eval harness:
[`scripts/repro_kpool_tail_overrun.sh`](../scripts/repro_kpool_tail_overrun.sh).

76-token prompt, 32,768-token generation. The constraints are close to
unsatisfiable, so the model loops in thinking and runs to its budget, which is
what drives `pos` high enough to matter.

A clean run is **not** proof a build is unaffected. Whether the write faults or
corrupts silently depends on where that layer's view sits in the pool.

## The correct mapping already exists, and is conditionally skipped

An earlier revision of this document proposed adding a one-block mapping. That
was wrong in an instructive way: **the correct code is already there.** In
`vllm/v1/attention/backends/mla/indexer.py`:

```python
def compute_kpool_tail_slot_mapping(...):
    """Map every token to its request's one circular tail block."""
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
```

That is exactly `block_table[req, 0] * KPOOL + pos % KPOOL`, the addressing the
kernel documents. There is even a dedicated `KpoolTailMetadataBuilder` described
as building "only the circular slot mapping needed by the storage-only tail".

The defect is its caller:

```python
slot_mapping = common_attn_metadata.slot_mapping   # generic paged mapping
positions = common_attn_metadata.positions
if positions is not None:                          # silent fallthrough
    slot_mapping = compute_kpool_tail_slot_mapping(...)
```

When `positions` is None the correction is skipped and the **generic paged
mapping is used unchanged** — the one that indexes a one-entry row by
`pos // block_size`. A guard that silently degrades to incorrect addressing
rather than failing.

Note also the second construction site in the same file, which builds a
`DeepseekV32IndexerMetadata` from `compressed_slot_mapping` with no tail
correction at all. If the tail group's metadata comes from that path, the
correction never runs regardless of `positions`.

### What was tried and did not work

Clamping the block index inside the generic slot-mapping kernel:

```python
block_indices = tl.minimum(block_indices, block_table_stride - 1)
```

Applied cleanly and changed nothing: 48 overruns before, 48 after, identical
magnitudes. That is the evidence that the tail mapping does not come from
`block_table.py` at all, and it is why the fix must go where the mapping is
actually produced.

### Why positions are None: the hybrid model-state path drops them

Measured with a probe at the tail builder: `positions_is_none=True` on every
call. The reason is one call site. vLLM's V2 model runner builds attention
metadata per model family in `v1/worker/gpu/model_states/`:

- `default.py` (plain transformers) calls `build_attn_metadata(...,
  positions=input_batch.positions, ...)`.
- `mamba_hybrid.py` (every hybrid model, including GLM-5.3 with its KDA
  layers) calls `build_attn_metadata(...)` **without** `positions=`, and the
  parameter defaults to `None`.

So on hybrid models the K-pool tail builder never receives positions, the
one-block correction is skipped, and the generic paged mapping is used against
a one-entry block-table row.

This is present in the ZJY0516/vllm pin this recipe builds from (`878631b6`)
**and** in the `vllm/vllm-openai:glm53-flash-arm64-cu130` image
(`487ecf187`) that the TR3-4bpw / 2x-Spark recipes run. It is not specific to
EXL3, K2, or single-GPU serving. Recipes on that image have not measured it;
whether a given layout crashes or silently corrupts a neighbouring layer's
sparse-attention index is decided by pool geometry, and a completed run is not
evidence either way.

### The fix

`scripts/patch_kpool_tail_positions.py` adds the missing argument to the hybrid
path, mirroring `default.py`:

```python
            positions=input_batch.positions,
```

One line. With real positions present, the existing
`compute_kpool_tail_slot_mapping` runs and produces
`block_table[req, 0] * KPOOL + pos % KPOOL`, exactly the addressing the kernel
documents.

Two earlier attempts are recorded because both looked right and were not:

- clamping the block index inside the generic slot-mapping kernel changed
  nothing (48 overruns before and after) — the tail mapping does not come from
  that kernel once positions are present, and the clamp only masked garbage;
- synthesizing positions from `seq_lens` and `query_start_loc` at the tail
  builder made every tail write in-bounds during warmup, then died with an
  `Xid 13 Out Of Range Address` one second after graph capture. Deriving values
  the runner already has is the wrong layer to fix at.

### Second half of the fix: the corrected mapping must be written in place

With positions present, `compute_kpool_tail_slot_mapping` runs every step and
returned `slot_mapping.clone()`, a fresh allocation. CUDA graph capture records
that transient address; on replay the tail kernels read a buffer that has since
been freed or reused. Measured: the one-line positions fix booted and passed
1,500- and 4,000-token generations under `--enforce-eager`, and died with
`Xid 13 Out Of Range Address` one second after graph capture with graphs on.
The patch therefore also makes the function write the tail group's persistent
buffer in place, which is the correct semantics anyway: that buffer is the tail
group's slot mapping.

### A measurement lesson that cost several boots

Python-side instrumentation inside a CUDA-graph-captured op runs at capture
time and never again: real requests replay the graph. A counter that prints
from Python will report zero on a build that is writing out of bounds on every
request. The detector this recipe ships accumulates in **device tensors updated
by the captured kernels**, so replays update it. Only that kind of counter is
evidence.

### Decode-path proof

With both halves of the fix applied, at `max-model-len` 65536 under
`--enforce-eager` with `CUDA_LAUNCH_BLOCKING=1`, the `ifeval:1300` prompt
(76 tokens) generated its full 4,096-token budget:

```text
KPOOL_TAIL_BOUNDS calls=19599 overruns=0 (seed 0/24, decode 0/19575) worst_block=2 tail_blocks=116
```

19,575 decode-path tail updates, every destination inside the 116-block tail
cache, highest block written 2. Before the fix the same path produced
destination blocks in the tens of thousands.

### Soak result on the shipped fix

`scripts/soak.sh` against the fixed build at 65,536 context under
`--enforce-eager` with the detector armed: three 4,096-token generations,
12,288 tokens total, **57,551 decode-path tail updates, 0 out of bounds**,
highest block written 10 of 202, no faults. (The soak script's own verdict line
was broken by a `grep -c || echo 0` double-zero until this commit; the detector
counter is the evidence.)

## Detecting it on your own build

**Limitation:** the detector's device-side ops are themselves captured by CUDA
graphs and kill the engine on the first request in graph mode at 64k. Run the
detector with `--enforce-eager` (`ENFORCE_EAGER=1` in `serve_one_spark.sh`),
where it is proven to count on the decode path. It is inert unless
`GLM_KPOOL_TAIL_BOUNDS=1` is set, so the shipped wheel is unaffected.

`scripts/patch_kpool_tail_detector.py` installs an opt-in counter on both write
paths, the prefill seed and the decode update. It is inert unless
`GLM_KPOOL_TAIL_BOUNDS=1` is set on the server process, and it ships in the
runtime so no rebuild is needed to check a box.

```bash
# serve with the detector armed, then
SERVER_LOG=/path/to/server.log bash scripts/soak.sh
```

The soak generates ~20,000 tokens using near-unsatisfiable prompts, which is
what drives sequence position high enough to matter, and exits non-zero if any
out-of-bounds write is counted.

**Why a counter and not a crash test.** Every affected build performs the bad
writes. Whether one escapes its allocation depends on where each tail layer's
view sits in the shared pool, so a completed run proves nothing and four
completed runs prove nothing four times. Only the counter distinguishes
"unaffected" from "lucky".

## Acceptance test

Instrumentation that reproduces the kernel's own write predicate and bounds only
the blocks it will actually store to:

```text
before: 48 overrunning calls / 120 clean, at ctx 8192
after:   0 overrunning calls
```

Then the one-request reproducer must complete, and the sixcat suite must reach
120/120 rather than dying at `ifeval:1300`.

Validate on the overrun counter, not on the request succeeding. A change that
merely moves the allocation will still show overruns while appearing to work.

## Upstream

GLM-5.3-Flash support is not in vLLM main; it is
[PR #53906](https://github.com/vllm-project/vllm/pull/53906) by ZJY0516, open
with merge conflicts as of 2026-08-29. That thread already lists "KV cache
indexer page size mismatches causing wrong memory access" and "block table
addressing errors exceeding bounds" as open problems, reported by people running
NVFP4 rather than EXL3. This is a mechanism and a reproducer for that class.
