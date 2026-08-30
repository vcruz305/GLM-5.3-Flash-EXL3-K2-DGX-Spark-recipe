# GLM-5.3 K-pool tail: out-of-bounds slot mapping

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

### Next step

Determine, at the seed call site, whether `positions` is None and which of the
two construction sites produced the tail metadata. That decides between
"populate `positions` for the tail group" and "route the tail group to
`KpoolTailMetadataBuilder`". Both are small; picking without measuring is how
the previous attempt was wasted.

Independently, both kernels should bounds-check their destination block. A guard
cannot be wrong, but it is not sufficient: if the mapping stays broken, guarding
converts a crash into silently dropped tail writes, which in a 2-bit model means
fluent wrong output rather than a loud failure.

## Detecting it on your own build

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
