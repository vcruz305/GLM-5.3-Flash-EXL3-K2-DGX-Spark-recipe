# Measurement log

All current ladder rows use one DGX Spark, TP=1, `EXL3_FUSED_MOE=1`, target
KV FP8, one sequence, temperature 0, thinking off, streamed OpenAI HTTP, one
warm-up plus three measured requests, and 400 requested completion tokens.
The exact four-prompt harness is [`scripts/bench_ladder.py`](../scripts/bench_ladder.py).

## Runtime A/B at 64k, MTP k=2

The response template and prompt rendering were held byte-identical. Each row
is the median of three measured requests. These numbers are intentionally kept
separate from the older 128-token microbenchmark in the README.

| Workload | Older runtime | Current local runtime | Delta |
|---|---:|---:|---:|
| prose | 14.2993 tok/s | 15.1284 tok/s | +5.8% |
| structured | 20.5624 tok/s | 20.5982 tok/s | +0.2% |
| code | 16.1909 tok/s | 16.1220 tok/s | -0.4% |
| math | 16.5053 tok/s | 17.3159 tok/s | +4.9% |
| arithmetic mean | 16.8895 tok/s | 17.2911 tok/s | +2.4% |

Current local runtime:

- vLLM `878631b6079d2cf9fb80830ef9cb41b43aded098`
- ExLlamaV3 `17bc3923259ffd48aab742edd261a0ca45d55459` / 1.4.4
- FlashInfer 0.6.18rc10
- PyTorch 2.13.0+cu130
- template SHA-256 `96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a`

The MTP change is small and workload-dependent; this update is not a universal
large speedup. Structured output was already near-perfectly accepted and was
flat, while prose and math improved.

## Current-runtime MTP depth ladder at 64k

The same four prompts, template, 400-token completion length, one warm-up, and
three measured runs were used for k=2 through k=4. Values below are per-workload
medians; the mean is the arithmetic mean of those four medians.

| Workload | MTP k=2 | MTP k=3 | MTP k=4 |
|---|---:|---:|---:|
| prose | **15.1284** tok/s | 14.2146 tok/s | 12.7166 tok/s |
| structured | 20.5982 tok/s | 24.1161 tok/s | **26.2140** tok/s |
| code | **16.1220** tok/s | 15.7839 tok/s | 15.5860 tok/s |
| math | **17.3159** tok/s | 16.9723 tok/s | 16.8565 tok/s |
| arithmetic mean | 17.2911 tok/s | 17.7717 tok/s | **17.8433 tok/s** |

MTP k=4 is the aggregate speed leader by only 0.4% over k=3 and 3.2% over k=2,
but it is not the balanced default: it loses to k=2 on three of four varied
workloads and wins the mean through the highly predictable structured row.
That row accepted 91.3% of all four proposed positions and reached 26.21
tok/s. The prose median accepted only 31.7% of proposed positions and fell to
12.72 tok/s. MTP k=2 remains the safer daily-driver setting when workload
predictability is unknown; k=4 is appropriate for rigid schemas, sequences,
and other highly predictable output.

## Current-branch DFlash2 enablement ladder

The Spark has the corrected upstream checkpoint-update revision
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410` (2026-08-28), not the original
release `7d74cdd881ed7e32c31175984a67823127b66cfe`. Both files are
2,342,169,800 bytes, but the weight hashes differ: the original is
`8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683`
and the installed update is
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.
The download helper pins and verifies the latter. The position-128 failure
below was reproduced with that corrected checkpoint, so it is not evidence
that the first set of weights was accidentally used.

These are setup/correctness rungs, not throughput claims:

| Rung | Result | Fix or conclusion |
|---|---|---|
| unmodified GLM target interface | failed after the 91 GiB load: target did not expose EAGLE3 auxiliary states | add the five configured auxiliary-state taps, including mHC contraction |
| auxiliary states enabled | target and BF16 draft loaded; failed cache grouping because the generic unifier cannot pad sparse-MLA index pages | preserve the target groups and allocate regular draft pages in disjoint slabs |
| DFlash k=7, target-size draft manager page | reached health, then the first 128-token stream died with a CUDA illegal-memory access at absolute sequence position 128 | not an out-of-memory event; the request used only a fraction of KV capacity |
| DFlash k=7, draft-page cap 1024 | target block 9216, resolved draft block 1024; KV capacity rose to 24,940 tokens, but the same position-128 CUDA fault remained | rules out the oversized draft page and KV exhaustion |
| DFlash k=3, draft-page cap 1024 | target block 8960, resolved draft block 896; KV capacity rose to 40,140 tokens, but the same position-128 CUDA fault remained | rules out k=7 and the exact draft manager block as the trigger |
| DFlash k=3 greedy, cap 1024 | two complete 64-token streams: 12.40 tok/s cold, 12.00 tok/s warm; 26.3% draft acceptance; the 128-token stream again died at absolute position 128 | sampler choice and probabilistic-logit caching are not the cause; greedy is slower |
| DFlash k=3 greedy, eager, prefix cache off, synchronous CUDA launches | a 100-token prompt plus 16 output tokens completed; a 120-token prompt plus 16 output tokens failed as it crossed position 128 | the synchronous traceback identifies `kpool_decode_update_and_maybe_write_cache_batched` in the target sparse indexer; CUDA graphs, prefix reuse, and asynchronous error attribution are ruled out |
| DFlash k=3, sequential per-token K-pool update experiment | failed at the same absolute position 128 boundary | the batched update call is not the root cause; the sequential workaround is rejected and is not installed by this recipe |
| DFlash k=3 BF16, Triton draft attention, manager cap 1024 | resolved DFlash group 6 to manager 896 / kernel 896 / split 1; completed 120 prompt + 32 output and 32 prompt + 256 output, with health 200 afterward, then a full four-workload ladder | crosses position 128. Every sequence was inside its own 896-token page, so this rung alone does not cover page transitions; the page-crossing rung below does |
| DFlash k=3 BF16, Triton draft attention, page-crossing probes | 832 inside page 0, 932 across the first boundary, 1832 across two boundaries, and 32 prompt + 1024 output crossing during decode; all returned on `length`, zero CUDA faults, health 200 afterward | Triton survives page transitions in prefill and in decode. The Triton configuration is validated beyond one page and the ladder stands |
| DFlash k=3 BF16, FlashAttention draft attention, manager cap 128 | resolved DFlash group 6 to manager 128 / kernel 128 / split 1, so no page split exists; 100 prompt + 16 output passed and 120 prompt + 16 output faulted | refutes the virtual-page-split explanation, and locates the fault at the first page transition. Also collapsed the pool to 16,384 KV tokens at 2.00x concurrency against 40,140 at the 1024 cap |

The mixed-cache planner treats `GLM_DFLASH_MANAGER_BLOCK_SIZE` as an upper
bound and selects the largest valid divisor of the target manager block. This
keeps the scheduler least common multiple equal to the target block while
letting regular draft attention use smaller physical pages. The specialized
allocation remains LBHNC and gives every auxiliary group its own non-overlapping
slab; sparse MLA rows are never padded.

No DFlash row is promoted into the speed leaderboard until a complete stream
crosses position 128 and the same benchmark harness finishes. A partial stream
remains useful for debugging but is not a throughput result.

### DFlash k=3 BF16, Triton draft attention, at 8k

The Triton rung cleared both halves of that gate. The two boundary probes
completed with `finish_reason: length`, meaning they ran out of requested
tokens rather than faulting: 120 prompt plus 32 output in 4.461 s, and 32
prompt plus 256 output in 19.802 s. The server answered health 200 afterward.
The same `bench_ladder.py` harness then finished all four workloads at one
warm-up plus three measured runs, 400 requested completion tokens.

| Workload | Decode tok/s | Draft acceptance | Verified tokens/step |
|---|---:|---:|---:|
| prose | 12.1284 | 26.31% | 1.7892 |
| structured | **22.5627** | 79.27% | 3.3782 |
| code | 13.2316 | 32.51% | 1.9752 |
| math | 14.5249 | 40.33% | 2.2099 |
| arithmetic mean | 15.6119 | | |

This is the first DFlash configuration on this pack to produce a complete
four-workload ladder. It clears the 9.6 to 9.8 tok/s no-spec floor by a wide
margin and it beats MTP k=2 on the structured row, 22.56 against 20.60.

Every sequence in this ladder is shorter than the 896-token DFlash page, so the
ladder on its own does not show that the configuration survives a page
transition. A dedicated page-crossing rung was run for exactly that reason and
it passed at 932, 1832, and a decode-time crossing. See the page-transition
section below. The ladder stands.

### DFlash k=3 against MTP k=2 at matched context

The ladder above ran at 8192 while every MTP ladder ran at 65536, so it could
not be ranked against MTP without mixing attention page size into the result.
An MTP k=2 ladder was therefore run at 8192, matched on context, batched
tokens, GPU memory utilisation, chat template, harness settings and completion
length. The speculative method is the only variable.

| Workload | MTP k=2 | DFlash k=3 | MTP delta | MTP accept | DFlash accept |
|---|---:|---:|---:|---:|---:|
| prose | **14.8806** | 12.1284 | +22.7% | 54.7% | 26.3% |
| structured | 20.4476 | **22.5627** | -9.4% | 97.1% | 79.3% |
| code | **15.9834** | 13.2316 | +20.8% | 64.6% | 32.5% |
| math | **16.7781** | 14.5249 | +15.5% | 70.9% | 40.3% |
| arithmetic mean | **17.0224** | 15.6119 | +9.0% | | |

MTP k=2 wins the mean by 9.0% and wins three of four workloads outright. DFlash
k=3 wins structured by 9.4%, and it wins there **despite lower acceptance**,
79.3% against 97.1%, because k=3 proposes a deeper draft: 3.38 verified tokens
per step against 2.94. That is the same shape as the MTP depth ladder, where
k=4 also won structured and lost everywhere else. Deeper speculation pays only
where output is highly predictable.

This also retires the context caveat. MTP k=2 measures 17.0224 at 8k against
17.2911 at 64k, a 1.6% difference, so page size was never carrying the earlier
cross-context comparison. The DFlash deficit is real and it is not an artifact
of the 8k-versus-64k mismatch.

**Conclusion: DFlash2 does not displace native MTP k=2 on this pack.** It needs
a sidecar checkpoint, an EAGLE3 auxiliary-state patch, a mixed-cache planner,
and a Triton-only draft attention backend, and it still ends up 9.0% slower on
the mean. It is worth keeping only for rigid schema workloads, and even there
MTP k=4 reaches 26.21 tok/s at 64k against DFlash's 22.56.

One property of DFlash is worth recording anyway: its acceptance spread across
workloads is far wider than MTP's, 26.3% to 79.3% against MTP's 54.7% to 97.1%.
A speculator whose acceptance collapses on prose is a poor default even when
its mean looks competitive.

The `scheduled_spec_decode_tokens=[-1, ...]` values in the scheduler failure
dump are intentional shape padding for the first speculative-shaped step. They
are not invalid candidates leaking into the model. Do not chase them.

### The virtual-page-split explanation is refuted

An earlier revision of this document proposed that the fault came from a
virtual-page split: with Triton, manager and kernel blocks are both 896, while
with FlashAttention the 256-wide DFlash head uses a fixed 128-token kernel page
inside the 896-token manager page, and the fault followed that first page
transition. That explanation is wrong.

The test was to keep FlashAttention and cap the draft manager block at 128, so
that manager and kernel pages are the same size and no split exists. The planner
resolved it exactly as intended:

```text
GLM DFlash manager block: target=8960 requested_max=128 resolved=128
DFLASH_BOUNDARY_GEOMETRY static gids=[6] manager=[128] kernel=[128] split=[1]
GLM5 KV group 6: layers=5 types=['SlidingWindowSpec'] manager_blocks=[128] pages=[524288]
```

The 120 prompt plus 32 output probe still died with an illegal memory access and
took the engine down with `EngineDeadError`. Split 1 is therefore present in both
the passing configuration (Triton, 896/896/1) and the failing one
(FlashAttention, 128/128/1), so the split is not the discriminator. The draft
attention backend is.

Two secondary observations from that run. Capping the draft manager block at 128
collapsed the pool to 16,384 KV tokens at 2.00x concurrency, against 40,140 at
the 1024 cap, so block 128 is expensive independently of the fault. And the
error surfaced at `async_utils.get_output` on a copy-event synchronize, which is
an asynchronous attribution point; the geometry log reached sequence 135 before
the engine died, but that does not establish that the faulting kernel ran at 135
rather than earlier. This run does not pin the fault position.

MTP did not fail because it has no DFlash dense-SWA auxiliary group.

### The fault tracks the first DFlash page transition

Repeating the 128-cap configuration with `CUDA_LAUNCH_BLOCKING=1`, eager
execution, and prefix caching off gives a synchronous traceback that names the
faulting kernel directly:

```text
vllm/models/glm5next/nvidia/attention.py:590   in forward   -> self.mla_attn(...)
vllm/model_executor/layers/mla.py:225          in forward   -> self.indexer(...)
vllm/models/glm5next/nvidia/attention.py:392   in forward   -> self.indexer_op(...)
vllm/model_executor/layers/sparse_attn_indexer_kpool.py:722 -> kpool_decode_update_and_maybe_write_cache_batched(
vllm/models/glm5next/nvidia/ops/kpool_compress.py:676       -> _kpool_decode_update_batched_kernel[(num_requests,)](
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

The faulting kernel is `_kpool_decode_update_batched_kernel`, inside the
**target** model's sparse MLA K-pool indexer. It is not a draft kernel.

The graduated probes locate it precisely. At manager and kernel block 128:

| Probe | Total sequence | Page | Result |
|---|---:|---|---|
| 100 prompt + 16 output | 116 | inside page 0 (0 to 127) | pass |
| 120 prompt + 16 output | 136 | crosses into page 1 | **fault** |

That raised a fair objection to the Triton rung. Its page was 896 tokens and its
longest sequence was 492, so nothing in it had crossed a page boundary either.
If the fault were simply "the first page transition at any page size," Triton
would only have looked healthy because its page was large.

That objection was tested directly and does not hold. Same Triton configuration,
probes chosen against the 896-token page:

| Probe | Total sequence | Boundary | Result |
|---|---:|---|---|
| 800 prompt + 32 output | 832 | inside page 0 | pass |
| 900 prompt + 32 output | 932 | crosses the first boundary | pass |
| 1800 prompt + 32 output | 1832 | crosses two boundaries | pass |
| 32 prompt + 1024 output | 1056 | crosses during decode, not prefill | pass |

All four returned on `finish_reason: length`, the server log contains zero
illegal-memory-access events, and health was 200 afterward.

So the draft attention backend **is** the discriminator, and the mechanism is not
page splitting and not page transitions as such:

- FlashAttention faults at its first page transition, with or without a split.
- Triton crosses page transitions cleanly, in prefill and in decode, more than
  once per sequence.

What remains unexplained is why a **target** kernel,
`_kpool_decode_update_batched_kernel`, reads out of bounds as a function of the
**draft's** attention backend. The leading candidate is that the FlashAttention
draft path leaves the target's kpool block-table indexing pointing outside its
slab once a second draft page exists, since a single-page sequence resolves
through `block_table[0]` and hides any stride or offset error.

Two notes for anyone picking this up. Upstream has fixed this exact bug class
once already for the draft's own metadata, in #53002 and its reapply #53336,
by sourcing attention geometry from the group's `kv_cache_spec` instead of
model-wide config; both are already in the pinned build. And the kpool indexer
is not upstream code, so no upstream fix will reach it.

### Draft quantization decision gate

The BF16 sidecar is about 2.18 GiB. Roughly 72% of its source weight bytes are
the input `fc` plus five MLP gate/up/down triplets, so a dense EXL3 draft could
reduce draft bandwidth materially. It is not a one-flag conversion, however:

- the existing EXL3 plugin deliberately quantizes routed experts only;
- DFlash2's context-KV precompute slices native Q/K/V projection weights and
  uses a dense linear call, so quantizing attention would require rewriting
  that path;
- dynamic-convolution kernels, candidate-selector codebooks/projection, norms,
  embeddings, and the vocabulary head should remain native in a first version.

The first packaged-quantization probe is draft-only online FP8, with identical
target weights, prompts, sampling, and KV settings. Stock draft quantization on
this branch reaches `fc`, the five MLPs, and attention QKV/O. That is unsafe for
DFlash2 because its context-KV builder directly slices the dense QKV weight.
The recipe therefore applies `patch_dflash2_selective_quant.py`: DFlash2 QKV/O
remain BF16 while online FP8 reaches only `fc` and the 15 MLP matrices. Dynamic
convolution, selector, norms, embeddings, and the head also remain native.

A custom dense-EXL3 draft is justified only if BF16 DFlash beats no-spec and
profiling shows draft weight bandwidth remains material. Selective FP8 is a
useful directional A/B but still does not prove an EXL3 kernel will have the
same overheads. An EXL3 V1 should preserve the same module boundary: `fc` plus
the 15 MLP matrices only.

## 131072 context, MTP k=2

The context ladder previously stopped at 65536 with 128k recorded as not
attempted. It was attempted. The result is a partial pass, and it is worth
stating precisely because "128k works" and "128k allocates" are different
claims.

| Stage | Result |
|---|---|
| allocation at util 0.91 | **succeeds**, GPU KV cache 786,432 tokens, 6.00x concurrency for 131,072 tokens per request |
| `/v1` identity | serves, `max_model_len` reports 131072 |
| 65,408 prompt + 16 output | **pass**, 92.979 s wall, `finish_reason: length` |
| 130,944 prompt + 16 output | **fault**, HTTP 500, engine down with an illegal memory access |

Two things follow.

First, the KV pool is the same 786,432 tokens at 64k and at 128k. Raising
`max-model-len` does not buy more KV; it divides the same memory-bound pool
into fewer concurrent slots, 12.00x down to 6.00x. The ceiling here is the
91 GiB of weights, not the context flag.

Second, this fault and the DFlash page-transition fault are **the same
component**. Re-run with `CUDA_LAUNCH_BLOCKING=1` and eager execution, and the
synchronous traceback names a different kernel from the asynchronous one:

```text
sparse_attn_indexer_kpool.py:403   sparse_attn_indexer_kpool
kpool_compress.py:425              kpool_seed_tail_cache
                                   _kpool_tail_seed_kernel[(n,)](
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

An earlier revision of this document reported the EXL3 fused MoE
(`exl3.py:361 apply_exl3_fused_moe`) as the faulting frame, with the caveat
that launches were asynchronous. That caveat mattered: the MoE was simply the
next launch to notice, and the fault is in the **K-pool tail seed**, not in the
MoE. The chunk-size test pointed the same way, since halving
`--max-num-batched-tokens` to 1024 did not move the 98,304 threshold, which a
per-forward MoE buffer overflow would have.

So both open faults live in the GLM-5.3 sparse-MLA K-pool indexer, in two
different kernels:

| Fault | Kernel | Path |
|---|---|---|
| DFlash page transition | `_kpool_decode_update_batched_kernel` | decode |
| Long context ≥98,304 | `_kpool_tail_seed_kernel` | prefill seed |

`_kpool_tail_seed_kernel` computes its destination as `blk = t // KPOOL` and
stores at `(blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM`. The only guard is
`t < 0`. **Nothing checks that `blk` is inside the allocated tail cache**, so a
tail slot mapping that exceeds capacity writes off the end. That is the leading
hypothesis for both faults: the `KpoolTailSpec` group, which allocates with a
manager block size of 4, is mis-sized or mis-indexed, and neither kernel
bounds-checks its destination.

`scripts/patch_glm53_sm121_nope.py` modifies `KpoolTailSpec` and
`expand_pools_and_append_tail`, so the defect may be in this recipe's patch
rather than in the fork.

One measurement caveat. The 81,920 pass was recorded with the default serving
configuration. Under `--enforce-eager` with prefix caching off, 81,920 also
faults, so the ceiling is configuration dependent and 81,920 is verified only
for the defaults this recipe ships.

### Where the ceiling actually is

An ascending scan on one 131072 boot narrows it. The engine dies at the fault,
so the last passing length and the first failing length bracket the ceiling:

| Prompt tokens | Result | Wall |
|---:|---|---:|
| 65,408 | pass | 92.98 s |
| **81,920** | **pass** | 114.8 s |
| **98,304** | **fault** | — |

The failure at 98,304 has the same signature as the one at 130,944, down to the
frame: `exl3.py:361 apply_exl3_fused_moe`. So this is one reproducible bug with
a threshold between 81,920 and 98,304, not a near-limit edge case.

**Practical guidance: 81920 is verified to work at a 131072 budget, and 65536
remains the recommended serving context** because it is the setting the sixcat
run and the speed ladders were measured at. A prompt of 98,304 tokens or more
will take the engine down.

Note that prefill is chunked at `--max-num-batched-tokens`, so the MoE should
only ever see one chunk at a time and total prompt length should not reach it.
Something in that path is sized by the whole sequence rather than the chunk.
Halving the chunk to 1024 is the cheap discriminator: if the threshold moves,
the fault scales with chunk size and there is a config workaround; if 98,304
still dies, the fault tracks total sequence length and chunking is irrelevant.

## Root cause: the K-pool tail slot mapping is in pool block space

Both faults chased above have one mechanism. `_kpool_tail_seed_kernel` receives
`tail_kv_cache`, which is a **view into the shared KV pool**, one slice per tail
layer. The view is a few hundred blocks. The tail slot mapping it is indexed
with carries block ids from the **global pool**, and the kernel uses them
directly as offsets from the view's base. It masks only the head dimension;
nothing constrains the block index.

Instrumenting the kernel's own write predicate and bounding only the blocks it
actually stores to, at `max-model-len` 65536 with an 8,192-token prompt:

```text
max_written_block=34303    tail_blocks=186
tensor_bytes=380,928       storage_bytes=13,385,428,992
storage_offset_bytes=13,171,728,384 ... 13,332,003,840   (per tail layer)
escapes_allocation=False, False, False, True
```

At 2,048 bytes per tail block, block 34,303 sits about 67 MiB past the start of
the view. The consequence depends on where that layer's view sits in the pool:

| Tail layer | Result |
|---|---|
| lower offsets | the write lands **inside** the 12.5 GiB KV pool, on other layers' data. Silent corruption, no fault |
| highest offset | 13,332,003,840 + 70,254,592 = 13,402,258,432 against 13,385,428,992 bytes of storage. Overshoots by **16.8 MB** and CUDA raises an illegal access |

That single mechanism accounts for every symptom in this document:

- The fault is intermittent because only the highest-offset layer escapes.
- Long context faults sooner because more blocks are allocated, so block ids
  climb and the overshoot grows.
- DFlash faults at its first page transition because adding the auxiliary group
  changes the pool layout and the block ids handed out.
- The sixcat run still scored well because most overruns land on other tail
  regions rather than escaping, which degrades sparse-attention index quality
  subtly rather than crashing.

Counts at 65536 on an 8,192-token prompt: **132 overrunning calls against 252
clean ones**. A 2,048-token prompt stayed in bounds.

### Minimal reproducer, and the trigger is generation not prompt

The sixcat crash is deterministic and reduces to **one request**. It died at the
same point twice, on the original run and on a resume with a cold server and an
empty KV cache, which rules out accumulated state. The item is `ifeval:1300`:

```json
{"id": "ifeval:1300", "ptok": 76, "ctok": 32768, "finish": "length", "ok": false}
```

A **76-token prompt** that drives a **32,768-token generation**. The constraints
are close to unsatisfiable, no commas and no letter "c" in 250+ words, so the
model loops in thinking and runs to its full budget.

That corrects the framing the rest of this document was built on. Every context
probe here used long **prompts**. The trigger is total sequence length reached
by **decoding**, which is the far more common path in real serving and is why an
eval with thinking on found this when prefill probes only found the edges.

It also unifies the two runtimes. This is the same item that carries the
`truncated:instruct` / `trunc-in-think:instruct` / `loop-failures:instruct` flag
on the container run's 84.1667. There it generated its 32,768 tokens, the
out-of-bounds writes stayed inside the pool, and it scored as a loop failure. On
the local build the allocation layout differs, the write escapes, and the engine
dies. Same defect, two manifestations, decided by memory layout.

[`scripts/repro_kpool_tail_overrun.sh`](../scripts/repro_kpool_tail_overrun.sh)
issues exactly that request. It needs no eval harness.

A clean run of it is **not** proof a build is unaffected: whether the write
faults or corrupts silently depends on where that tail layer's view sits in the
shared pool.

### What this does and does not license

It is not established that this measurably degrades output. The 64k sixcat
result and the speed ladders were produced on this code path and look sane, so
the corruption is either landing somewhere benign or affecting index selection
too little to show. Treat the published numbers as reported, not as retracted.

Two things follow for anyone fixing it. The kernel should bounds-check its
destination block regardless, because a guard cannot be wrong. And the real fix
depends on intended semantics, which this investigation did not settle: either
the slot mapping should be rebased into the per-layer view, or the view should
span the pool the mapping addresses. Guarding alone would silently drop writes
if the mapping is meant to be global, so the semantics question has to be
answered first, upstream, by whoever owns `glm5next`.

## After the K-pool tail fix (2026-08-30)

Same harness as the runtime A/B above: 64k, MTP k=2, four workloads, one warm-up
plus three measured runs, 400 completion tokens, CUDA graphs on, prefix caching
on. The only change is the runtime fix in
[`KPOOL_TAIL_BUG.md`](KPOOL_TAIL_BUG.md).

| Workload | Before fix (same build) | After fix | Delta |
|---|---:|---:|---:|
| prose | 15.1284 tok/s | 14.8064 tok/s | -2.1% |
| structured | 20.5982 tok/s | 20.2678 tok/s | -1.6% |
| code | 16.1220 tok/s | 15.6495 tok/s | -2.9% |
| math | 17.3159 tok/s | 16.2294 tok/s | -6.3% |
| arithmetic mean | 17.2911 tok/s | 16.7383 tok/s | -3.2% |

Two readings are possible and this run does not separate them: the fix makes the
tail path do work every step that the broken path skipped (a per-step slot
mapping plus real tail writes), or this is run-to-run spread, which has been
2-3% on this box. A paired re-run on one boot would settle it. Either way the
before-fix numbers were produced by a runtime writing out of bounds, so they
were never a clean baseline.

Stability on the same boot, graphs on: 4,096- and 8,192-token generations and
a 32,000-token prompt all completed (the pre-fix build died at ~2.2k generated
tokens in the field).

## SGLang boundary

SGLang was not used as a performance rung because its [current quantization
documentation](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/quantization.mdx)
does not include EXL3. This checkpoint stores routed
experts as EXL3 trellis tensors, so trying a different launch flag cannot load
it. A fair SGLang comparison first requires porting the routed-expert loader,
the ExLlamaV3 fused-MoE call, GLM-5.3 sparse attention, and DFlash auxiliary
state handling. Until that backend exists, vLLM is the only tested server in
this recipe; this is a compatibility boundary, not a claim that SGLang would
be slower.

## 256k boot and needle ladder (2026-08-31)

See the README's "Long context: measured ceiling" section for the full table:
262,144 boots (KV 1,093,332 tokens, 93.74 GiB), needle-perfect through 163,479
prompt tokens, engine wedge between ~163k and ~180k under investigation. Raw
records: `ctx256-k2.jsonl` from `scripts/ctx_bench.py`.
