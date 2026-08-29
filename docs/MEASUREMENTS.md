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

Two limits keep it out of the leaderboard as a ranked row. First, this ladder
ran at `max-model-len` 8192 while every MTP ladder above ran at 65536, so the
15.61 mean and the 17.29 MTP k=2 mean sit on different attention page sizes.
That is the same confound the README's ranking methodology exists to avoid, and
it runs in both directions here. Second, acceptance tracks workload
predictability far more sharply than MTP does: 79.27% on structured against
26.31% on prose, a spread of 53 points, where MTP k=4 spanned 91.3% to 31.7%.
A same-context MTP k=2 ladder at 8192 is the remaining measurement.

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
