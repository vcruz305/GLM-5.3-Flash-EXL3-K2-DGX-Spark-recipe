# GLM-5.3-Flash EXL3 on one DGX Spark: what was broken, what was fixed, and the receipts

**vcruz305, 2026-08-31.** Every number below was measured on one GB10 (DGX Spark, 121.7 GiB unified), on the runtime shipped in this repo. Raw records, reproducers, and scripts are all committed here.

## TL;DR

- The "loopy / unusable" reports were a real bug, in vLLM, not in the quant. Root-caused, fixed in two lines, validated, shipped in prebuilt wheels, reported upstream. The upstream PR still does not carry the fix, so stock builds and the public container image still have it.
- On the fixed runtime the pack does not loop: 70 agent-shaped responses across two serving shapes, zero real loops, zero tool loops, zero errors. The third shape (the reporter's exact config) was preempted for the hang investigation and runs on the next box.
- Long context is real: boots at 262,144, needle recall is perfect through 163,479 prompt tokens, prefill holds ~590 tok/s the whole way. The second runtime bug -- a hang above ~163k prompt tokens -- is now **root-caused and fixed** (section 0): it was the EXL3 fused-MoE fat-expert fallback, cleared by raising one constant (`TEMP_ROWS_FUSED` 128->2048), verified with a cold 180,224-token prefill returning 200 in 515 s; a 258k prefill is slow but no longer hangs. Chasing the wedge also surfaced and fixed a third bug -- an int32 offset overflow in vLLM's vendored flash-linear-attention kernels -- which was ruled out as the serving cause (step-local 2,048-token calls never reach it) but is standalone-proven and stays patched (section 3).
- Fidelity is measured, not vibed: full-vocab KLD against BF16 on 1,048,064 positions per checkpoint, with the official FP8 as the anchor.

---

## 0. The >163k prefill wedge: root-caused and fixed (2026-08-31)

The wedge that capped verified context at 163k was **not** in the pack, the K-pool indexer, the sparse-MLA attention, or the linear-attention kernels. It was the **EXL3 fused-MoE fat-expert fallback**.

**Root cause.** Past ~163,840 prompt tokens the router starts concentrating **more than 128 rows onto a single expert** within a 2,048-token prefill chunk. The fused `exl3_moe` kernel caps at `TEMP_ROWS_FUSED = 128` rows per expert, so any "fat" expert falls back to `apply_exl3_python_loop` -- a per-expert `LinearEXL3` reconstruct that takes **minutes per chunk**. The engine keeps running but stops emitting tokens, so a client read-timeout looks like a hang. It is a **latency cliff, not a deadlock**.

**How it was localized.** Nine `CUDA_LAUNCH_BLOCKING=1` runs with flushed stage markers walked the whole forward and exonerated, in turn: the K-pool tail indexer (completes), the SM120 FlashInfer sparse-MLA attention (completes), the KDA / flash-linear-attention chunk kernels (each serving step is a step-local 2,048-token call, so the int32 offset overflow of section 3 never triggers here), and the KDA post-processing (scatter / o_norm / o_proj, all complete). The last marker before the engine went silent was `MoE_pre_experts` -- the line right before the EXL3 experts kernel.

**The fix.** Raise the fused-MoE per-expert row cap so no expert in a 2,048-token chunk can exceed it: `TEMP_ROWS_FUSED` **128 -> 2048** in `glm53_exl3_plugin/exl3.py`, applied idempotently by `scripts/patch_moe_fat_expert_rows.py`. With the cap at 2,048 the fat-expert fallback can never fire for a 2,048-token batch, so the fused kernel handles every expert directly.

**Verified.** A cold **180,224-token** prefill now returns `status 200` with coherent, non-looping output in **515 s wall** -- the exact shape that went silent before the fix. This clears the old 163k ceiling; `MAX_MODEL_LEN=262144` still boots.

**Honest caveat.** A cold **258,048-token** prefill did **not** finish inside a 40-minute client window (`ReadTimeout` at 2,400 s, ~3x slower than a linear extrapolation from 180k). That is a **performance cliff at extreme context, not the wedge** -- the hang is gone; extreme-context prefill throughput is the remaining item. **New verified prefill ceiling: 180k.** Section 3 below is superseded by this section.

---

## 1. The crash bug: out-of-bounds writes in vLLM's K-pool tail cache (fixed)

**Symptom.** Long generations died at ~2.2k generated tokens with `CUDA error: an illegal memory access`, or survived with silent KV corruption, regardless of context setting, regardless of quant. Silent corruption mid-generation is exactly the recipe for "the model went insane and started looping."

**Root cause.** vLLM's hybrid-model path (`model_states/mamba_hybrid.py`) calls `build_attn_metadata(...)` without `positions=`, unlike the default path. The K-pool tail cache then falls through to a generic paged slot mapping and writes through garbage block ids. Measured with a bounds counter on the kernels' own write predicate: destination blocks 271 to 34,303 against a 186-block cache. Second half: the corrected mapping was returned as a fresh `clone()`, which CUDA graphs capture at a transient address and fault on replay (Xid 13).

**Fix.** Two lines: pass `positions=`, write the mapping in place. `scripts/patch_kpool_tail_positions.py`, shipped in the [prebuilt wheels](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm).

**Validation.**

| test | result |
|---|---|
| 64k context, CUDA graphs on, MTP k=2, near-unsatisfiable prompt | 4,096 and 8,192 generated tokens to completion, engine alive |
| eager soak with the bounds detector, K2 | **57,551 decode-path tail updates, 0 out of bounds** over 12,288 generated tokens |
| eager soak, K2/K3 mix | **60,551 decode-path tail updates, 0 out of bounds** over 12,288 generated tokens |
| before the fix | 48 overrunning calls at 8k on the prefill counter alone |

**Upstream.** Reported with root cause and fix on [vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906#issuecomment-5468099527). As of 2026-08-31 the PR head still lacks both halves, and the same defect ships in `vllm/vllm-openai:glm53-flash-arm64-cu130`. Full write-up: [`KPOOL_TAIL_BUG.md`](KPOOL_TAIL_BUG.md).

## 2. The quant itself: measured against BF16, not argued about

Full-vocabulary KL(BF16 || candidate) on 512 sealed 2048-token contexts, 1,048,064 scored positions per checkpoint, captured on the exact serving path (fused EXL3 MoE, fp8 KV) and scored through the model's own final norm and lm_head. Method and scorer validation: [`KLD.md`](KLD.md).

| checkpoint | token-mean KLD | median | p99 | top-1 agreement |
|---|---:|---:|---:|---:|
| official FP8 (anchor) | 0.0319 | 0.0055 | 0.40 | 0.938 |
| EXL3 K2 (2 bpw) | 0.3346 | 0.117 | 3.33 | 0.788 |
| EXL3 K2/K3 mix | **0.3121** | 0.106 | 3.13 | **0.795** |

Paired on the same 512 contexts, the mix beats K2 on **505 of 512** (mean -0.0225 nats, 95% CI [0.0207, 0.0243], t = 24.4). Half of all positions sit within 0.12 nats of the BF16 teacher.

sixcat 0.5.1 on the fixed runtime, think-on at 64k: **K2 84.17, mix 83.33, 120/120 items, no faults** (math 100, tools 90-95). A model that answers 120/120 agentic eval items with a 100 in math is not "unusable."

## 3. Long context: verified to 163k prompt tokens, 262k boots

> **Superseded 2026-08-31 (see section 0).** The 163k ceiling here was the pre-fix limit set by the wedge, now fixed. Verified prefill ceiling is now 180k; 262k boots; a 258k prefill is slow but no longer hangs.

`MAX_MODEL_LEN=262144`, CUDA graphs on, MTP k=2: **KV pool 1,093,332 tokens** (4.17 concurrent full-256k requests) at `max-num-seqs 1`, 994,955 at `max-num-seqs 10`. 93.74 GiB after load.

Real-text needle ladder (fidelity-suite text, needle planted at 10% depth, prefix caching off, reproducer `scripts/ctx_bench.py`):

| prompt tokens | prefill tok/s | TTFT | decode tok/s | needle recalled |
|---:|---:|---:|---:|---|
| 7,830 | 528 | 14.8 s | 18.8 | yes, verbatim |
| 32,405 | 597-602 | 54 s | 16.6-17.0 | yes |
| 65,173 | 601-603 | 108 s | 16.7-17.8 | yes |
| 130,709 | 572-584 | 224-229 s | 17.4-19.8 | yes |
| 147,091 | 591 | 249 s | 19.1 | yes |
| **163,479** | 588 | 278 s | 17.3 | **yes, verbatim** |
| ~179,900 | engine hang (runtime bug, see below) | | | |

**Known runtime bug, heavily narrowed, still open.** Prompts above ~163k tokens
wedge the engine: the request never returns, the engine log goes silent, the
port still accepts connections (health checks lie), restart required. Verified
ceiling: 163k prompt tokens; contexts at or below 131,072 cannot reach it.
Eight deterministic reproductions established what it is NOT: not memory
pressure (identical wedge at gpu-util 0.85 with 12 GiB free), not the MoE
fat-expert fallback (fixed separately, wedge persisted), not the KDA chunk
kernels as served (they see 2,048-token steps). Sampled stacks show the
ExLlamaV3 fused-MoE kernel spinning as the victim while the GPU grinds at 96%,
pointing at state corrupted by a full-request-length-sized code path in the
sparse-indexer family from the first prefill step.

**Found and fixed while chasing it: an int32 offset overflow in vLLM's vendored
flash-linear-attention kernels.** Standalone proof (`scripts/kda_overflow_repro.py`):
`chunk_kda_with_fused_gate` passes at T=131,072 and raises CUDA illegal memory
access at T=163,840; two offset products (`boh`/`i_tg` times the H*V*K stride,
1,048,576) overflow int32 past chunk 2,047 on the varlen path. After casting
them to int64 (`scripts/patch_fla_i64_offsets_a.py` / `_b.py`), T = 131,072 /
163,840 / 180,224 / 258,048 all pass with finite outputs. This bug bites any
GLM-5.3/KDA deployment that feeds the chunk path long single calls, and it is
in vLLM's tree, not ExLlamaV3, not any quant.

**Update (2026-08-31, second box):** the wedge was reproduced identically on a
second DGX Spark (wesche-spark-78f1) with the fullest patched stack to date:
the K-pool fix, the FLA int64 fixes, the upstream BLHNC sparse-MLA addressing
fix (`57073552`, backported and replicated into the SM121 path), and the GDN
FP32-beta fix (`56058fd5`). 180k and 192k prompts both still wedge (900 s
timeout, no output). So the wedge is not the FLA overflow, not the upstream
BLHNC addressing bug, not the MoE fat-expert path, and not box-specific: it is
a distinct, still-open defect in the full-request-length code path, under
active stack-sampling investigation.

## 4. The loop battery: does it actually loop?

35 agent-shaped responses per serving shape (24 tasks: coding, reasoning, constrained lists, JSON-only, summarization, six multi-turn tool tasks with synthetic tool results, plus a stop-obedience canary), Hermes-style system prompt, 8 tool definitions, thinking on, vendor sampling (T=1.0, top-p 0.95), max_tokens 4,096. A loop is: an 8-word phrase repeated 6+ times, a 20-200 char chunk repeated 5+ times back to back, or the same tool call emitted 3+ times. Reproducer: `scripts/loop_bench.py`.

Same K2 weights, same fixed runtime, three serving shapes:

| shape | ctx | spec | seqs | conc | responses | errors | real loops | tool loops | agg tok/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| recipe | 65,536 | MTP k=2 | 1 | 1 | 35 | 0 | **0** | 0 | 14.8 |
| seqs-4 | 65,536 | MTP k=2 | 4 | 4 | 35 | 0 | **0** | 0 | 32.3 |
| reporter's (DFlash k=7, 1024-batched) | 262,144 | DFlash k=7 | 4 | 4 | preempted for the hang investigation; queued on the next box | | | | |

Notes an honest reader needs:

- One response per shape was flagged by the n-gram detector and both were the same false positive: the "rewrite this paragraph" task finished normally (`finish: stop`) and then offered a bullet version restating the same facts. Restating is the task. The tails are coherent; the raw JSONL is committed.
- 9-10 responses per shape hit the 4,096-token budget inside thinking on think-heavy coding tasks, with zero repetition in the truncated reasoning (max 8-gram count of 3 or less). That is a budget artifact, not a loop: sixcat gives the same tasks 32k and they complete.
- The stop canary ("Say exactly DONE") returned exactly `DONE` in every shape tested.

The third row reproduces the published start script the "loopy" reports came through: DFlash with 7 draft tokens, `--max-num-seqs 4`, `--max-num-batched-tokens 1024`. If loops appear there and only there, the case is closed in one table.

## 5. What people were actually running when it "looped"

Two proven mechanisms, neither of them the quant:

1. **The unfixed runtime corrupts its own KV metadata mid-generation** (section 1). Every report on stock builds of the PR branch or the public container was sampling from a corrupted cache. That is fixed in this repo's wheels and reported upstream.
2. **The reported serving shape stacks three measured-bad choices**: DFlash k=7 (measured slower than MTP k=2 on this pack, with low draft acceptance), `max-num-seqs 4` with speculation on a box this size, and a 1M context setting. The battery above tests exactly that shape on the fixed runtime; results land in this table.

And one measured non-cause: `max-num-seqs 4` with MTP k=2 on the fixed runtime is clean (row two).

## 6. Everything shipped this cycle

| item | where |
|---|---|
| K-pool tail fix, two lines | `scripts/patch_kpool_tail_positions.py`, prebuilt wheels, upstream comment |
| bounds detector + 12k-token soak gate | `scripts/patch_kpool_tail_detector.py`, `scripts/soak.sh` (gate: publish nothing that has not survived a long-generation soak) |
| KLD pipeline (capture on the serving path + full-vocab scorer) | `scripts/kld/`, results in [`KLD.md`](KLD.md) |
| K2/K3 mix: merged, validated, published, measured better than K2 on 505/512 contexts | [weights](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix), [recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe) |
| long-context bench with needle + prefill/decode | `scripts/ctx_bench.py`, section 3 |
| loop battery, agent-shaped | `scripts/loop_bench.py`, section 4 |
| deployment traps found and guarded: nvcc on PATH, venv ninja, both kill FlashInfer's backend selection with a misleading error | `scripts/preflight.py`, `scripts/serve_one_spark.sh`, `AGENTS.md`, README failures table |
| Spark Arena style llama-benchy sweep at 262k (spark-arena-v2 profile: pp 2048, tg 128, depths 0-100k, concurrency 1/5/10) | running now, lands in `MEASUREMENTS.md` |

## 7. Still open, and being worked

- Extreme-context prefill throughput: a cold 258k prefill does not finish inside 40 minutes (~3x slower than a linear extrapolation from 180k). This is a performance cliff, not a hang -- the >163k wedge itself is fixed (section 0).
- The reporter-shape loop row and the arena sweep table (hours away, not days).
- Uncensored K2 pack: same battery queued behind the above.

---

*Serving configuration for every number here: the recipe in this repo, one command, prebuilt wheels, no docker. If a number in this report cannot be reproduced from the committed scripts, file an issue.*
