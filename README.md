# GLM-5.3-Flash EXL3 K2 on one NVIDIA DGX Spark

Reproducible **vLLM** recipe for **[vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2)** on a **single NVIDIA DGX Spark / GB10 (SM121)**.

Install the prebuilt runtime, download the Hub pack, and run `vllm serve`. Start with `python scripts/preflight.py`; it takes a second and tells you what is missing. Agents should read [`AGENTS.md`](AGENTS.md) first.

> ### K-pool tail bug: fixed in the 2026-08-30 wheels
>
> GLM-5.3's sparse-MLA K-pool tail cache was written out of bounds on long
> generations (hybrid models never passed positions to the tail metadata
> builder, so its one-block mapping was skipped). Not EXL3-specific; it is in
> every GLM-5.3 build on this vLLM lineage, including the TR3 / 2x-Spark image.
> Fixed by [`scripts/patch_kpool_tail_positions.py`](scripts/patch_kpool_tail_positions.py),
> shipped in [spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm).
> Reinstall with `bash scripts/install_prebuilt.sh`. Details, reproducer and
> detector: [`docs/KPOOL_TAIL_BUG.md`](docs/KPOOL_TAIL_BUG.md).

> Independent community engineering. Not affiliated with or endorsed by Z.ai, NVIDIA, or vLLM.

| What | Where |
|---|---|
| **Runtime** | [vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) — prebuilt wheels, install in minutes |
| **Pack** | [vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) — 120 shards, 91.017 GiB |
| **This repo** | install scripts, serve flags, and [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) |
| Source | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) BF16 |
| Engine | vLLM, `--quantization exl3`, TP=1. **Stock vLLM cannot load this pack** |
| Spec | **native MTP k=2** (in the checkpoint). Do not mix with a DFlash sidecar |

Jump: **[Agent instructions](AGENTS.md)** · [Headline](#headline-what-is-verified) · [Install vLLM](#1-install-vllm) · [Download](#2-download-the-pack) · [Serve](#3-serve) · [Smoke](#4-identity-smoke) · [Speed](#speed-leaderboard-same-prompt) · [Why 8k](#why-speed-ranks-are-at-8k) · [Ctx ladder](#context-ladder) · [Sixcat](#sixcat-051) · [Full eval report](docs/SIXCAT.md) · [Pitfalls](#failures-already-paid-for)

---

> **Full evidence report** (the bug that caused the "loopy" reports, the fix, KLD, the loop battery, 256k context): [`docs/IMPROVEMENTS_AND_EVIDENCE.md`](docs/IMPROVEMENTS_AND_EVIDENCE.md)

## Headline (what is verified)

Measured **2026-08-29** on one GB10 (~121 GiB unified memory). Tool: streamed `/v1/chat/completions`, thinking **off**, 128 completion tokens, `max-num-seqs 1`. **Spec A/B (none / DFlash / MTP) is ranked at 8k** so page size does not confound accept. **Max ctx that allocated:** MTP k=2 at **131072** (KV 786,432), though a near-limit prompt faults there, so supported serving ctx is **65536**. See [Why 8k](#why-speed-ranks-are-at-8k) and [Ctx ladder](#context-ladder). The 91 GiB load is ~12 minutes per boot.

| Item | Value |
|---|---|
| Architecture | `Glm5NextForConditionalGeneration` |
| Pack | EXL3 **bits=2**, codebook **mcg**, routed experts only (288 local). Attn / shared / embed / head / vision stay native BF16 |
| Shards | **120/120**, **97,728,721,536 B (91.017 GiB)** |
| Hardware | DGX Spark GB10, SM121, TP=1 |
| Quant flag | `--quantization exl3` |
| MoE | `EXL3_FUSED_MOE=1` (log must show `fused_moe=exl3_moe`). **Do not** pass `--moe-backend marlin` |
| KV | `--kv-cache-dtype fp8` |
| Decode winner @ 8k | **MTP k=2: 15.7–16.5 tok/s**, mean accept **~2.2**, pos1 ~74–83% / pos2 ~44% |
| Same MTP @ 64k | **14.6–15.7 tok/s** (warm 14.6, TTFT 314 ms). Same winner — slightly slower pages |
| No-spec floor @ 8k | **9.6–9.8 tok/s** |
| Max `max-model-len` allocated | **131072** (MTP k=2, util 0.91, KV **786,432**, 6×). Context does **not** bound the known K-pool fault; see the warning below |
| sixcat 0.5.1 | 120/120 think-on at 64k — **overall 84.1667 is flagged**, see [`docs/SIXCAT.md`](docs/SIXCAT.md) |

A cold **258,048-token** single-request prefill is verified passing as of **2026-09-01**, with CUDA graphs on, a pinned 3 GiB KV pool, and speculative decoding off -- see [Long context: measured ceiling](#long-context-measured-ceiling-2026-09-01).

This recipe is scoped to **2-bit routed experts on one Spark**.

---

## Prove the pack is packed

`config.json` must contain:

```text
quantization_config.quant_method = exl3
quantization_config.bits = 2
quantization_config.codebook = mcg
```

Load log that means the fused path is live (repeats per layer):

```text
EXL3 MCG trellis engaged for routed experts: bits=2 experts_local=288 ... fused_moe=exl3_moe
```

If you see a per-expert `LinearEXL3` loop, `EXL3_FUSED_MOE` is off. The local plugin in this repo natively accepts the pack's 2-bit MCG tensors.

---

## Quick start

### 1. Install the runtime

> **Do not `pip install vllm`.** Stock vLLM has neither the `exl3` quantization
> method nor the `Glm5Next` architecture, and no flag turns them on. Both come
> from the runtime below. Installing stock vLLM is the single most expensive
> mistake with this recipe: it fails only after a 91 GiB download.

Check the machine first. This takes about a second and tells you exactly what is
missing:

```bash
python scripts/preflight.py
```

Then install the prebuilt runtime. Wheels, no compiler, minutes:

```bash
bash scripts/install_prebuilt.sh
```

That installs CUDA 13 PyTorch if needed, the patched vLLM and ExLlamaV3 wheels,
FlashInfer, and the routed-expert EXL3 plugin, then reruns preflight. The wheels
are `aarch64` + Python 3.12 + CUDA 13 and carry compiled CUDA extensions, so they
are not portable to another architecture or Python minor version. FlashInfer
JIT-compiles its kernels, so the CUDA 13 toolkit's `nvcc` must be on PATH when
the server starts (`/usr/local/cuda-13.0/bin`); preflight checks it and the serve
script adds it. Without it engine init fails with `No valid attention backend
found for cuda`.

<details>
<summary>Building from source instead (only if you are changing the patches)</summary>

`scripts/install_local_runtime.sh` builds the same runtime from the pinned vLLM
and ExLlamaV3 revisions, applies the SM121 NoPE sparse-MLA and DFlash
auxiliary-state fixes, and installs the plugin. Budget tens of minutes at best
and hours on a cold machine; the vLLM build alone was about 22 minutes with
`MAX_JOBS=12`.

```bash
python3 -m venv ~/venvs/glm53-exl3-local
~/venvs/glm53-exl3-local/bin/python -m pip install --upgrade pip
~/venvs/glm53-exl3-local/bin/python -m pip install   --index-url https://download.pytorch.org/whl/cu130   torch==2.13.0+cu130 torchvision==0.28.0+cu130

VENV=~/venvs/glm53-exl3-local bash scripts/install_local_runtime.sh
```

Do not set `VLLM_ATTENTION_BACKEND` globally on this branch. Backend selection
must remain per module: sparse FlashInfer MLA for the target and FlashAttention
for vision/DFlash.

</details>

Loading the checkpoint takes about 11 to 12 minutes per server boot regardless of
how the runtime was installed. That is the 91 GiB, not a hang.

### 2. Download the pack

[`scripts/download_weights.sh`](scripts/download_weights.sh):

```bash
hf download vcruz305/GLM-5.3-Flash-EXL3-K2 \
  --local-dir ~/models/GLM-5.3-Flash-EXL3-K2
```

Last path component **must** be the Hub basename. No `KEEP` / `HOLD` / `STASH`.  
`hf download` resumes via `--local-dir`. There is **no** `--resume-download` flag (it errors). Never `--force-download` a partial dest.

Expect **120** `*.safetensors` and **97,728,721,536** bytes.

Optional DFlash2 ladder checkpoint:

```bash
bash scripts/download_dflash2.sh
```

### 3. Serve

Patch the model-provided template once, activate the local runtime, and launch
exactly one speculative method:

```bash
source ~/venvs/glm53-exl3-local/bin/activate
python scripts/patch_chat_template_thinking.py \
  ~/models/GLM-5.3-Flash-EXL3-K2/chat_template.jinja

# Current default: native MTP k=2 at the speed-ranking context.
SPEC_METHOD=mtp MTP_TOKENS=2 MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.87 \
  bash scripts/serve_one_spark.sh
```

| Flag | Why |
|---|---|
| `--quantization exl3` | This pack is EXL3, not NVFP4 / GPTQ / compressed-tensors |
| `EXL3_FUSED_MOE=1` | fused `exl3_moe` decode. `0` falls back to a per-expert loop |
| **no marlin** | MoE backend stays auto. Marlin is the wrong kernel class here |
| `--kv-cache-dtype fp8` | measured KV path on GB10 |
| `--max-num-seqs 1` | keep 1 with spec. `np>1` + long draft garbles |
| MTP k=2 | native heads in this checkpoint. Capture sizes must include **3** |
| `--skip-mm-profiling` | vision stays **on**; skip only the MM profile pass |
| 8k for spec A/B | hold page size while ranking MTP vs DFlash; climb ctx after. Max allocated: **64k** |

No-spec baseline:

```bash
SPEC_METHOD=none MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.87 \
  bash scripts/serve_one_spark.sh
```

DFlash2 BF16 and draft-only online FP8 use the same target process. The
launcher defaults the draft to **`TRITON_ATTN`**, which is the only draft
backend that works: with FlashAttention the target's sparse-MLA K-pool indexer
faults at the first cache page transition and kills the engine. DFlash also
loses to MTP k=2 by 9% at matched context, so this path is for reproducing the
measurement, not for serving:

```bash
SPEC_METHOD=dflash DFLASH_TOKENS=3 \
  DFLASH_DIR=~/models/GLM-5.3-Flash-DFlash2 \
  bash scripts/serve_one_spark.sh

SPEC_METHOD=dflash DFLASH_TOKENS=3 DFLASH_QUANTIZATION=fp8 \
  DFLASH_DIR=~/models/GLM-5.3-Flash-DFlash2 \
  bash scripts/serve_one_spark.sh
```

Think-on 64k MTP: `SPEC_METHOD=mtp MTP_TOKENS=2 MAX_MODEL_LEN=65536 GPU_MEM_UTIL=0.91 bash scripts/serve_one_spark.sh`

**Never** pass MTP and a DFlash draft on the same server.

Weight reload is ~12 minutes. Treat a flag flip as a new boot.

### 4. Identity smoke

```bash
curl -s http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/v1/models
# id must be GLM-5.3-Flash-EXL3, max_model_len matches the flag

curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
# content: pong, completion_tokens: 2
```

TPS:

```bash
python scripts/bench_v1.py --base-url http://127.0.0.1:8888/v1
# run twice; quote the warm row
```

Spec metrics (engine log): `SpecDecoding metrics: Mean acceptance length: … Per-position acceptance rate: …`

---

## Historical 128-token speed leaderboard (older runtime)

Thinking off, 128 gen, 8k, fused MoE on, seqs=1, KV fp8.

| Config | Decode tok/s | Accept |
|---|---:|---|
| no spec (batched 1024) | 9.77 | — |
| no spec (batched 2048 + FLASH_ATTN) | 9.63 | — |
| DFlash sidecar k=7 | 11.5 | mean ~1.8 / 7, 11–18% draft |
| DFlash sidecar k=3 | 12.8 | mean ~1.8 / 3, ~27% draft |
| MTP k=1 | 14.8 | 76–80% |
| **MTP k=2, batched 2048** | **15.7–16.5** | **~74/44%, mean ~2.2** |
| MTP k=2, batched 4096 | 15.6 | wash |

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, 5-layer Qwen3, ~2.18 GiB) was remeasured on the current runtime rather than carrying the old loader conclusion forward. It now runs correctly, but only with `TRITON_ATTN` draft attention; the FlashAttention draft path faults at its first cache page transition. At matched 8k context on the four-workload ladder, **DFlash k=3 reaches 15.61 tok/s against MTP k=2's 17.02**, losing prose, code and math and winning only structured. It does not displace native MTP. See [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md).

vLLM warning: `num_speculative_tokens > 1` reruns the **same** MTP layer. k=2 still beat k=1 on tok/s. k=3 was not worth another 12-minute reload.

---

## Why speed ranks are at 8k

The 8k table is **spec method A/B** (none vs DFlash k vs MTP k), not “this is the longest context we can run.”

1. **Accept vs overhead.** DFlash vs MTP is a rejection-rate question. Holding `--max-model-len 8192` keeps attention page size in the same band (~6912 tokens) so a 2 tok/s delta is the speculator, not a different MLA page.
2. **32k DFlash changed the page.** At 32k the engine set attention block **7168**. That is a different decode shape. Ranking DFlash k=3 vs MTP k=2 on that boot would mix page size into the result, so 32k was allocation-only.
3. **Reload cost.** 91 GiB is ~12 minutes per `max-model-len` flip. Spec k A/B already burned several boots at 8k. Ctx climb was a second axis: allocate KV, smoke `/v1`, do not re-rank k.
4. **Leftover UMA is tens of GiB, not a million tokens.** After 91 GiB weights, 64k MTP still fits (786k KV tokens). We did not chase a huge ctx number for the tok/s table.

**Serve/eval ctx is 64k** (`GPU_MEM_UTIL=0.91`) so sixcat think-on budgets fit. That boot’s short-prompt decode is in the ladder below — it did **not** dethrone MTP k=2.

---

## Long context: measured ceiling (2026-09-01)

`MAX_MODEL_LEN=262144` boots and serves with CUDA graphs and MTP k=2:
**KV pool 1,093,332 tokens** (4.17 concurrent full-256k requests), 93.74 GiB
after load. Real-text needle ladder on that server, prefix caching off, needle
planted at 10% depth and recalled **verbatim at every passing point**:

| prompt tokens | prefill tok/s | TTFT | decode tok/s | needle |
|---:|---:|---:|---:|---|
| 7,830 | 528 | 14.8 s | 18.8 | found |
| 32,405 | 597–602 | 54 s | 16.6–17.0 | found |
| 65,173 | 601–603 | 108 s | 16.7–17.8 | found |
| 130,709 | 572–584 | 224–229 s | 17.4–19.8 | found |
| 147,091 | 591 | 249 s | 19.1 | found |
| **163,479** | 588 | 278 s | 17.3 | found |
| **180,224** | ~350 | 515 s | — | **found — wedge fixed 2026-08-31** |
| **258,048** | ~604 | 427 s | not measured | not run -- summary task, returned 200; passes with `expandable_segments` (2026-09-01) |

**Two independent causes, not one.** The ~163k-180k hang above (fixed
2026-08-31) was the EXL3 fused-MoE fat-expert fallback: past ~163,840 tokens
the router puts >128 rows on one expert per 2,048-token chunk, tripping a slow
per-expert `LinearEXL3` reconstruct. Raising `TEMP_ROWS_FUSED` to 2048 via
`scripts/patch_moe_fat_expert_rows.py` clears it, and that fix stands --
180,224 passes because of it. A **second, independent cause** remained and
wedged every prefill past roughly 200k-230k tokens even with the row-cap fix
in place.

**Root cause of the second wedge: an allocator ratchet on unified memory.**
vLLM's sparse-indexer chunked prefill
(`vllm/v1/attention/backends/mla/indexer.py`, `split_indexer_prefill_chunks`)
allocates an fp32 logits buffer of shape `(sub_m, N_compressed)` per
sub-chunk, sized up to `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512 MB in
this fork). Because the prefix grows every 2,048-token step, each step's
buffers are slightly larger than the last, so the PyTorch caching allocator
never reuses freed blocks and keeps requesting new segments. On GB10 unified
memory `cudaMalloc` never fails until the kernel itself is starved, so the
allocator never flushes its cache (`num_alloc_retries` stays 0) and reserved
memory ratchets up with every prefill step until host memory is exhausted
(32 GiB reserved by 262k in the no-model replay below) -- a
page-lock livelock (kernel stacks in `folio_wait_bit_common`), engine silent,
`/health` still returning 200.

**No-model reproducer** (`scripts/ratchet_replay.py`, 2 seconds, no weights)
replays the same allocation pattern at L=262144, MNBT=2048, compression
ratio 4:

| config | peak reserved | segments | `num_alloc_retries` |
|---|---:|---:|---:|
| default allocator | 32.26 GiB | 128 | 0 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1.49 GiB | 0 | -- |
| default allocator + `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | 1.04 GiB | 27 | -- |

`--max-num-batched-tokens` is **not** a lever here: smaller chunks mean more
allocation events, same ratchet.

**Live verification (2026-09-01).** Serve config:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, KV pool pinned to 3 GiB
(`--kv-cache-memory-bytes 3221225472`, 349,525 fp8 KV tokens -- enough for one
262,144-token request), `MAX_MODEL_LEN=262144`, `--max-num-batched-tokens
2048`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`, prefix caching off, CUDA
graphs on (default vLLM graph mode, no `--enforce-eager`), speculative
decoding off, plus an opt-in venv-side indexer workspace right-sizing patch
(`GLM53_INDEXER_WORKSPACE=rightsize`) that is **not shipped in this repo** --
it was on during this run and has not been separated out. A cold
258,048-token prefill returned HTTP 200 in **427.3 s wall**, about 604
prompt tokens/s end to end (the table's prefill column is prompt tokens over
wall, as for every other row). `MemAvailable` was
15.0 GiB at serve-up, 13.5 GiB when prefill started, then flat between 13.82
and 13.85 GiB for the entire prefill -- about 1.2 GiB total growth, zero
drift over 7 minutes.

The control run (identical config, no `expandable_segments`) drained
`MemAvailable` at an accelerating 3.0-4.5 GiB/min and was aborted by a
watchdog at a 100 MB floor before finishing. An earlier control with the
row-cap fix and the default allocator (utilization-derived 8.39 GiB KV pool)
passed 180,224 in 309.8 s, then hit the OOM floor (22 MB available) at
229,376 and was aborted. All three runs had swappiness 10 and a 16 GB swap
file present.

**Fix:** `scripts/serve_one_spark.sh` now defaults
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Verified prefill
ceiling is now 258,048 tokens** on one Spark, for a single request, with the
pinned 3 GiB KV pool and speculative decoding off. Reproducer:
`scripts/ctx_bench.py`; ratchet reproducer: `scripts/ratchet_replay.py`.

**MTP k=2 on the fixed config (2026-09-01, 9:23-9:31 PM PDT).** Same
configuration with `SPEC_METHOD=mtp MTP_TOKENS=2` and the pool pinned to
3758096384 bytes (3.5 GiB, 332,475 fp8 tokens with the draft layer's KV
included): a cold 258,048-token prefill plus 512 completion tokens returned
HTTP 200 in **463.6 s wall**; `MemAvailable` was 12.2 GiB at serve-up and
flat at 10.59 GiB for the entire request (minimum 10.43 GiB). The wedge fix
holds with speculation on. vLLM's 10-second logger windows during the decode
phase read 19.0 and 20.4 tok/s with mean acceptance length 2.31 in the first
window rising to 3.00 (every draft accepted) in the last two. Acceptance that
climbs to 100% on a summarize task is the signature of a repetitive tail, and
the client kept only the first 60 characters of the completion, so **no decode
tok/s figure is claimed at 258k yet**; the next run keeps the full text.

**Not yet measured on the fixed config:** decode tok/s at 258k on captured
text, needle recall at 258k (the verification prompt was a summarize task;
the model returned a coherent summary opening, not a needle probe), and
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` in a live serve (replay-only so far).
CUDA graphs are confirmed working at this config (the verification run
captured graphs and booted with the default, non-eager graph mode). Memory
headroom during the 258k prefill was ~13.8 GiB without MTP and ~10.6 GiB
with it.

MiaAI's TP=2 two-Spark recipe has not reported this at 256k on two boxes,
but their PR #70 documents the same livelock at
~236k on a 4x TP=4 setup and mitigates with `vm.swappiness=0` and a swap
cycle between serves. That is their mitigation, not something this recipe
verified as a root fix.

## Context ladder

Same box, fused `exl3_moe`, seqs=1, KV fp8. **Decode tok/s** = `bench_v1.py` thinking off, 128 gen, warm row when present. Empty decode = we allocated KV and `/v1` pong’d, no 128-token bench on that boot.

| max_model_len | spec | util | GPU KV tokens | conc. | `/v1` | Decode tok/s | Notes |
|---:|---|---:|---:|---:|---|---:|---|
| 8192 | none | 0.87 | 192,139 | 23.45× | pong | **9.77** | first boot, batched 1024 |
| 8192 | none | 0.87 | 207,778 | 25.36× | pong | **9.63** | fair floor, batched 2048 + FLASH_ATTN |
| 8192 | DFlash k=7 | 0.87 | **15,281** | 1.87× | pong | **11.5** | draft KV collapse |
| 8192 | DFlash k=3 | 0.87 | (same class) | — | pong | **12.8** | best DFlash; still loses to MTP |
| 16384 | DFlash k=7 | 0.91 | 45,095 | 2.75× | pong | *not benched* | 16k would not allocate at 15k KV / util 0.87 |
| 32768 | DFlash k=7 | 0.91 | 90,035 | 2.75× | allocated | *not benched* | attention block **7168** — skip for spec ranking |
| 8192 | MTP k=1 | 0.87 | — | — | pong | **14.8** | 76–80% accept |
| 8192 | MTP k=2 | 0.87 | 104,857 | 12.80× | pong | **15.7–16.5** | **ranking winner** |
| 8192 | MTP k=2 | 0.87 | 99,942 | 12.20× | pong | 15.6 | batched 4096 wash |
| **65536** | MTP k=2 | 0.91 | **786,432** | **12.00×** | pong | **14.6–15.7** | sixcat think-on; warm 14.6 / TTFT 314 ms; accept ~71/35% on the warm window |
| **131072** | MTP k=2 | 0.91 | **786,432** | **6.00×** | pong | *not benched* | allocates; **81,920-token prompt passes** (114.8 s); **98,304 faults** in the sparse-MLA K-pool tail and kills the engine |

Highest ctx **tried and allocated:** **131072**. Longest prompt that completes: **81,920**; **98,304 faults** in the sparse-MLA K-pool tail. **Context is not what protects you** (see the warning at the top): the same fault hits at ~2.2k *generated* tokens regardless of `max-model-len`. See [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md). DFlash at 8k cannot climb until util 0.91 because draft KV eats the pool.

Do not quote sixcat suite TPS (16.5 wall across mixed think-on items) as a 64k decode-only number. The 64k decode row above is the same 128-token bench as 8k.

---

## KLD against the BF16 teacher

Full-vocabulary KL(BF16 || K2) on the fidelity suite's 512 sealed 2048-token
contexts, 1,048,064 scored positions, scored through the model's own final norm
and lm_head from hidden states captured on the serving path (fused EXL3 MoE,
fp8 KV).

| checkpoint | token-mean KLD | 95% CI | median | p99 | top-1 agreement |
|---|---:|---|---:|---:|---:|
| official FP8 (anchor, 24 contexts) | 0.0319 | [0.023, 0.041] | 0.0055 | 0.40 | 0.938 |
| **EXL3 K2** | **0.3346** | [0.320, 0.349] | 0.117 | 3.33 | **0.788** |
| EXL3 K2/K3 mix (6 layers K3) | 0.3121 | [0.299, 0.325] | 0.106 | 3.13 | 0.795 |

Half of all positions are within 0.12 nats of BF16; the mean is carried by a
~1% tail (p99.9 6.47), which is the same tail that produces the rare long-task
derailments in sixcat. The mix is lower on 505 of 512 contexts (paired: -0.0225 nats, CI [0.0207, 0.0243]). Method, scorer validation and the paired analysis:
[`docs/KLD.md`](docs/KLD.md).

## sixcat 0.5.1

[`scripts/run_sixcat.sh`](scripts/run_sixcat.sh). HTTP `/v1` only. Do not point sixcat at an agent/harness stdio.

```bash
python -m sixcat \
  --base-url http://127.0.0.1:8888/v1 \
  --model GLM-5.3-Flash-EXL3 \
  --policy vendor --policy-family glm-5.x \
  --thinking on --limit 20 --max-minutes 0 \
  --request-timeout 1800 --ctx 65536 --concurrency 1 \
  --transport openai --no-resume
```

`--limit 20` is **20 per category** (~120), not 20 total. Serve at **64k** so think-on budgets fit (knowledge 8192 / math 16384 / instruct+code 32768).

Two sixcat-side edits required on this vLLM:

1. glm-5.x `stop` list is 6 strings. vLLM OpenAI schema allows **4**. Trim to `<|user|> <|end|> <|eot|> <|endoftext|>`.
2. glm-5.x `preclose_think: true` **forces** `enable_thinking=false` in `chat_template_kwargs`. Set it **false** or you are scoring answer-mode while the receipt says thinking on.

### Measured receipt (2026-08-29) — flagged overall

Parser v4, host-guarded HumanEval, selection `challenge-v1` / `05c04833fcdb`, fingerprint `f63dd8393f13`, 120/120 parsed at high confidence, not timed out. Full report: **[`docs/SIXCAT.md`](docs/SIXCAT.md)**.

| Category | Score | n | trunc | loop | ctok p50 | ctok max | suite_tps |
|---|---:|---:|---:|---:|---:|---:|---:|
| knowledge | 65.0 | 20 | 0 | 0 | 118 | 4,256 | 13.56 |
| math | **100.0** | 20 | 0 | 0 | 109 | 240 | 17.22 |
| truth | 85.0 | 20 | 0 | 0 | 25 | 122 | 12.40 |
| instruct | 75.0 | 20 | **1** | **1** | 564 | **32,768** | 17.19 |
| code | 90.0 | 20 | 0 | 0 | 392 | 2,105 | n/a |
| tools | 90.0 | 20 | 0 | 0 | 22 | 119 | n/a |

**overall[vendor] 84.1667** flags: `truncated:instruct`, `trunc-in-think:instruct`, `loop-failures:instruct`. One instruct item hit its full 32,768-token budget and returned an empty answer. **Do not quote 84.2 as a clean overall.**

Wall: 57,191 ctok / 3,470.2 s → suite **16.48 tok/s**, `tps_mean` 13.92. Two caveats:

- That suite rate covers **80 of 120 items**. Code and tools record no per-item throughput, and the token total is knowledge + math + truth + instruct exactly. It is a four-category suite rate, not a decode rate.
- **Instruct is 77% of the wall clock** (2,676 s of 3,470 s) and 80% of the tokens, from the single 32,768-token item. Median instruct output was 564 tokens.

Prefill/decode TPS n/a (`speed_n` 0). `rtok`/`atok` n/a (engine omitted `reasoning_tokens`).

That receipt was served by the container runtime. **Re-run on the fixed local build (2026-08-30): overall 84.17 again**, knowledge 70 / math 100 / truth 80 / instruct 70 / code 90 / tools 95, 120/120, no faults, `--request-timeout 3600`. Per-category differences are one item each; see [`docs/SIXCAT.md`](docs/SIXCAT.md).

---

## Failures already paid for

| Symptom | Cause | Fix |
|---|---|---|
| EXL3 is absent from vLLM's registry | stock vLLM was installed, or the plugin did not load | `python scripts/preflight.py`, then `bash scripts/install_prebuilt.sh`. Stock vLLM can never work here |
| ~11 tok/s with DFlash, accept ~1.8 | DFlash2 vs K2 logits | native MTP k=2 |
| 9.8 tok/s “must be a missing kernel” | that **is** the no-spec floor | fused MoE already on |
| DFlash KV 15k @ 8k | draft KV slot-share | util 0.91 before climbing ctx |
| HTTP 400 `stop` list too long | vLLM max 4 stops | trim glm-5.x extra.stop |
| 2-token sixcat rows, `enable_thinking` in the journal | `preclose_think` | set false |
| 12 minutes per A/B | 91 GiB reload | expect it; don’t chain silent retunes |
| `hf --resume-download` | flag does not exist | `--local-dir` only |
| marlin MoE | wrong backend on this EXL3 pack | omit |
| `No valid attention backend found for cuda ... FLASHINFER_MLA_SPARSE_SM120` at engine init | `nvcc` is not on PATH, so vLLM's `has_flashinfer()` rejects the only sparse-MLA backend for GB10 | `export PATH=/usr/local/cuda-13.0/bin:$PATH`; `serve_one_spark.sh` adds it and `preflight.py` checks it |

---

## Hardware (measured box)

```text
GPU:     NVIDIA GB10
Arch:    SM121
Memory:  ~121 GiB unified
Engine:  vLLM --quantization exl3, EXL3_FUSED_MOE=1
```

Single Spark. Occupancy is compute-app **plus** VRAM, not 0% util.

---

## Reproducibility boundary

This repo documents **one measured configuration**. It does not claim:

- 16 tok/s is the GB10 ceiling for every vLLM build
- 64k decode (14.6 tok/s warm) is a different spec winner than 8k — it is not
- DFlash 16k/32k tok/s (those boots were `/v1` pong + KV only)
- that 128k is safe end to end (it allocates and serves an 81,920-token prompt; 98,304 faults)
- DFlash2 will ever match MTP on this 2-bit pack
- 84.2 is an untruncated sixcat overall
- pip `vllm` without EXL3 / Glm5Next / SM121 is sufficient

---

## Related repositories

Three pieces, and you need all three:

| Repo | Role |
|---|---|
| [**vllm-exl3**](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin's canonical home: source, releases, issues. The wheel below is its distribution mirror |
| [**spark-vllm**](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) | prebuilt vLLM + ExLlamaV3 + EXL3 plugin wheels for GB10. This is one runtime, not two: vLLM is the engine, ExLlamaV3 supplies the CUDA kernels that decode the 2-bit trellis weights |
| [**GLM-5.3-Flash-EXL3-K2**](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) | the weights |
| **this repo** | preflight, install, serve, bench, and the measurement log |
| [**GLM-5.3-Flash-EXL3-K2K3-mix**](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix) + [its recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe) | the K2 base with six layers at K3; measured against K2 on the same runtime |

## License / attribution

MIT for the scripts and notes in this repo. Weights are **not** redistributed — pull them from Hugging Face and respect the GLM-5.3-Flash license. vLLM, ExLlama EXL3, FlashInfer, and sixcat-eval have their own licenses.
