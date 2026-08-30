# GLM-5.3-Flash EXL3 K2 on one NVIDIA DGX Spark

Reproducible **vLLM** recipe for **[vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2)** on a **single NVIDIA DGX Spark / GB10 (SM121)**.

Install the prebuilt runtime, download the Hub pack, and run `vllm serve`. Start with `python scripts/preflight.py`; it takes a second and tells you what is missing. Agents should read [`AGENTS.md`](AGENTS.md) first.

> Independent community engineering. Not affiliated with or endorsed by Z.ai, NVIDIA, or vLLM.

| What | Where |
|---|---|
| **Runtime** | [vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) — prebuilt wheels, install in minutes |
| **Pack** | [vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) — 120 shards, 91.017 GiB |
| **This repo** | install scripts, serve flags, and [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) |
| Source | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) BF16 |
| Engine | vLLM, `--quantization exl3`, TP=1. **Stock vLLM cannot load this pack** |
| Spec | **native MTP k=2** (in the checkpoint). Do not mix with a DFlash sidecar |

Jump: **[Agent instructions](AGENTS.md)** · [Headline](#headline-what-is-verified) · [Install vLLM](#1-install-vllm) · [Download](#2-download-the-pack) · [Serve](#3-serve) · [Smoke](#4-identity-smoke) · [Speed](#speed-leaderboard-same-prompt) · [Why 8k](#why-speed-ranks-are-at-8k) · [Ctx ladder](#context-ladder) · [Sixcat](#sixcat-051) · [Pitfalls](#failures-already-paid-for)

---

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
| Max `max-model-len` allocated | **131072** (MTP k=2, util 0.91, KV **786,432**, 6×). Longest prompt verified: **81,920**. **98,304 faults.** Recommended serving ctx stays **65536** |
| sixcat 0.5.1 | 120/120 think-on at 64k — **overall 84.2 is flagged**, see below |

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
are not portable to another architecture or Python minor version.

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

DFlash2 BF16 and draft-only online FP8 use the same target process; the
launcher selects FlashAttention for the non-causal draft and aligns its cache
pages with the target's sparse-MLA allocation:

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
| **131072** | MTP k=2 | 0.91 | **786,432** | **6.00×** | pong | *not benched* | allocates; **81,920-token prompt passes** (114.8 s); **98,304 faults** in the EXL3 fused-MoE path and kills the engine |

Highest ctx **tried and allocated:** **131072**. Longest prompt that completes: **81,920**; **98,304 faults** in the EXL3 fused-MoE path, so **65536 remains the recommended serving ctx**. See [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md). DFlash at 8k cannot climb until util 0.91 because draft KV eats the pool.

Do not quote sixcat suite TPS (16.5 wall across mixed think-on items) as a 64k decode-only number. The 64k decode row above is the same 128-token bench as 8k.

---

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

Parser v4, host-guarded HumanEval, selection `challenge-v1` / `05c04833fcdb`, fingerprint `f63dd8393f13`, 120/120, not timed out.

| Category | Score | n | trunc | loop |
|---|---:|---:|---:|---:|
| knowledge | 65.0 | 20 | 0 | 0 |
| math | **100.0** | 20 | 0 | 0 |
| truth | 85.0 | 20 | 0 | 0 |
| instruct | 75.0 | 20 | **1** | **1** |
| code | 90.0 | 20 | 0 | 0 |
| tools | 90.0 | 20 | 0 | 0 |

**overall[vendor] 84.2** flags: `truncated:instruct`, `trunc-in-think:instruct`, `loop-failures:instruct` (`ifeval:1300` hit 32768 tokens, empty answer). **Do not quote 84.2 as a clean overall.**

Wall: 57 191 ctok / 3470 s → suite **16.5 tok/s**. Prefill/decode TPS n/a. `rtok`/`atok` n/a (engine omitted `reasoning_tokens`).

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
| [**spark-vllm**](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) | prebuilt vLLM + ExLlamaV3 + EXL3 plugin wheels for GB10. This is one runtime, not two: vLLM is the engine, ExLlamaV3 supplies the CUDA kernels that decode the 2-bit trellis weights |
| [**GLM-5.3-Flash-EXL3-K2**](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) | the weights |
| **this repo** | preflight, install, serve, bench, and the measurement log |

## License / attribution

MIT for the scripts and notes in this repo. Weights are **not** redistributed — pull them from Hugging Face and respect the GLM-5.3-Flash license. vLLM, ExLlama EXL3, FlashInfer, and sixcat-eval have their own licenses.
