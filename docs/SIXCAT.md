# sixcat 0.5.1 evaluation report

GLM-5.3-Flash EXL3 K2 on one DGX Spark (GB10), served through vLLM with native
MTP k=2 at 65,536 context, thinking **on**.

**Headline: `overall[vendor] 84.1667`, and it is flagged. Do not quote it as a
clean overall.**

## Run identity

| Field | Value |
|---|---|
| Served model | `GLM-5.3-Flash-EXL3` |
| `max_model_len` | 65,536 |
| Speculation | native MTP k=2 |
| Thinking | on |
| Transport | `openai`, HTTP `/v1` |
| Parser | v4 |
| Code execution | host-guarded |
| Selection profile | `challenge-v1` |
| Selection fingerprint | `05c04833fcdb` |
| Policy | `vendor`, family `glm-5.x`, temperature 1.0 |
| Policy fingerprint | `f63dd8393f13` |
| Policy source | `zai-org/GLM-5.2`, reviewed 2026-08-22 |
| Result schema | `sixcat-v2` |
| Limit | 20 **per category**, 120 items total |
| Timed out | no |

**Runtime provenance.** `/v1/models` reported `root: /model`, so this run was
served by the **container runtime**, not the local source build documented in
[`MEASUREMENTS.md`](MEASUREMENTS.md), which reports
`root: /home/markus/models/GLM-5.3-Flash-EXL3-K2`. The scores below therefore
belong to the older runtime. They have not been re-run on the current local
build.

## Scores

| Category | Score | n | truncated | loop failures | Budget |
|---|---:|---:|---:|---:|---:|
| knowledge | 65.0 | 20 | 0 | 0 | 8,192 |
| math | **100.0** | 20 | 0 | 0 | 16,384 |
| truth | 85.0 | 20 | 0 | 0 | 8,192 |
| instruct | 75.0 | 20 | **1** | **1** | 32,768 |
| code | 90.0 | 20 | 0 | 0 | 32,768 |
| tools | 90.0 | 20 | 0 | 0 | 16,384 |
| **overall[vendor]** | **84.1667** | 120 | 1 | 1 | |

Flags: `truncated:instruct`, `trunc-in-think:instruct`, `loop-failures:instruct`.

All 120 items parsed at high confidence. No category recorded a low-confidence
or missing parse.

## Output length and throughput

| Category | ctok p50 | ctok p95 | ctok max | total ctok | wall s | tps_mean | suite_tps |
|---|---:|---:|---:|---:|---:|---:|---:|
| knowledge | 118 | 1,915 | 4,256 | 7,952 | 586.6 | 12.36 | 13.56 |
| math | 109 | 218 | 240 | 2,407 | 139.8 | 16.96 | 17.22 |
| truth | 25 | 109 | 122 | 840 | 67.7 | 11.50 | 12.40 |
| instruct | 564 | 3,647 | **32,768** | 45,992 | 2,676.1 | 14.88 | 17.19 |
| code | 392 | — | 2,105 | — | — | n/a | n/a |
| tools | 22 | — | 119 | — | — | n/a | n/a |

Suite totals as reported by sixcat:

```json
{"items": 80, "total_ctok": 57191.0, "total_wall_s": 3470.249486484943,
 "suite_tps": 16.480371288212318, "tps_mean": 13.924622875318306}
```

`rtok` / `atok` are null throughout: this engine did not emit
`reasoning_tokens`, so thinking-token accounting is unavailable. Per-item
prefill and decode TPS are also null; `speed_n` is 0 in every category.

## Two things not to misread

**The 16.48 tok/s suite figure covers four categories, not six.** `speed.items`
is **80 of 120**. Code and tools record no per-item throughput, and the token
total confirms it: 7,952 + 2,407 + 840 + 45,992 = 57,191 exactly, which is
knowledge, math, truth and instruct alone. Quote it as a four-category suite
rate, and never as a decode rate. The comparable decode number is the
128-token bench in the README.

**One instruct item dominates the run.** Instruct is 2,676 s of 3,470 s wall
(77%) and 45,992 of 57,191 completion tokens (80%), with `ctok_max` exactly
32,768, its full budget. Median instruct output was 564 tokens, so that single
item ran roughly 58x the category median. It is the `truncated`,
`trunc-in-think` and `loop-failures` flag all at once: the model kept thinking,
hit the budget, and produced an empty answer.

That is what the flag on 84.1667 means. Fixing or excluding that one item would
move both the instruct score and the wall-clock profile materially.

## Re-run on the local build: did not complete

The scores above were served by the container runtime, so the suite was re-run
unchanged on the local source build, same selection profile, policy fingerprint,
context and speculation. Only the runtime differed.

**It crashed.** The server came up at 04:12:37, sixcat ran for about 22 minutes,
and the engine died after **69 of 120 items** with
`HTTP 500: EngineCore encountered an issue` and, in the server log, a CUDA
illegal memory access.

| Stage | Result |
|---|---|
| knowledge, math, truth | 60 items completed |
| instruct | died roughly 9 items in, on `ifeval:*` |
| code, tools | never reached |

Launch traceback surfaces at `async_utils.get_output` on a copy-event
synchronize, which is asynchronous attribution rather than the faulting kernel.
That is the same signature the long-context investigation produced before
synchronous launches identified `_kpool_tail_seed_kernel`, and instruct is the
longest and most thinking-heavy category, so this is **consistent with** the
K-pool tail overrun documented in [`MEASUREMENTS.md`](MEASUREMENTS.md). It is
not proven to be the same fault; that would need a `CUDA_LAUNCH_BLOCKING=1`
repeat, which is impractical across a 58-minute suite.

### What this changes

The K-pool tail overrun is not a latent curiosity. It ends a real evaluation
workload at the recipe's documented serving context.

Why the container run survived all 120 items and the local build did not is
unresolved, and the mechanism suggests luck rather than correctness: most
overrunning writes land inside the shared KV pool on other layers' data and
pass silently, and only the highest-offset tail layer escapes the allocation and
faults. A different build produces a different allocation layout, so it changes
which writes escape. Neither runtime is demonstrated to be free of the bug; one
of them merely got away with it.

Treat this as a single observation. It has not been repeated.

## Reproducing

[`scripts/run_sixcat.sh`](../scripts/run_sixcat.sh). HTTP `/v1` only; do not
point sixcat at an agent or harness stdio.

```bash
python -m sixcat \
  --base-url http://127.0.0.1:8888/v1 \
  --model GLM-5.3-Flash-EXL3 \
  --policy vendor --policy-family glm-5.x \
  --thinking on --limit 20 --max-minutes 0 \
  --request-timeout 3600 --ctx 65536 --concurrency 1 \
  --transport openai --no-resume
```

`--request-timeout 3600`: `ifeval:1300` runs to its full 32,768-token budget, about 1,830 s at 16 tok/s; at the earlier 1,800 s the client timed out on it while the engine was healthy.

`--limit 20` is 20 **per category**, about 120 items, not 20 total. Serve at
65,536 so the think-on budgets fit: knowledge 8,192, math 16,384, truth 8,192,
instruct 32,768, code 32,768, tools 16,384.

Two sixcat-side edits are required on this vLLM:

1. The `glm-5.x` `stop` list has 6 strings. The vLLM OpenAI schema allows **4**.
   Trim to `<|user|> <|end|> <|eot|> <|endoftext|>` or the request 400s.
2. `preclose_think: true` **forces** `enable_thinking=false` into
   `chat_template_kwargs`. Set it false, or you score answer-mode while the
   receipt claims thinking is on.

## Raw artifacts

```text
results/hermes/glm53-flash-exl3-k2-mtp2-64k-thinkon-v051.json    431,583 B
results/hermes/glm53-flash-exl3-k2-mtp2-64k-thinkon-v051.jsonl   369,206 B
```

The `.jsonl` carries one record per item, including prompt, completion, parse
confidence and per-item timing.
