# TP1 EXL3 policy A/B qualification

This protocol qualifies unreleased `vllm-exl3` changes on **one DGX Spark / TP1** without mixing unrelated variables. It is intended for the K2 pack first and then the K2/K3-mix pack using the exact same procedure.

## Candidate source boundary

Install the normal runtime first, then the development candidate:

```bash
bash scripts/install_prebuilt.sh
bash scripts/install_candidate_plugin.sh
python scripts/preflight.py
```

`install_candidate_plugin.sh` is pinned to an exact `vllm-exl3` Git SHA. Do not replace that SHA with `main` for a benchmark receipt. If testing another SHA, set `VLLM_EXL3_REF=<40-char-sha>` and record it with the results.

The current candidate exposes a **bounded K2/K3 grouped-prefill planner**, but no grouped CUDA executor yet. `runtime_diagnostics()` must therefore report:

```text
grouped_prefill.execution_available = false
```

Setting `VLLM_EXL3_GROUPED_PREFILL=1` at this stage exercises only planning/diagnostics and must **not** be reported as a throughput optimization.

## Rules for comparable measurements

Keep these fixed inside one A/B set:

- exact model directory/checkpoint revision;
- `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, KV-cache policy, prefix-caching policy, and attention backend;
- speculative method and depth;
- request prompt, generation length, sampling parameters, and concurrency;
- runtime/vLLM fork and CUDA/PyTorch builds;
- GPU clock policy and thermal state where practical.

Change **one EXL3 variable per serve**. Reboot/restart the engine between policies so CUDA graphs and allocator state do not leak across variants.

## Phase 1: K2 decode-row dispatch

Use the K2 checkpoint and keep the recipe's native MTP `k=2` baseline fixed. Test candidate defaults first, then only the K2 native row ceiling:

| Label | Environment change |
|---|---|
| `candidate-default` | none |
| `k2-cap-1` | `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2=1` |
| `k2-cap-2` | `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2=2` |
| `k2-cap-4` | `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2=4` |
| `k2-cap-8` | `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2=8` |

Leave K3 and K4 caps unset during this sweep. The candidate diagnostics must show the intended effective cap before benchmarking.

For each serve:

```bash
python scripts/bench_tp1_policy.py \
  --runs 7 \
  --warmup 2 \
  --max-tokens 512 \
  --label candidate-default
```

Use seven measured runs as the default because single-request token throughput has enough run-to-run variation that one or two samples are not promotion evidence.

Repeat the winner with `SPEC_METHOD=none` to separate target-kernel speed from MTP acceptance effects. Do **not** choose a row cap solely because acceptance rises; optimize end-to-end emitted tokens/second and TTFT while preserving correctness.

## Phase 2: mixed K2/K3 checkpoint

Repeat the winning K2 configuration against `GLM-5.3-Flash-EXL3-K2K3-mix`, then sweep K3 separately:

```text
VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3=1
VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3=2
VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K3=4
```

Do not assume the K2 row ceiling transfers to K3. Trellis bandwidth and launch economics differ by bit width.

## Phase 3: prefill baseline for the future grouped kernel

Grouped execution is not implemented in the current candidate, so this phase establishes the baseline that a later K2/K3 kernel must beat.

Use deterministic prompt files at representative lengths, for example approximately 8K, 32K, 64K, and the largest context that fits comfortably without host-memory pressure. Run each from a **cold prefix state** when measuring prefill/TTFT; cached-prefix requests are a separate measurement.

Example:

```bash
python scripts/bench_tp1_policy.py \
  --prompt-file prompts/prefill-32k.txt \
  --max-tokens 128 \
  --runs 5 \
  --warmup 1 \
  --label prefill-32k-baseline
```

The future grouped executor must be compared against the same checkpoint and serve configuration, and its receipt must include the active grouped row-window cap and effective scratch policy.

## Promotion gates for a grouped K2/K3 executor

A future grouped CUDA path should remain default-off until all of the following pass:

1. packed K2 and K3 expert numerical parity against an independent/reference execution path, including real checkpoint experts;
2. repeated-route, invalid-route, clipping, and distinct gate/up scaling cases;
3. CUDA graph capture and replay with changed inputs/routing;
4. bounded scratch behavior with no allocation growth during graph capture;
5. no host synchronization introduced into the hot grouped dispatch;
6. end-to-end cold-prefill/TTFT improvement on TP1, not only an isolated kernel microbenchmark;
7. peak unified-memory and KV-capacity receipts showing that the speedup does not consume unacceptable context headroom;
8. K2/K3-mix validation in addition to pure K2.

## Attribution requirement for the future executor

The high-level grouped-expert optimization direction was informed by MiaAI-Lab's GLM-5.3 E3 work. The local planner is independently written and K2/K3-specific. If the future CUDA executor directly derives any historical MIT implementation, its commit must identify the exact upstream snapshot/files and enumerate local changes. If it is independently implemented, the commit should still cite the prior art as design influence and say explicitly that no source was copied.

Do not describe an implementation as independent merely because identifiers/comments were renamed. Provenance describes the actual development relationship.
