# Agent instructions: GLM EXL3 on one GB10

Use this guide and [README.md](README.md) for current instructions. The archived README and dated measurement reports are historical evidence, not commands to execute indiscriminately.

## Scope and source of truth

This recipe targets `vcruz305/GLM-5.3-Flash-EXL3-K2` and `vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix` on one NVIDIA GB10 / SM121, TP=1. Set `MODEL_DIR` explicitly and record the selected checkpoint revision, config, index and template hashes. Reuse existing shards; do not assume both quants have identical sizes.

The active plugin lives in [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3). `runtime/exl3_plugin/` is historical provenance. Do not patch that copy and mistake it for the installed candidate.

The current executable candidate is `d3cfd394920360d69f820d2dc96f8292a9e10283` (development metadata `0.4.2`). The pre-SUH-cache control is `28041c423a81fe033e7128d8888e3762fb914910`. Both have the per-bit policy; their comparison isolates the cache change, not the entire development line. Preserve the original working installed wheel/revision as a separate baseline.

## Preserve the working system

Verify the authorized machine, active processes, CUDA capability, disk and unified-memory headroom before changing anything. Do not rent, restart, destroy or resize an instance, alter system drivers or swap policy, delete checkpoints, or stop unrelated processes as part of a plugin qualification. Restart only the experiment's serving process when needed.

Keep a rollback wheel/source revision and dependency manifest. Never overwrite local uncommitted work or publish secrets, hostnames, credentials or raw environment dumps. Use an allowlist for environment capture and sanitize logs before publication.

## Runtime requirements and order

The prebuilt path expects aarch64, GB10 capability 12.1, Python 3.12 and PyTorch 2.13.0+cu130. CUDA 13 `nvcc`, the active venv's `bin/`, `ninja`, and compiled ExLlamaV3 must resolve. **Do not install stock vLLM** to fix a missing EXL3/Glm5Next integration.

For an existing runtime, activate it and run `python scripts/preflight.py` first. Diagnose a nonzero result before proceeding. For a fresh environment only, use `scripts/install_prebuilt.sh`, activate the resulting venv, and rerun preflight before downloading weights. Its floating plugin dependency is not a reproducible A/B baseline: record resolved artifacts and verify imports.

Run `scripts/install_candidate_plugin.sh` only after preserving the baseline and checking native build prerequisites. Rebuild the plugin alone when possible; do not rebuild the entire vLLM/ExLlamaV3 stack unnecessarily. Verify the actual imported Python and native-extension paths, ABI and hashes. The candidate-ref marker alone can be stale and does not identify a live worker.

## Qualification rules

Follow [docs/TP1_POLICY_AB.md](docs/TP1_POLICY_AB.md). The schema-1 benchmark/comparison helpers have known timing and provenance gaps listed there; correct/validate the measurement path before using results for promotion.

Start at 8K, TP=1, FP8 KV, native MTP k=2, `MAX_NUM_SEQS=1`, and `MAX_NUM_BATCHED_TOKENS=2048`. Use `HOST=127.0.0.1`. Keep `EXL3_FUSED_MOE=1`, no Marlin override, and no DFlash sidecar. Run a matched no-spec control separately. Disable prefix caching for cold-prefix TTFT; cached-prefix tests are separate. Keep allocator, template, sampling and memory policy identical within each A/B pair.

Do not confuse configured backend/caps with executed kernels. Capture actual native/fallback dispatch and MTP verification row shapes. Never force unsupported calls past contract checks.

`VLLM_EXL3_GROUPED_PREFILL=1` enables planning/diagnostics only: there is no grouped CUDA executor. `VLLM_EXL3_FUSED_TEMP_ROWS` is a request only: the allocation override is inactive and actual capacity remains 2048. Neither is a measured optimization. The existing tiled fat kernel is K4-gated; its microbenchmarks are not K2/K3 grouped-prefill results.

The SUH cache assumes loaded rotations remain immutable. Verify normal load-time and direct-call lazy-cache paths; rebuild fused state after any rotation/weight changes rather than reusing stale metadata.

## Context and historical caveats

Earlier 98,304-token faults are historical K-pool bugs, not the current universal limit. The 258,048-token historical result required a particular configuration and an unshipped indexer workspace patch; do not claim this checkout alone reproduces it. Start with 8K, then 32K/64K as headroom allows. Keep `TEMP_ROWS_FUSED=2048` and the launcher's allocator policy. Monitor host `MemAvailable`, swap and request progress; a responsive `/health` is not proof that generation is progressing.

The K-pool bounds detector requires eager mode. Run detector/eager checks separately from graph-mode speed measurements and retain the exact settings. Respect existing FlashInfer/runtime pins; a known metadata conflict is not permission to upgrade dependencies during A/B testing.

## Reporting and promotion

Retain exact source/artifact identity, requests, full responses, token counts, finish reasons, errors, timing formulas, dispatch evidence, MTP metrics, memory and raw logs. Report skipped GPU tests as unqualified, not passed. Keep aggregate throughput separate from per-request speed and historical results separate from the new candidate.

Use a review branch for test/harness fixes. Do not publish a wheel, move release tags, change production defaults or rewrite historical measured results merely because CPU CI is green. Preserve upstream notices and actual code/design provenance; never claim source-level reuse is independent implementation.
