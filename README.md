# GLM-5.3-Flash EXL3 on one NVIDIA DGX Spark

Serve **GLM-5.3-Flash EXL3 K2 or K2/K3-mix** on a single GB10 using the compatible vLLM fork, ExLlamaV3 and the canonical `vllm-exl3` plugin. The standard recipe uses TP=1 and native MTP k=2. **Stock vLLM is not a replacement for this runtime.**

## Credits and provenance

The **EXL3 trellis format, codebooks and quantization method** are [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by Turboderp. The original routed-expert integration preserved under `runtime/exl3_plugin/` is substantially derived from the earlier `overlay/exl3.py` by **Mia's AI Lab**, including @plotarmordev, in [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).

Full source lineage, historical MIT notices and copyright statements remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Current recipe work is **AGPL-3.0-only**; its previous MIT text is preserved in [LICENSE.MIT](LICENSE.MIT). The active plugin's provenance policy is maintained in [vllm-exl3](https://github.com/vcruz305/vllm-exl3). This is independent community work, not an endorsement by Z.ai, NVIDIA or vLLM.

## Start here

Read [AGENTS.md](AGENTS.md) before modifying an existing installation. Reuse downloaded checkpoints and a working runtime; do not reinstall the stack simply to test a plugin change.

| Component | Source or requirement |
|---|---|
| Standard quant | [vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2): K2 routed experts; historical pack size 91.017 GiB, 120 shards |
| Mixed quant | [vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix): per-layer K3 overrides; inspect the actual config/index and revision |
| Prebuilt runtime | [spark-vllm wheels](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm), aarch64, Python 3.12, PyTorch 2.13.0+cu130 / CUDA 13 |
| Active plugin | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3); `runtime/exl3_plugin/` here is historical provenance, not the candidate source |
| Hardware | One GB10 / SM121, compute capability 12.1; shared CPU/GPU memory must retain OS headroom |
| Source model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |

K2 describes **routed-expert precision**, not every parameter in the checkpoint. The mixed pack does not necessarily have the K2 pack's byte size. Verify the selected pack's files instead of applying one size/count claim to both.

## New candidate: what to test

The development plugin reports version `0.4.2`. The pinned executable candidate is **`d3cfd394920360d69f820d2dc96f8292a9e10283`**, including the SUH-cache change. CPU/source/packaging CI passed for that revision; **new GB10 speed or quality gains have not yet been established**.

| Feature | Current status |
|---|---|
| K2/K3/K4 native row-cap controls | Affect the real resolver after plugin registration; test actual native/fallback dispatch |
| Gate/up SUH compatibility cache | Cached at fused-state construction, with first-use fallback for direct callers; test shared/distinct rotations and parity |
| Runtime diagnostics | Reports the calling process; verify the live worker separately |
| K2/K3 grouped-prefill planner | Default off, bounded planning only; **no grouped CUDA executor** |
| Fused scratch-row request | Diagnostic only; actual capacity remains **2048**, override inactive |

Use [the TP1 qualification protocol](docs/TP1_POLICY_AB.md). To isolate the SUH-cache change, compare the candidate against **`28041c423a81fe033e7128d8888e3762fb914910`**, which already has the per-bit policy. For the broader candidate comparison, preserve and benchmark the exact known-working installed artifact first.

The existing K4 fat-GEMM/scatter microbenchmarks do **not** demonstrate accelerated K2/K3 grouped prefill. Native backend selection also does not guarantee that every call takes a native kernel.

## 1. Establish the runtime

For an existing installation, activate it and inspect it first:

```bash
source "${VENV:-$HOME/venvs/glm53-exl3-local}/bin/activate"
python scripts/preflight.py
```

On a fresh compatible machine, use `bash scripts/install_prebuilt.sh`, then activate its environment and rerun preflight **before downloading weights**. Treat a failed preflight as a stop requiring diagnosis. The installer has a floating `vllm-exl3>=0.3.1` dependency and therefore is **not an immutable benchmark baseline**. Record the resolved artifacts and hashes. Do not blindly replace an existing working wheel or install from moving `main`.

Historical source-tag/import issues are documented in [the archived README](README_HISTORY_2026-09-09.md). Test the actual artifact rather than assuming a Git tag and wheel contain identical files. Source builds of the entire runtime are an advanced alternative in `scripts/install_local_runtime.sh`, not the default response to an import failure.

Keep the active environment's `bin/`, CUDA 13 `nvcc`, and `ninja` on PATH. The launcher adds common CUDA paths, but cannot repair an incompatible runtime. The prebuilt recipe uses FlashInfer `0.6.18rc10`; retain the measured stack during A/B testing rather than upgrading/downgrading dependencies opportunistically.

For the **candidate**, preserve a rollback artifact, then in a prepared environment with the native build prerequisites:

```bash
bash scripts/install_candidate_plugin.sh
python scripts/preflight.py
python -c "import torch, vllm_exl3_c; print(vllm_exl3_c.P2B_MOE_ABI_VERSION)"
```

This rebuilds the plugin, not the entire vLLM stack. The exact-ref marker is an installation record, not live-server identity. Restart the test server after replacing the plugin and verify its workers loaded the expected code/extension.

## 2. Select and verify one pack

Reuse the local model directory. When it is missing, download **one** selected pack after runtime preflight passes:

```bash
hf download vcruz305/GLM-5.3-Flash-EXL3-K2 \
  --local-dir "$HOME/models/GLM-5.3-Flash-EXL3-K2"
```

For the mixed variant, substitute `GLM-5.3-Flash-EXL3-K2K3-mix` in both places. `scripts/download_weights.sh` is also available. Resume with the existing local directory; do not force a new download or download a second large pack unnecessarily.

Verify `config.json`, the safetensors index, referenced shards, model revision and chat template. Routed metadata must declare `quant_method=exl3`, base `bits=2`, and `codebook=mcg`; preserve mixed `layer_bits` overrides. Set `MODEL_DIR` explicitly so the launcher cannot choose the other installed pack.

## 3. Serve and smoke-test

The template patch is local checkpoint metadata, not a weight conversion. Back up/hash the template before patching and use the same template for every A/B variant.

```bash
source "${VENV:-$HOME/venvs/glm53-exl3-local}/bin/activate"
export MODEL_DIR="$HOME/models/GLM-5.3-Flash-EXL3-K2"
python scripts/patch_chat_template_thinking.py "$MODEL_DIR/chat_template.jinja"

HOST=127.0.0.1 PORT=8888 \
SPEC_METHOD=mtp MTP_TOKENS=2 \
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=2048 \
GPU_MEM_UTIL=0.87 EXL3_FUSED_MOE=1 VLLM_EXL3_MOE_KERNEL=native \
ENABLE_PREFIX_CACHING=0 \
bash scripts/serve_one_spark.sh
```

These are **cold-prefix qualification settings**. The launcher normally enables prefix caching; cached-prefix conversational measurements belong in a separate test. It binds to `0.0.0.0` by default, so use `HOST=127.0.0.1` for local testing. Do not expose an unauthenticated benchmark server publicly.

The launcher selects TP=1, FP8 KV, model-specific reasoning/tool parsers and MTP graph capture sizes including 3. Do not add a Marlin MoE backend or combine MTP with DFlash. For the matched no-spec control, use `SPEC_METHOD=none` and hold all other settings fixed.

From a second shell:

```bash
curl --fail http://127.0.0.1:8888/health
curl --fail http://127.0.0.1:8888/v1/models
curl --fail http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Check the served model ID, context, nonempty response and finish reason. `/health` alone is insufficient: historical memory stalls left it responsive. Capture packed-expert loading and actual dispatch evidence; the absence of a particular log string is not by itself proof of the selected kernel.

## 4. Measure without misleading receipts

**Do not use the current schema-1 `bench_tp1_policy.py` output alone as promotion evidence.** Its known limitations are:

- It times the first `content` delta, ignores reasoning deltas, uses total completion tokens over post-first-content time, and waits until stream closure. This can misstate TTFT/decode rate.
- It records the client's local policy/environment and candidate marker, not verified serving-worker state.
- It retains only a response prefix, omits a prompt hash and finish reason, and does not establish quality, MTP acceptance or server peak memory.
- `compare_policy_receipts.py` computes percentage changes without rejecting mismatched models, prompts, serving configurations or measurement semantics.

The [qualification protocol](docs/TP1_POLICY_AB.md) requires a validated measurement harness and server manifest before performance conclusions. Retain full outputs, exact request payloads, timing boundaries, token usage and errors. Prefix-cache hits, prompt prefill, post-first-output throughput and total request throughput are different measurements. MTP can emit multiple tokens in one SSE event; event counts are not token counts.

## Evidence and long-context boundary

Existing measurements are retained, not rewritten as candidate results:

| Evidence | Where to read it |
|---|---|
| Serving/speculation measurements | [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) |
| K-pool tail bug and fixed-build verification | [docs/KPOOL_TAIL_BUG.md](docs/KPOOL_TAIL_BUG.md) |
| Fat-expert stall, allocator behavior and long-context runs | [docs/IMPROVEMENTS_AND_EVIDENCE.md](docs/IMPROVEMENTS_AND_EVIDENCE.md) |
| Teacher/K2/mixed precision fidelity analysis | [docs/KLD.md](docs/KLD.md) |
| Sixcat results, including truncation/loop caveats | [docs/SIXCAT.md](docs/SIXCAT.md) |
| Original README tables, dense-overlay and DFlash experiments | [README_HISTORY_2026-09-09.md](README_HISTORY_2026-09-09.md), an unchanged historical snapshot containing superseded instructions/claims |

Use 8K for the initial matched performance sweep and 64K for follow-on serving/evaluation once the candidate passes. Historical 98,304-token faults describe an earlier K-pool build, not a universal current limit. Conversely, the reported **258,048-token completion** used a specific configuration, pinned KV pool, expandable allocator segments and an additional **unshipped indexer workspace patch**. It is not an out-of-the-box guarantee for this checkout. High-context recall, output quality and sustained decode rate require separate evidence.

Keep `TEMP_ROWS_FUSED=2048`. The launcher defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; preserve allocator policy during matched tests and monitor host `MemAvailable`, swap and memory drift. Do not jump directly to 256K, change global swap settings, or equate a KV-capacity quotient with tested concurrent sessions.

The archived headline also mixes end-to-end loading, raw shard reads, K4 kernel tests and whole-model figures. Those are not interchangeable; reproduce a clearly specified measurement before reusing a headline. Existing dense-overlay/DFlash experiments are optional historical paths and should not be added to this candidate's baseline.

## License

Current recipe work is distributed under **AGPL-3.0-only**; see [LICENSE](LICENSE). [LICENSE.MIT](LICENSE.MIT) preserves the earlier project text. Original third-party notices remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and runtime dependencies are separately licensed; this repository does not redistribute the model weights.
