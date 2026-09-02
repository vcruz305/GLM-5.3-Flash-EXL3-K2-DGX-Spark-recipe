# Instructions for coding agents

Read this before running anything. It exists because the common failure here
costs hours: an agent installs stock vLLM, downloads 91 GiB, and only then finds
that the runtime cannot load the model.

## The one thing to get right

**Never run `pip install vllm`.** Stock vLLM cannot serve this pack. It has:

- no `exl3` quantization method, so `--quantization exl3` is rejected
- no `Glm5Next` architecture, so the config will not map to a model class

Neither is a flag, an extra, or a version away. `glm5next` lives in a fork and
EXL3 comes from this repo's out-of-tree plugin. If you find yourself
troubleshooting "vLLM lacks EXL3 and glm5next", you installed the wrong vLLM;
stop and install the runtime below.

## K-pool tail bug: fixed in the 2026-08-30 wheels

Wheels published before 2026-08-30 write GLM-5.3's K-pool tail cache out of
bounds on long generations (engine death or silent attention corruption).
Reinstall with `bash scripts/install_prebuilt.sh`. To verify a build yourself,
serve with `ENFORCE_EAGER=1 GLM_KPOOL_TAIL_BOUNDS=1` and run `scripts/soak.sh`;
the detector only counts in eager mode. Details: `docs/KPOOL_TAIL_BUG.md`.

## Order of operations

```bash
python scripts/preflight.py          # seconds. Do this FIRST, before anything else.
bash scripts/install_prebuilt.sh     # minutes. Prebuilt wheels, no compiler.
bash scripts/download_weights.sh     # 91 GiB. Only after preflight passes.
python scripts/patch_chat_template_thinking.py ~/models/GLM-5.3-Flash-EXL3-K2/chat_template.jinja
SPEC_METHOD=mtp MTP_TOKENS=2 MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.87 bash scripts/serve_one_spark.sh
```

`scripts/preflight.py` exits non-zero and prints the fix. Treat a non-zero exit
as a hard stop, not as something to work around.

## Do not build from source unless you mean to

`scripts/install_local_runtime.sh` compiles vLLM and ExLlamaV3. It takes tens of
minutes at best and hours on a cold machine. It is for changing the patches, or
for a Python or CUDA combination the wheels do not cover. It is not the normal
path and an agent should not reach for it to "fix" an import error.

## Hard requirements

The prebuilt wheels carry compiled CUDA extensions, so these are not negotiable:

| Requirement | Value |
|---|---|
| Architecture | `aarch64` |
| GPU | GB10, compute capability 12.1 (SM121) |
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 (CUDA 13) |

On anything else, build from source and expect to fix things.

## Things that look like bugs and are not

- **~12 minutes per server start.** The checkpoint is 91 GiB. Every flag change
  is a fresh load. Budget for it rather than assuming a hang.
- **9.6 to 9.8 tok/s with no speculation.** That is the floor, not a missing
  kernel. Use MTP k=2.
- **A pip conflict on `flashinfer-python`.** vLLM's metadata pins 0.6.17; this
  recipe runs 0.6.18rc10, which is what every measurement was taken on. The
  warning is expected. Do not downgrade to silence it.
- **`hf download --resume-download`.** The flag does not exist. `--local-dir`
  already resumes. Never `--force-download` a partial destination.
- **`scheduled_spec_decode_tokens=[-1, ...]`** in a scheduler dump is shape
  padding for the first speculative step, not corruption.

## Choices already measured, do not re-litigate

- Speculation: **native MTP k=2**. Never pass MTP and a DFlash draft together.
- MoE: `EXL3_FUSED_MOE=1`. Do **not** pass `--moe-backend marlin`.
- KV: `--kv-cache-dtype fp8`. Sequences: `--max-num-seqs 1` with speculation.
- Serving context: **65536**. 131072 allocates but a prompt at or above 98,304
  tokens faults and kills the engine.

Full numbers and the reasoning are in [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md).

## nvcc must be on PATH

vLLM's `has_flashinfer()` returns False without `nvcc` on PATH and then rejects the only sparse-MLA backend for GB10 (`No valid attention backend found for cuda ... FLASHINFER_MLA_SPARSE_SM120`). `scripts/serve_one_spark.sh` adds `/usr/local/cuda-13.0/bin` itself and `scripts/preflight.py` checks it; if you launch `vllm serve` by hand, `export PATH=/usr/local/cuda-13.0/bin:$PATH` first. FlashInfer's JIT also runs `ninja` from the venv's `bin/`, so activate the venv (or put `~/venvs/glm53-exl3-local/bin` on PATH) rather than calling the venv's python by absolute path from a bare shell.

## Context limits (measured 2026-08-31, updated 2026-09-01)

`MAX_MODEL_LEN=262144` boots (KV pool 1.09M tokens on K2) and prompts up to
**163,479 tokens are verified** with perfect needle recall (prefill ~590 tok/s,
decode ~17–20 tok/s). The 163k-180k wedge (fused-MoE fat-expert fallback) is
fixed by the `TEMP_ROWS_FUSED` row cap below. A second, independent wedge past
roughly 200k-230k prompt tokens, an allocator ratchet on GB10 unified memory
in vLLM's sparse-indexer chunked prefill, is fixed by
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which
`scripts/serve_one_spark.sh` now sets by default; it is **required** for
prefills above ~200k tokens on one Spark. A cold 258,048-token prefill is
verified passing with that flag set (single request, pinned 3 GiB KV pool,
speculative decoding off). See docs/IMPROVEMENTS_AND_EVIDENCE.md section 0b
and the README's long-context section.

Also field-verified: a root-owned `~/.triton/cache` (from an earlier sudo run)
breaks the user-mode serve; chown it or set `TRITON_CACHE_DIR` to a writable dir.

## Plugin home (updated 2026-09-01)
The EXL3 plugin's canonical home is https://github.com/vcruz305/vllm-exl3
(package `vllm_exl3`; `glm53_exl3_plugin` remains as a compat shim). The copy
under `runtime/exl3_plugin/` is provenance only — make plugin changes in the
vllm-exl3 repo and mirror the built wheel to the HF spark-vllm repo, replacing
the previous wheel (two wheels break `pip install dir/*.whl`). The fat-expert
row cap `TEMP_ROWS_FUSED` is 2048; 128 caused the >163k prefill stall
(see docs/IMPROVEMENTS_AND_EVIDENCE.md section 0).
