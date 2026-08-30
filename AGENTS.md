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

## Known engine-fatal bug: read before you trust a result

GLM-5.3's sparse-MLA K-pool tail cache is written out of bounds on long
generations. It is not EXL3 and not fixed by any flag.

**Do not conclude a build is healthy because a run completed.** Every affected
build performs the bad writes; whether one escapes its allocation and kills the
engine depends on memory layout. Contained writes silently corrupt another
layer's sparse-attention index instead of crashing.

- Trigger is **generated** tokens (~2.2k reported), not context length.
- Reproducer: `bash scripts/repro_kpool_tail_overrun.sh`
- Mechanism and fix status: `docs/KPOOL_TAIL_BUG.md`

**Check your own build instead of guessing.** Serve with
`GLM_KPOOL_TAIL_BOUNDS=1`, then:

```bash
SERVER_LOG=/path/to/server.log bash scripts/soak.sh
```

It generates ~20k tokens and fails if a single out-of-bounds tail write occurs.
Zero overruns over a long generation is the only evidence that means anything;
a run completing is not.

If you are running an eval or benchmark on this runtime, say in your report that
it ran on a build with this open defect.

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
