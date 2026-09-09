# GB10 / TP1 candidate qualification

Qualify `vllm-exl3` on the existing authorized **single GB10** using `vcruz305/GLM-5.3-Flash-EXL3-K2`, then the K2/K3-mix pack when already available or explicitly provisioned. Reuse the existing compatible runtime and downloaded weights. This protocol does not authorize instance lifecycle changes, unrelated process termination, checkpoint deletion or production promotion.

## 1. Pin source and preserve rollback

Executable candidate: `d3cfd394920360d69f820d2dc96f8292a9e10283`.

Pre-SUH-cache control: `28041c423a81fe033e7128d8888e3762fb914910`.

The control already includes the per-bit row policy. Use it to isolate SUH caching; preserve the known-working installed artifact as an additional baseline for the entire candidate. Development version `0.4.2` alone cannot distinguish these builds.

Read both repositories' AGENTS/README files. Save clean source revisions, any local diffs, dependency manifests, and rollback wheels before changes. Verify aarch64/GB10 capability 12.1, Python 3.12, the compatible vLLM fork, CUDA 13/PyTorch stack, ExLlamaV3 headers/extension and available memory/disk. Run recipe preflight before any new model download.

The candidate installer now pins the candidate above. Build only the plugin where possible; do not let dependency resolution replace the working vLLM fork. Check PEP 610 `direct_url.json` for VCS installs, package/extension paths and hashes, native ABI, build flags and target architecture. A local candidate-ref marker is supporting evidence, not sufficient provenance or proof of what the server loaded.

Record the recipe Git SHA too. For known-working artifacts that lack a source SHA, report that honestly and retain the artifact hash rather than inventing provenance.

## 2. Validate the measurement harness before collecting evidence

The checked-in schema-1 `scripts/bench_tp1_policy.py` is a legacy helper, not a complete qualification harness. It ignores reasoning deltas, starts timing at first answer content, divides total output tokens by post-first-content time, waits for connection closure, keeps only a response prefix and captures the client's environment. `compare_policy_receipts.py` does not reject mismatched experiments.

Either fix these helpers on a review branch with tests or use a validated compatible benchmark harness with equivalent evidence. Use the **same corrected harness revision** for baseline and candidate; do not compare old and corrected metric definitions silently.

Required behavior:

- Detect the first actual generated output from the stream, including this fork's reasoning field (`reasoning_content` or `reasoning`, as applicable), not a role-only delta. Track time to first answer content separately.
- Timestamp the last generated output separately from finish/usage/DONE. Stop at `[DONE]`, reject stream errors/truncation at transport level, and handle usage-only final events. Retain finish reasons, missing-usage status and zero/one-token edge cases without fabricated rates.
- Count output **tokens**, not SSE events. MTP may emit several tokens per event. Record total-output end-to-end tokens/s separately. For a vLLM-style normalized post-first-output rate, document the convention `(N-1)/(T-TTFT)` for `N>1`, define T exactly, and disclose first-event batching. It is not an exact per-token timestamp trace. Never divide all reasoning+answer tokens by answer-only elapsed time.
- Fix the request's thinking mode explicitly. Use thinking off for the initial speed sweep; evaluate thinking on separately and retain both reasoning and answer text.
- Save the complete request payload, prompt SHA256, full outputs, token usage, finish reason, errors and timing samples. Never silently exclude failed requests.
- Bind each result to a **server** manifest: worker start/run identifier, PID linkage in private logs, model/config/index/template revisions, plugin source and loaded extension hash, vLLM/ExLlamaV3/PyTorch/CUDA versions, exact serve argv and allowlisted environment. Link manifests to the measured process. Clearly label standalone client diagnostics as client-side.
- Validate comparator invariants: same model and prompt hashes, request parameters/thinking mode, context/batching/KV/prefix-cache policy, concurrency and timing schema. Permit only the explicitly declared experiment differences, such as plugin revision or one row cap. Label intentional cross-quant comparisons separately.

Add synthetic-stream tests for role-only chunks, reasoning then content, batched content, usage-only events, DONE on an open connection, missing usage, errors, malformed/incomplete streams and zero/one-token output. Add comparator tests proving mismatched requests/configurations are rejected. Do not use a helper's successful exit as proof that its measurements are meaningful.

Metric reference: [vLLM benchmark CLI definitions](https://docs.vllm.ai/en/stable/benchmarking/cli/). Use the installed fork's verified API/CLI rather than assuming all current upstream options exist there.

## 3. Establish correctness and real dispatch

From the candidate source checkout, run:

```bash
python -m pytest -q tests/test_runtime_policy.py tests/test_prefill_policy.py
python -m pytest -q tests/test_native_moe_contract.py tests/test_fat_distinct_suh.py
python -m pytest -q
```

Retain the entire result, including skips. Check the Python source under test is the same revision as the installed server; source-path pytest imports alone do not prove this. Required CUDA checks that skip because a module or device is missing are unqualified.

In a fresh preflight process use `vllm_exl3.register()` before `runtime_diagnostics()`. Confirm registration/ABI/caps, then separately capture live-worker policy and execution evidence. Check each width's override independently, fallback on invalid/unset input and zero disabling native eligibility. Defaults should match the control unless a variable is deliberately changed.

For SUH caching, add tests proving equal and distinct gate/up rotations choose the same path/output as the uncached control, normal fused-state construction caches the decision, and direct callers compute it only once. Run actual checkpoint experts, not synthetic data alone. Preserve immutable loaded rotations; rebuild state when weights/rotations change. Report max absolute error, relative error and the existing allclose criteria, not cosine alone. Do not relax tolerances to make a failure disappear.

Test the supported multi-row Python dispatch and its fallback boundaries. The underlying fused C entry point has its own row contract; do not bypass it to force a larger batch. Exercise clipping, repeated/zero-weight routes, supported invalid-route handling, noncontiguous inputs and graph replay with changed inputs/routing where the API promises support. Use malformed/null pointers only in tests that guarantee rejection before launch.

Use CUDA synchronization/events **outside capture** for isolated timing. Do not insert synchronization or host tensor reads into a graph hot path. Record profiler traces in separate diagnostic runs so profiler overhead does not contaminate performance receipts.

## 4. Matched serving matrix

Initial serve settings: TP=1, FP8 KV, explicit MODEL_DIR, `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=1`, `GPU_MEM_UTIL=0.87` unless the established baseline requires another fixed value, `EXL3_FUSED_MOE=1`, `VLLM_EXL3_MOE_KERNEL=native`, `SPEC_METHOD=mtp`, `MTP_TOKENS=2`, `HOST=127.0.0.1`, and `ENABLE_PREFIX_CACHING=0`.

Hold the allocator, launch flags, checkpoint, template, thinking/sampling settings, output budget, runtime and GPU power/thermal conditions fixed within a pair. Keep a request's actual tokenized prompt plus generation allowance within max context. Restart only the experiment server between code/policy variants. Never change a variable in the benchmark client's shell and assume it changes an existing worker.

Run the known-working baseline, pre-cache control and post-cache candidate at their defaults. Then screen K2 caps **1, 2, 4, 8**, leaving K3/K4 unset. Keep actual verification-row and native/fallback evidence: several caps can select identical paths for a particular workload. Avoid redundant full sweeps once equivalence is demonstrated; report it instead of inventing a cap-specific gain.

Use a fixed small battery of prose, code, reasoning/arithmetic and structured-output requests. Warm the runtime separately from prefix caching. Screen settings, then use at least two warmups and seven measured requests per workload for finalists, with a baseline/candidate/baseline order to check drift. Repeat the winning/control pair with speculation **off** to distinguish target-path effects from MTP acceptance changes.

For the mixed pack, validate its real K3 overrides, preserve the selected K2 cap and sweep K3 **1, 2, 4** separately. If that pack is absent, report it untested rather than downloading another large checkpoint without scope approval. Never treat a cross-quant quality/performance difference as a same-quant plugin speedup.

## 5. Prefill, quality and memory

Compare pre-cache control and candidate using identical actual prompt lengths approximately 2K, 8K and 32K, then 64K only if headroom permits. Choose a fixed larger max context for each matched set and leave room for generation. Prefix caching must be disabled or verifiably reset between cold-prefix measurements. Cold prefix does not mean dropping the OS page cache. Warm-prefix conversational latency is a separate suite.

Measure TTFT and server-side prefill duration where available. Prompt tokens divided by TTFT is an end-to-end prompt rate, not pure kernel throughput. Profile fat-path activity and SUH equality-call counts; report a missing hot-path exercise as a coverage gap, not evidence that caching cannot help.

Retain full-output quality checks: exact arithmetic, instruction adherence, valid JSON, repeated-token/loop detection, usable code checked in a restricted sandbox, and needle recall at tested longer contexts. Compare against the same quant before the changes. Small floating-point differences can alter greedy outputs; investigate divergence rather than equating any string difference with corruption. HTTP 200 and three smoke questions are not intelligence benchmarks.

Capture **worker-side** peak allocated and reserved CUDA memory, plus OS `MemAvailable`, swap, process memory and available GPU telemetry over repeated workloads. Client-side CUDA statistics are not server peaks. Use a conservative documented host-memory guard, such as 8 GiB available or a higher operational reserve; stop only the test server on sustained pressure, no request progress, NaNs or CUDA faults. Record why the run stopped. Do not alter swappiness/drivers/clock policy or drop global caches as a hidden optimization.

Reference: [PyTorch CUDA memory and synchronization semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## 6. Negative controls and reporting

For this candidate assert:

```text
per_bit_native_policy_installed = true
fused_temp_rows_actual = 2048
fused_temp_rows_override_active = false
grouped_prefill.execution_available = false
```

The requested flags can change diagnostic/planner fields, not activate a nonexistent executor or shrink the live arena. A default-off/ineligible plan must report zero windows/scratch, and an eligible plan must remain within its own row budget. The planner's scratch formula is not an end-to-end bound on existing runtime memory. Do not implement a new grouped CUDA kernel inside this qualification exercise.

Keep 258K out of the initial matrix: the historical run included an unshipped indexer workspace patch. A KV-capacity quotient is not demonstrated concurrent serving. Concurrency greater than one is a separate qualification after the C1 case passes.

Deliver a report plus raw JSON/JSONL receipts, manifests, logs, test results, correctness outputs, selected profiler traces, harness diff/tests and rollback commands. State PASS / FAIL / INCONCLUSIVE / NOT TESTED for each gate. Report median and spread, actual prompt/output lengths, warm/cold semantics, memory and errors. Quantify gains only where repeated matched comparisons support them; show negative or noisy results too.

Keep defaults unchanged until correctness, GPU/graph coverage and reproducible end-to-end benefit are established. A practical screening threshold such as 5% is a proposed promotion criterion, not a measured result; require repeatability beyond observed variance and no material quality/memory/latency regressions. Do not publish wheels/tags or push performance claims from this test without a separate promotion decision. Preserve upstream attribution in any harness/test fixes.
