# KLD against the BF16 teacher

Method: the [GLM-5.3-Flash fidelity suite v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1)
protocol. The suite ships BF16 reference **pre-final-norm hidden states** for
512 sealed 2,048-token contexts (2,047 scored positions each) and one shared
BF16 head (`final_norm` + `lm_head`). A candidate is scored by capturing its own
pre-norm hidden states on the same token windows, replaying both operands
through the shared head, and computing exact full-vocab KL(reference ||
candidate) per position in fp32. No BF16 model is needed at scoring time.

Scripts: [`scripts/kld/capture_hidden.py`](../scripts/kld/capture_hidden.py)
(vLLM offline, eager, forward pre-hook on the language model's final RMSNorm via
`LLM.apply_model`), [`scripts/kld/score_kld.py`](../scripts/kld/score_kld.py).

## Scorer validation

Before any number of ours is reported, the scorer must reproduce the suite's
own published measurement on its FP8 capture.

| check | result | published |
|---|---:|---:|
| reference vs itself, 4 contexts | KLD 0.0, top-1 1.0 | (identity) |
| FP8 as-served vs BF16, 24 contexts, 49,128 positions | **0.0319 nats**, CI95 [0.023, 0.041]; median 0.0055; p99 0.396; top-1 **0.938** | 0.0281 nats; median 0.0049; p99 0.354; top-1 0.943 (10.48M positions) |

The subset lands inside its own confidence interval of the full-panel value and
matches on median, p99 and top-1, so the pre-norm assumption and the replay are
correct. Noise floor from the suite's determinism study: 8.7e-4 nats.

## Results

| checkpoint | routed-expert width | contexts | token mean KLD | CI95 | top-1 | notes |
|---|---:|---:|---:|---:|---:|---|
| Official FP8 (anchor) | 8-bit | 24 | 0.0319 | [0.023, 0.041] | 0.938 | our scorer on the suite's capture |
| EXL3 K2 | 2 bpw | 512 | **0.3346** | [0.3204, 0.3487] | **0.788** | median 0.117, p99 3.33, p99.9 6.47; 1,048,064 positions; captured as served (fused EXL3 MoE, fp8 KV) |
| EXL3 K2/K3 mix (6 layers K3) | 2.14 bpw | pending | | | | |

### Reading the K2 number

- **0.33 nats mean against BF16, about 12x the FP8 anchor (0.03).** That is the
  price of 2-bit routed experts on every MoE layer (K2 on layers 3 to 44; 0 to 2
  are dense), measured on the exact path that serves: fused EXL3 MoE kernels,
  fp8 KV cache, one 2048-token forward per context.
- **The median is 0.117.** Half of all positions sit within 0.12 nats of the
  teacher. The mean is pulled up by a tail: p99 is 3.33 nats and p99.9 is 6.47,
  so roughly one position in a hundred gets a distribution that disagrees hard
  with BF16. That tail is the mechanism behind the rare derailments sixcat sees
  on long constrained tasks (`ifeval:1300`) while math and tools stay at 100/95.
- **Top-1 agreement 78.8%.** Greedy picks the same token as BF16 on about four
  positions in five (FP8: 94%). MTP acceptance in the ladders tracks this
  number, which is why the K3 layers in the mix show up first as acceptance.
- The 95% CI is +-0.014 nats, so a mix-vs-K2 difference above ~0.03 nats on the
  same 512 contexts is real; sixcat's 120 items cannot see differences that size.

K2 and the mix are captured on the same 512 windows through the same recipe
runtime, after the K-pool tail fix, so the two are directly comparable and
comparable to the FP8 anchor.
