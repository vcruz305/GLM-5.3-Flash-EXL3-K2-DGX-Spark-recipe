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
| EXL3 K2/K3 mix (6 layers K3) | 2.14 bpw | 512 | **0.3121** | [0.2993, 0.3249] | **0.795** | median 0.106, p99 3.13, p99.9 6.33; same 512 contexts and path as K2 |

### K2 vs the mix, paired on the same 512 contexts

Both captures scored the same contexts, so the per-context difference is the
right test, and it is not close:

| statistic | K2 | mix | difference |
|---|---:|---:|---:|
| token-mean KLD | 0.3346 | 0.3121 | **-6.7%** |
| median KLD | 0.117 | 0.106 | -9.4% |
| p99 / p99.9 | 3.33 / 6.47 | 3.13 / 6.33 | -6% / -2% |
| top-1 agreement | 0.788 | 0.795 | **+0.71 pts** |
| worst context mean | 1.219 | 1.133 | |

Paired over 512 contexts: the mix is lower by **0.0225 nats per context, 95% CI
[0.0207, 0.0243], t = 24.4**, and lower on **505 of 512 contexts (98.6%)**. The
top-1 gain is +0.0071 with CI [+0.0065, +0.0077]. Six K3 layers (24, 27, 35, 37,
42 in the scored decoder, plus 45, the MTP layer, which only matters at serve
time) buy 6.7% of the K2 divergence for 5.7 GiB. That is the same signal the
ladders showed as higher MTP acceptance, now measured directly; sixcat's 120
items (84.17 vs 83.33) could not resolve it.

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
