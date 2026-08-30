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
| EXL3 K2 | 2 bpw | pending | | | | |
| EXL3 K2/K3 mix (6 layers K3) | 2.14 bpw | pending | | | | |

K2 and the mix are captured on the same 512 windows through the same recipe
runtime, after the K-pool tail fix, so the two are directly comparable and
comparable to the FP8 anchor.
