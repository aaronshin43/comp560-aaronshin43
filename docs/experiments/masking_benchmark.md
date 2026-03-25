# Experiment 6 — Target Masking Benchmark

**Directory:** `masking_benchmark/`
**Question:** Does target masking improve training efficiency — i.e., does it reach the same accuracy in fewer iterations?

Full results: see [masking_benchmark/EXPERIMENT.md](../../masking_benchmark/EXPERIMENT.md)

---

## Setup

- **Dataset:** 2-digit addition, 10,000 samples, `{"input": "12+34", "output": "46"}`
- **Model:** `n_layer=6, n_head=4, n_embd=128, block_size=64` (same as Exp 4)
- **Split:** 9,000 train / 1,000 val
- **Training script:** `train_benchmark.py`, `max_iters=10000`, `eval_interval=500`, `enable_tf_eval=False`
- **Eval metric:** AR generation exact-match accuracy, collected post-hoc on 20 checkpoints per run
- **Replications:** 3 seeds per condition (1337, 1338, 1339)

### Conditions

| Condition | `target_mask` |
|---|---|
| A — No mask | False |
| B — Target mask | True |

---

## Results

### Final Accuracy (iter 10,000)

| | s1 | s2 | s3 | Mean |
|---|---|---|---|---|
| Cond A | 83.9% | 82.5% | 84.2% | **83.5%** |
| Cond B | 89.8% | 90.2% | 89.8% | **89.9%** |

### Iterations to Threshold

| Threshold | Cond A | Cond B |
|---|---|---|
| 80% | ~4,000 iters | ~4,667 iters |
| 85% | ~5,500 iters (2/3 runs) | ~5,667 iters |
| 90% | never reached | ~8,667 iters |

---

## Key Findings

1. **The hypothesis is not supported.** Target masking does not converge faster to the 80% threshold — both conditions reach it at roughly the same iteration count. Early convergence speed is similar.

2. **The real effect is a higher accuracy ceiling.** After iter 5,000, Cond B continues improving to ~90% while Cond A plateaus and oscillates between 80–85% for the remaining 5,000 iterations.

3. **Val loss divergence is the mechanism.** Cond A's val loss reaches a minimum (~1.07) around iter 3,000, then diverges to 2.1–3.0 by iter 10,000 — the model overfits to input-token patterns in the training stream. Cond B's val loss decreases monotonically throughout (2.6 → 0.12), indicating a clean optimization landscape.

4. **Overhead is negligible.** Target masking adds ~8% wall-clock time (180s → 194s) with no difference in GPU utilization.

---

## Comparison with Exp 4

| Experiment | Condition | Iters | Val AR Accuracy |
|---|---|---|---|
| Exp 4 | No mask, block_size=64 | 5,000 | 76.7% |
| Exp 4 | Target mask, block_size=64 | 5,000 | 88.0% |
| Exp 6 | No mask — mean | 10,000 | 83.5% |
| Exp 6 | Target mask — mean | 10,000 | **89.9%** |

Doubling iterations improves Cond A by ~6.8 points but it does not catch up with the masked model. The masked model shows only marginal gain from 88.0% to 89.9% over the same additional 5,000 iters, suggesting it largely converged by iter 5,000.
