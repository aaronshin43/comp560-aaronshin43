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

## Key Findings — Plain Addition (A vs B)

1. **The hypothesis is not supported.** Target masking does not converge faster to the 80% threshold — both conditions reach it at roughly the same iteration count. Early convergence speed is similar.

2. **The real effect is a higher accuracy ceiling.** After iter 5,000, Cond B continues improving to ~90% while Cond A plateaus and oscillates between 80–85% for the remaining 5,000 iterations.

3. **Val loss divergence is the mechanism.** Cond A's val loss reaches a minimum (~1.07) around iter 3,000, then diverges to 2.1–3.0 by iter 10,000 — the model overfits to input-token patterns in the training stream. Cond B's val loss decreases monotonically throughout (2.6 → 0.12), indicating a clean optimization landscape.

4. **Overhead is negligible.** Target masking adds ~8% wall-clock time (180s → 194s) with no difference in GPU utilization.

---

## Stretch Conditions — Scratchpad (C vs D)

- **Dataset:** 2-digit addition with bracket-based scratchpad outputs (e.g. `[5+7=12,C1][1+2+1=4,C0]42`)
- **Model:** same architecture, `block_size=128` to accommodate longer sequences
- **Question:** Does the masking efficiency gap grow with longer outputs?

### Final Accuracy (iter 10,000)

| | s1 | s2 | s3 | Mean |
|---|---|---|---|---|
| Cond C (scratchpad, no mask) | 86.3% | 87.7% | 86.4% | **86.8%** |
| Cond D (scratchpad, target mask) | 86.7% | 84.8% | 86.6% | **86.0%** |

### Key Findings — Scratchpad (C vs D)

1. **The secondary hypothesis is not confirmed.** C and D perform nearly identically (~86–88%), and neither reaches 90%. The C vs D gap at 10,000 iterations is effectively zero (−0.8pp), compared to the +6.4pp gap for plain addition.

2. **The masking benefit scales with the input token fraction, not output length.** For scratchpad, the input (`12+34=`) is only ~22% of the sequence vs ~70% for plain. Masking removes far less noise per sequence, so the accuracy benefit disappears.

3. **Block density matters.** With scratchpad sequences averaging ~32 chars and `block_size=128`, each block contains ~4 equations vs ~8 for plain. Fewer cross-equation boundaries reduce the overfitting pressure even without masking.

4. **Both C and D show val loss divergence** (minimum ~0.27–0.34 at iter 3,500, rising to 0.43–0.65 by iter 10,000) — unlike Cond B which decreased monotonically. However, the divergence is far less severe than Cond A (which reached 2.1–3.0).

---

## Comparison with Exp 4

| Experiment | Condition | Iters | Val AR Accuracy |
|---|---|---|---|
| Exp 4 | No mask, block_size=64 | 5,000 | 76.7% |
| Exp 4 | Target mask, block_size=64 | 5,000 | 88.0% |
| Exp 6 | No mask — mean (A) | 10,000 | 83.5% |
| Exp 6 | Target mask — mean (B) | 10,000 | **89.9%** |
| Exp 6 | Scratchpad, no mask — mean (C) | 10,000 | 86.8% |
| Exp 6 | Scratchpad, target mask — mean (D) | 10,000 | 86.0% |

Doubling iterations from 5,000 to 10,000 improves Cond A by ~6.8 points but it does not catch up with the masked model. The masked model shows only marginal improvement from 88.0% to 89.9% over the same additional 5,000 iters, suggesting it largely converged by iter 5,000.
