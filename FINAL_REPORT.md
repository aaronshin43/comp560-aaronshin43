# Target Masking in a Tiny Transformer: Higher Ceiling, Not Faster Convergence

Aaron Shin — COMP560, Spring 2026
Experiment directory: `masking_benchmark/`

---

## 1. Introduction

When a causal language model is trained on instruction-style data of the form `input → output`, the standard cross-entropy objective is computed over every token in the concatenated sequence. **Target masking** modifies this objective so that only output tokens contribute to the loss; input-region targets are replaced with a sentinel value (`-1`) that the loss function ignores. The forward pass, attention pattern, and parameter set are unchanged — only the reduction set of the loss differs.

A common intuition is that target masking should accelerate training: by concentrating the gradient signal on the tokens we actually care about, the optimizer should reach any given accuracy threshold in fewer iterations. This report tests that intuition under controlled conditions on a tiny Transformer trained on exhaustive 2-digit addition, and shows that it is wrong.

The central claim is:

> Target masking does not accelerate training. It raises the long-run accuracy ceiling, and the mechanism is val-loss divergence on input-region tokens that the unmasked model is silently overfitting to.

A secondary cross-task experiment shows that the magnitude of the masking benefit is governed by the **fraction of each sequence that is input**, not by the absolute output length.

---

## 2. Background and Setup

### 2.1 What Target Masking Does Mechanically

The training engine (`comp560-nanoGPT`) computes loss with PyTorch's `cross_entropy(..., ignore_index=-1)`. Target masking is implemented as a single pre-batch transformation: for each training sequence, the targets `y` corresponding to input-region tokens are replaced with `-1`, while output-region targets are left intact.

For example, the training pair `{"input": "12+34=", "output": "46\n"}` produces a target sequence of the form

```
y_unmasked = [ '2', '+', '3', '4', '=', '4', '6', '\n', ... ]
y_masked   = [ -1,  -1,  -1,  -1,  -1,  '4', '6', '\n', ... ]
```

Crucially, the input tokens are still consumed by the forward pass — the model attends over them and produces hidden states normally. The only change is that the optimizer is no longer asked to make those predictions accurate.

### 2.2 Why a 2-Digit Addition Testbed

The full Cartesian product of two-digit operands gives an exhaustive 10,000-sample dataset of pairs `{"input": "AB+CD=", "output": "ZZZ"}`. This testbed has three properties that make it suited for isolating the masking effect:

- **Deterministic ground truth**, so AR generation accuracy is a clean exact-match metric with no labeling noise.
- **High input/output token ratio** (≈70/30) — if the unmasked model is wasting capacity on input prediction, this is the regime where it should be most visible.
- **Character-level vocabulary** with 13 unique tokens, removing tokenizer artifacts as a confound.

The split is 9,000 train / 1,000 val, shuffled with a fixed seed.

### 2.3 Conditions

| Condition | `target_mask` | Description |
|---|---|---|
| **A — No mask** | `False` | Loss computed over all tokens (input + output) |
| **B — Target mask** | `True` | Loss computed on output tokens only |

### 2.4 Architecture and Training

| Parameter | Value |
|---|---|
| `n_layer` | 6 |
| `n_head` | 4 |
| `n_embd` | 128 |
| `block_size` | 64 |
| `max_iters` | 10,000 |
| `eval_interval` | 500 |
| Optimizer | AdamW, β = (0.9, 0.95) |
| Learning rate | 3e-4 → 3e-5, cosine, 200 warmup |
| Batch size | 128 |
| Seeds | 1337, 1338, 1339 |

Three seeds per condition were used to bound run-to-run variance.

### 2.5 Evaluation Protocol

AR accuracy is evaluated **post-hoc** on 20 named checkpoints saved every 500 iterations (`ckpt_00500.pt` through `ckpt_10000.pt`). At each checkpoint, the model is prompted with `input=` only and generates autoregressively until newline; the produced digits are compared exactly to ground truth on the full 1,000-sample val split. This decouples the convergence-curve measurement from the training loop and guarantees identical evaluation conditions across all checkpoints and seeds.

---

## 3. The Speed Hypothesis Fails

The first prediction to test is: *if target masking concentrates gradient signal, it should reach any given accuracy threshold in fewer iterations.*

The threshold-crossing data tells a clear story:

| Threshold | Cond A (no mask) | Cond B (target mask) |
|---|---|---|
| 80% | ~4,000 iters | ~4,667 iters |
| 85% | ~5,500 iters (2/3 seeds) | ~5,667 iters |
| 90% | never reached | ~8,667 iters |

For the first ~5,000 iterations the two conditions are essentially indistinguishable. If anything, the unmasked condition reaches 80% slightly *earlier* on average (4,000 vs 4,667 iters), though the difference is well within seed-to-seed variance. There is no early-training acceleration from masking. The speed hypothesis is refuted.

This negative result is sharp enough that it does not require a statistical test — three seeds, two conditions, twenty checkpoints each, and the early curves overlap. Any speedup the masked condition enjoys is small enough to be invisible against ordinary seed noise.

---

## 4. The Ceiling Effect

The two conditions diverge after iteration 5,000, but not in the way the speed hypothesis predicted. Instead of one curve overtaking the other on the way to a shared asymptote, the unmasked condition simply stops improving while the masked condition continues to climb.

| Phase | Cond A | Cond B |
|---|---|---|
| iter 0 – 5,000 | Similar trajectory, ~82% mean | Similar trajectory, ~84% mean |
| iter 5,000 – 10,000 | Flat oscillation, 80–85% | Sustained improvement to 89–90% |
| Final (iter 10,000) | **83.5 ± 0.9%** | **89.9 ± 0.2%** |

The +6.4pp gap is consistent across seeds (Cond B's seeds land at 89.8, 90.2, 89.8 — a standard deviation of 0.2pp), and Cond A never crosses 85% on any seed in the second half of training. This is the central empirical finding of the report:

> Target masking buys a higher ceiling, not a steeper slope.

The cost is a wall-clock overhead of ~8% (180s → 194s mean training time) and no measurable change in GPU utilization (~58–59% in both conditions). For a 6.4pp accuracy improvement under identical model, data, and training compute, this is a one-sided trade.

---

## 5. Mechanism: Val-Loss Divergence on Input Tokens

The accuracy curves alone show *that* masking changes the long-run behavior; the validation loss curves show *why*.

### 5.1 The Val-Loss Curves

| Iter | A (mean) | B (mean) |
|---|---|---|
| 0 | 2.67 | 2.63 |
| 1,000 | 1.43 | 1.02 |
| 2,000 | 1.14 | 0.26 |
| **3,000** | **1.08 (min)** | 0.17 |
| 5,000 | 1.27 | 0.14 |
| 7,000 | 1.95 | 0.12 |
| 10,000 | **2.69 (final)** | **0.12 (final)** |

Condition B's val loss decreases monotonically across all 10,000 iterations and stabilizes near 0.12. Condition A's val loss reaches a minimum near 1.07 at iteration 3,000 — the same iteration window where its AR accuracy first crosses 80% — and then diverges steadily upward, ending the run at 2.1–3.0. This is a textbook overfitting signature.

### 5.2 What the Unmasked Loss Is Overfitting To

The cross-entropy loss is averaged over every token position in the sequence. The divergence in Cond A must therefore be located on some specific subset of token positions. The output tokens are tightly determined by the input (addition is functional), and Cond A's AR accuracy on those output tokens does not collapse — so the divergence cannot be on the output positions.

The remaining candidates are the input-region tokens themselves, and the inter-sequence transitions present in the concatenated training stream. With `block_size = 64` and average sequence length ~10 characters, each training block contains ~6–8 complete equations end-to-end. The unmasked objective asks the model to predict, for example, the first operand of equation *k+1* from the trailing newline of equation *k*. Such transitions are statistical noise — the next equation's operands are drawn independently — so the only way for the model to drive that loss down is to memorize specific (training-stream-position → operand) correspondences. Those correspondences never recur at AR generation time, when the model is given only a single prompt with no surrounding training context.

The masked model cannot follow this path by construction: the input-region targets are `-1`, and the inter-sequence transition tokens are inputs of the next sequence. The gradient simply cannot push the model toward spurious cross-equation patterns. This is consistent with B's monotonic loss curve.

### 5.3 The Loss/Accuracy Paradox

A surprising sub-observation: Cond A's val loss reaches 2.7 — a number that, taken at face value, would normally suggest a model that has lost its ability to make sensible predictions — yet its AR accuracy holds at 80–85%, only ~6 points below the masked baseline.

The reconciliation is that the two metrics measure different things. Val loss is the average per-token negative log-likelihood across the entire sequence, dominated by the populous input/transition tokens once those positions begin to diverge. AR accuracy is computed only on the model's output digits, which are the minority of token positions and are not where the divergence is concentrated. The unmasked model has not forgotten how to add; it has additionally acquired a high-loss, high-confidence-but-wrong prediction habit on input tokens that the AR eval simply never asks about.

Two practical consequences follow:

1. In the unmasked setting, **val loss is not a reliable proxy for downstream AR accuracy**. Monitoring val loss alone gives a much more pessimistic picture than the model's actual capability.
2. The +6.4pp ceiling gap is the residual capacity that Cond A is spending on input-region patterns and is therefore not spending on output-token accuracy. Masking does not give the model new ability — it stops a leak.

---

## 6. Cross-Task Validation: Input Token Fraction

A natural follow-up question: is the +6.4pp gap a property of "masking" in the abstract, or does it depend on how much of each sequence is input?

To test this, the same A/B comparison was repeated on a scratchpad-format variant of 2-digit addition, where the output is a bracket-by-bracket carry chain rather than a bare answer:

```
Plain:      input = "12+34=",   output = "46"
Scratchpad: input = "12+34=",   output = "[2+4=6,C0][1+3=4,C0]46"
```

The scratchpad output is ~3–5× longer than the plain output, so the input fraction drops from ~70% to ~22% of each sequence. Conditions C (no mask) and D (target mask) replicate the A/B design otherwise (3 seeds each, `block_size=128` to fit the longer outputs).

The result:

| Comparison | Final Accuracy Gap (masked − unmasked) |
|---|---|
| A vs B (plain, ~70% input) | **+6.4pp** (83.5% → 89.9%) |
| C vs D (scratchpad, ~22% input) | **−0.8pp** (86.8% → 86.0%) |

The masking benefit collapses to zero — slightly negative, within noise — when the input fraction drops to ~22%. The val-loss diagnostic agrees: Cond C's val loss does diverge after its iter-3,500 minimum, but only to 0.44–0.65, far less catastrophic than Cond A's 2.1–3.0. There is less input to overfit on, so the divergence is smaller, so the ceiling gap is smaller.

A secondary contributor is **block density**. With plain sequences averaging ~10 characters and `block_size=64`, each block contains ~6–8 complete equations and therefore ~5–7 inter-equation boundaries. With scratchpad sequences averaging ~32 characters and `block_size=128`, each block contains ~4 equations and ~3 boundaries. Fewer boundaries mean fewer noisy transitions for the unmasked model to memorize, attenuating the same overfitting mechanism.

The two factors point in the same direction and yield a single empirical rule:

> The target-masking benefit scales with the fraction of each sequence that is input, not with the absolute output length.

This rule is consistent with the qualitative observation that instruction-tuning recipes for production LLMs already mask prompts in the loss as a default — at instruction-tuning input/output ratios, the effect this report quantifies on a tiny model would be substantial.

---

## 7. Discussion and Limitations

The findings should be read with the following scope:

- **Model scale.** A 6-layer, 128-dim Transformer is small enough that overfitting pressure on a 9,000-sample dataset arrives quickly. Whether the same input-fraction scaling holds at larger scales is an open question; the conventional wisdom that prompt-masking matters most at small/medium scales is consistent with the direction of effect here.
- **Synthetic task.** Exhaustive 2-digit addition is a degenerate case — the input/output relation is exactly determined and the input tokens contain almost no information about *other* inputs. Real instruction data has richer input distributions, which may either amplify or attenuate the input-overfitting mechanism.
- **Three seeds.** The +6.4pp gap and the val-loss divergence are robust across the three seeds tested (Cond B's std is 0.2pp), but a more thorough sweep would tighten the variance bounds.

What the report contributes:

1. A clean, controlled refutation of the "target masking accelerates training" intuition.
2. A mechanistic explanation grounded in val-loss divergence and supported by a token-position decomposition argument.
3. A scaling rule (gap ∝ input fraction) supported by an A/B/C/D cross-task experiment that holds the model class and optimizer fixed.

The natural follow-up — a controlled manipulation of input fraction at fixed task difficulty — was attempted in a subsequent experiment but fell to a ceiling effect from a [10, 99] digit-range choice that made the unmasked baseline trivially solve the task. The input-fraction scaling rule therefore stands as observed but not yet causally isolated.

---

## 8. Conclusion

Target masking on a tiny Transformer trained on 2-digit addition does not speed up convergence. Both masked and unmasked models reach 80% accuracy in roughly the same number of iterations, and their early-training accuracy curves are visually indistinguishable. What target masking does instead is raise the long-run accuracy ceiling — from 83.5% to 89.9% on plain addition, a +6.4pp gain at ~8% wall-clock overhead — by preventing the unmasked model from spending capacity on val-loss-divergent input-region patterns that AR generation never exercises. Cross-task replication on scratchpad-format addition shows the gap collapses to zero when the input fraction drops from ~70% to ~22%, indicating that the magnitude of the effect is governed by how much of each training sequence is input, not by absolute output length.

The shorter version: target masking does not give the model new ability; it stops a leak.

---

## Appendix A. Per-Seed Final Accuracies

**Plain addition (A vs B):**

| Seed | Cond A (no mask) | Cond B (target mask) |
|---|---|---|
| 1337 | 83.9% | 89.8% |
| 1338 | 82.5% | 90.2% |
| 1339 | 84.2% | 89.8% |
| **Mean ± std** | **83.5 ± 0.9%** | **89.9 ± 0.2%** |

**Scratchpad addition (C vs D):**

| Seed | Cond C (no mask) | Cond D (target mask) |
|---|---|---|
| 1337 | 86.3% | 86.7% |
| 1338 | 87.7% | 84.8% |
| 1339 | 86.4% | 86.6% |
| **Mean ± std** | **86.8 ± 0.8%** | **86.0 ± 1.1%** |

## Appendix B. Figures (to be inserted)

1. AR accuracy convergence — Cond A vs Cond B, mean ± std over 3 seeds, with horizontal threshold lines at 80/85/90%.
2. Val loss curves — Cond A vs Cond B, mean ± std over 3 seeds; mark Cond A's iter-3,000 minimum.
3. Cross-task gap — bar chart of final accuracy for A/B (plain, ~70% input) vs C/D (scratchpad, ~22% input), with input-fraction annotations.
