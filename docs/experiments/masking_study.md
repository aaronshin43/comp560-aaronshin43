# Experiment 7 — Masking Study: Input Fraction Hypothesis

**Directory:** `masking_study/`
**Status:** Complete
**Question:** Is input token fraction the causal variable behind the masking benefit gap between plain and scratchpad addition?

Full results: see [masking_study/EXPERIMENT.md](../../masking_study/EXPERIMENT.md)

---

## Setup

- **Background:** Exp 6 showed masking raises plain 2-digit accuracy by +6.4pp but has ~0pp effect on scratchpad 2-digit. Hypothesis: benefit scales with input token fraction (~75% plain vs ~19% scratchpad).
- **Phase 1 design:** Extend both formats to 3-digit and 4-digit; reuse Exp 6 A–D as 2-digit reference.
- **Dataset:** 30,000 samples per new variant, 90/10 train/val split.
- **Model:** `n_layer=6, n_head=4, n_embd=128`; `block_size=64` (plain), `block_size=128` (scratchpad).
- **Training:** `max_iters=20,000`, `eval_interval=500`, 5 seeds per condition (1337–1341).
- **Eval:** AR exact-match on 1,000-sample val set, post-hoc on 20 checkpoints per run.

### Conditions

| Condition | Format | Digits | Masking | Source |
|---|---|---|---|---|
| A | Plain | 2-digit | No | Exp 6 |
| B | Plain | 2-digit | Yes | Exp 6 |
| C | Scratchpad | 2-digit | No | Exp 6 |
| D | Scratchpad | 2-digit | Yes | Exp 6 |
| E | Plain | 3-digit | No | New |
| F | Plain | 3-digit | Yes | New |
| G | Plain | 4-digit | No | New |
| H | Plain | 4-digit | Yes | New |
| I | Scratchpad | 3-digit | No | New |
| J | Scratchpad | 3-digit | Yes | New |
| K | Scratchpad | 4-digit | No | New |
| L | Scratchpad | 4-digit | Yes | New |

---

## Results

### Peak AR Accuracy (mean across 5 seeds)

| Condition | Format | Digits | Mask | Peak Acc (mean±std) | First iter >80% |
|---|---|---|---|---|---|
| A | Plain | 2 | No | 83.5±0.7% | ~4,000 |
| B | Plain | 2 | Yes | 89.9±0.2% | ~4,667 |
| C | Scratchpad | 2 | No | 86.8±0.7% | ~3,000 |
| D | Scratchpad | 2 | Yes | 86.0±1.0% | ~3,000 |
| E | Plain | 3 | No | 98.1±1.1% | 5,200 |
| F | Plain | 3 | Yes | 90.9±20.0% | 6,750 |
| G | Plain | 4 | No | 98.2±1.0% | 5,400 |
| H | Plain | 4 | Yes | 99.1±0.6% | 8,400 |
| I | Scratchpad | 3 | No | 99.8±0.1% | 3,000 |
| J | Scratchpad | 3 | Yes | 99.9±0.1% | 3,000 |
| K | Scratchpad | 4 | No | 99.7±0.0% | 3,200 |
| L | Scratchpad | 4 | Yes | 99.7±0.2% | 3,200 |

### Masking Gaps

| Format | Digits | Gap (masked − unmasked) |
|---|---|---|
| Plain | 2 | +6.4pp (Exp 6) |
| Plain | 3 | −7.2pp* |
| Plain | 4 | +0.9pp |
| Scratchpad | 2 | −0.8pp (Exp 6) |
| Scratchpad | 3 | +0.1pp |
| Scratchpad | 4 | +0.0pp |

*Cond F mean distorted by one slow-converging seed (F_s4, 55.1% at iter 20k). Excluding F_s4, the adjusted 3-digit plain gap is approximately +1.1pp.

---

## Key Findings

- **Scratchpad gap stays at zero across digit lengths.** All 10 scratchpad-masked seeds (I/J and K/L) converge to 99.7–100.0% identically to unmasked, replicating the Exp 6 C≈D result at both 3-digit and 4-digit. Input fraction ~16–17% is too low for masking to have any effect.

- **Plain masking gap is inconclusive at 3/4-digit.** Both masked and unmasked conditions reach 98–99%, compressing any gap below the level that can be defended with a 1,000-sample eval subset and no statistical test. Cond F (3-digit masked) is additionally distorted by one outlier seed (F_s4) that peaked at 55.1% then declined.

- **Masking consistently slows convergence for plain addition.** Plain masked conditions take 1,550–3,000 more iters to reach 80% accuracy than their unmasked counterparts. Scratchpad conditions show no such delay.

- **Scratchpad converges faster than plain in all cases.** Scratchpad reaches >80% by iter 3,000–3,200; plain requires 5,200–8,400 iters. Chain-of-thought output structure helps optimization independent of masking.

- **Results are consistent with the hypothesis but do not establish it causally.** Phase 1 changes digit length, dataset size, training steps, and task difficulty simultaneously — it is a consistency check, not a causal test. The scratchpad replication is robust; the plain side is inconclusive.
