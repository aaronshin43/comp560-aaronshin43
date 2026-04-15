# Experiment 7 — Masking Study: Input Fraction Hypothesis

**Directory:** `masking_study/`
**Status:** Complete
**Question:** Is input token fraction the causal variable behind the masking benefit gap between plain and scratchpad addition?

Full results: see [masking_study/EXPERIMENT.md](../../masking_study/EXPERIMENT.md)

---

## Setup

- **Background:** Exp 6 showed masking raises plain 2-digit accuracy by +6.4pp but has ~0pp effect on scratchpad 2-digit. Hypothesis: benefit scales with input token fraction (~75% plain vs ~19% scratchpad).
- **Phase 1 design:** Extend both formats to 3-digit and 4-digit; reuse Exp 6 A–D as 2-digit reference.
- **Phase 2 design:** Hold format constant (2-digit scratchpad) and artificially vary input fraction by repeating the input prefix 1–5x (conditions M–V).

---

## Phase 1

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

30,000 samples per new variant, 90/10 split. `max_iters=20,000`, 5 seeds per condition.

### Peak AR Accuracy

| Condition | Format | Digits | Mask | Peak Acc (mean±std) |
|---|---|---|---|---|
| A | Plain | 2 | No | 83.5±0.7% |
| B | Plain | 2 | Yes | 89.9±0.2% |
| C | Scratchpad | 2 | No | 86.8±0.7% |
| D | Scratchpad | 2 | Yes | 86.0±1.0% |
| E | Plain | 3 | No | 98.1±1.1% |
| F | Plain | 3 | Yes | 90.9±20.0%* |
| G | Plain | 4 | No | 98.2±1.0% |
| H | Plain | 4 | Yes | 99.1±0.6% |
| I | Scratchpad | 3 | No | 99.8±0.1% |
| J | Scratchpad | 3 | Yes | 99.9±0.1% |
| K | Scratchpad | 4 | No | 99.7±0.0% |
| L | Scratchpad | 4 | Yes | 99.7±0.2% |

*F_s4 peaked at 55.1% and declined to 49.8% without converging. The four converged F seeds averaged 99.2%.

### Masking Gaps

| Format | Digits | Gap (masked − unmasked) |
|---|---|---|
| Plain | 2 | +6.4pp (Exp 6) |
| Plain | 3 | inconclusive* |
| Plain | 4 | +0.9pp |
| Scratchpad | 2 | −0.8pp (Exp 6) |
| Scratchpad | 3 | +0.1pp |
| Scratchpad | 4 | +0.0pp |

*3-digit plain gap is distorted by the F_s4 outlier; treated as inconclusive.

### Phase 1 Key Findings

- **Scratchpad gap stays at zero across digit lengths.** All 10 scratchpad-masked seeds (I/J, K/L) converge to 99.7–100.0% identically to unmasked, replicating the Exp 6 C≈D result at 3-digit and 4-digit.
- **Plain masking gap is inconclusive at 3/4-digit.** Both masked and unmasked plain conditions reach 98–99%, compressing any gap below what can be defended with a 1,000-sample eval subset and no statistical test.
- **Masking consistently slows convergence for plain addition.** Masked conditions take 1,550–3,000 more iters to reach 80% than unmasked. Scratchpad shows no such delay.
- **Phase 1 is a consistency check, not a causal test.** Digit length, dataset size, training steps, and task difficulty all changed simultaneously.

---

## Phase 2

### Conditions

| Condition | Multiplier | Input fraction | Masking |
|---|---|---|---|
| M | 1x | ~19% | No |
| N | 1x | ~19% | Yes |
| O | 2x | ~32% | No |
| P | 2x | ~32% | Yes |
| Q | 3x | ~41% | No |
| R | 3x | ~41% | Yes |
| S | 4x | ~48% | No |
| T | 4x | ~48% | Yes |
| U | 5x | ~54% | No |
| V | 5x | ~54% | Yes |

8,100 exhaustive pairs per variant (range 10–99), 90/10 split. `max_iters=10,000`, 3 seeds per condition.

### Peak AR Accuracy

| Multiplier | No-mask peak | Masked peak | Gap |
|---|---|---|---|
| 1x | 99.17±0.12% | 99.40±0.00% | +0.23pp |
| 2x | 99.77±0.15% | 99.80±0.00% | +0.03pp |
| 3x | 99.53±0.06% | 99.80±0.00% | +0.27pp |
| 4x | 99.63±0.23% | 99.73±0.12% | +0.10pp |
| 5x | 99.80±0.00% | 99.87±0.06% | +0.07pp |

### Phase 2 Key Findings

- **Ceiling effect renders Phase 2 uninformative.** All 10 conditions converge to 99–100% with masking gaps of 0.03–0.27pp and no monotonic trend. There is no gap to explain.
- **The ceiling arose from a design flaw.** The [10, 99] digit range produces a structurally uniform task (always `"XX+YY"` input), making the task easier than Exp 6's 0–99 range. Both masked and unmasked conditions trivially solve it before input fraction can have any effect.
- **Phase 2 failed to test the hypothesis.** A valid causal test requires a baseline where masking makes a meaningful difference. Phase 2 never established one.

---

## Overall Conclusion

Exp 7 did not produce a valid causal test of the input fraction hypothesis. Phase 1 is a consistency check; Phase 2 had a design flaw that produced a ceiling effect. **The hypothesis remains open.**
