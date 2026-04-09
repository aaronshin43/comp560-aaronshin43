# Exp 7 — Masking Study: Input Fraction Hypothesis

**Directory:** `masking_study/`
**Status:** Planning

---

## Background

Exp 6 found that target masking helps plain addition (+6.4pp) but barely helps scratchpad addition (~0pp). The proposed explanation:

> **Masking benefit scales with input token fraction** — the proportion of tokens in the sequence that are input (and would waste gradient without masking).

| Format | Input tokens | Total tokens | Input fraction | Masking gap |
|---|---|---|---|---|
| Plain 2-digit | ~6 | ~8 | ~75% | +6.4pp |
| Scratchpad 2-digit | ~6 | ~32 | ~19% | ~0pp |

This experiment validates the hypothesis with two independent tests.

---

## Research Questions

1. **Phase 1 (Digit Length Extension):** Does the masking gap stay high for plain addition and stay low for scratchpad as digit length increases? This confirms the pattern is not specific to 2-digit data.

2. **Phase 2 (Input Fraction Manipulation):** If we artificially increase the input fraction of scratchpad sequences (by repeating the input prefix), does the masking gap grow proportionally? This tests whether input fraction is the *causal* variable.

---

## Phase 1: Digit Length Extension

### Design

Reuse Exp 6 A/B (plain 2-digit) and C/D (scratchpad 2-digit) results. Add new conditions at 3-digit and 4-digit.

| Condition | Format | Digits | Masking | Reuse? |
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

### Dataset

- **Plain 3-digit:** random sample of 30,000 pairs (100–999 + 100–999), 90/10 train/val split
- **Plain 4-digit:** random sample of 30,000 pairs (1000–9999 + 1000–9999), 90/10 split
- **Scratchpad 3-digit:** random sample of 30,000 pairs, 90/10 split
- **Scratchpad 4-digit:** random sample of 30,000 pairs, 90/10 split

### Input Fraction by Condition

| Format | Digits | Approx. input tokens | Approx. total tokens | Input fraction |
|---|---|---|---|---|
| Plain | 2 | 6 | 8 | 75% |
| Plain | 3 | 8 | 11 | 73% |
| Plain | 4 | 10 | 14 | 71% |
| Scratchpad | 2 | 6 | 32 | 19% |
| Scratchpad | 3 | 8 | 46 | 17% |
| Scratchpad | 4 | 10 | 62 | 16% |

For plain addition, input fraction stays high (~70%+) across digit lengths.
For scratchpad, it decreases further as digits grow.

### Config

- `block_size`: 64 (plain), 128 (scratchpad)
- `max_iters`: 20,000
- `n_layer`: 6, `n_head`: 4, `n_embd`: 128 (same as Exp 6)
- Seeds: 5 per condition

### Primary Metric

AR exact-match accuracy on same-distribution test set at iter 20,000.
Masking gap = accuracy(mask) − accuracy(no mask).
Secondary metric: first iter where accuracy exceeds 80% (convergence speed).

### Expected Outcome

- Plain masking gap stays ~5–7pp at 3-digit and 4-digit (no change from Exp 6)
- Scratchpad masking gap stays near 0pp at 3-digit and 4-digit
- Confirms the pattern is not 2-digit specific

---

## Phase 2: Input Fraction Manipulation

### Design

Use 2-digit scratchpad but repeat the input equation N times before the output. This directly varies input fraction while holding output structure fixed.

| Condition | Input prefix | Example | Input fraction |
|---|---|---|---|
| 1x | Normal | `12+34=[...]46` | ~19% |
| 2x | Repeated ×2 | `12+34=12+34=[...]46` | ~32% |
| 3x | Repeated ×3 | `12+34=12+34=12+34=[...]46` | ~41% |
| 4x | Repeated ×4 | `12+34=...×4=[...]46` | ~48% |
| 5x | Repeated ×5 | `12+34=...×5=[...]46` | ~54% |

For each multiplier: train masked vs unmasked → 10 conditions total.

| Condition | Multiplier | Masking |
|---|---|---|
| M | 1x | No |
| N | 1x | Yes |
| O | 2x | No |
| P | 2x | Yes |
| Q | 3x | No |
| R | 3x | Yes |
| S | 4x | No |
| T | 4x | Yes |
| U | 5x | No |
| V | 5x | Yes |

### Dataset

- 10,000 samples per multiplier variant
- Same scratchpad output format as Exp 6
- The model never sees a "clean" version during training — the repeated prefix is always there

### Config

- `block_size`: 64 (all variants fit within 56 chars at 5x)
- `max_iters`: 10,000
- Seeds: 3 per condition

### Primary Metric

Plot masking gap (masked − unmasked accuracy) vs input fraction.
Expected: monotonically increasing curve.

### Expected Outcome

- At 1x (19%): masking gap ≈ 0pp (replicates Exp 6 C≈D)
- As multiplier increases: masking gap grows
- At 5x (54%): masking gap approaches plain-addition level
- This directly shows input fraction is causal, not just correlated

---

## Implementation Plan (Phase 2)

Phase 1 is complete. Remaining steps are for Phase 2 only.

### Step 1: Generate datasets
- Add repeated-prefix scratchpad generator to `gen_data.py` (multipliers 1–5)
- Run `prepare.py` on each variant → 5 datasets under `data/phase2_*`

### Step 2: Training
- Write configs for each multiplier under `config/`
- Add training commands to `COMMANDS.md`
- Use `train_benchmark.py` with `always_save_checkpoint=True`

### Step 3: Evaluation
- Run `eval_generation.py` on checkpoints per condition
- Collect results into `results/accuracy_phase2.csv`

### Step 4: Analysis
- Line plot of masking gap vs input fraction (M/N through U/V)
- Use `analyze-results` agent

### Step 5: Report
- Condense Phase 1 section in `EXPERIMENT.md`, add Phase 2 results
- Update `docs/experiments/masking_study.md`
- Use `write-docs` agent

---

## Reusable Assets

- `masking_benchmark/gen_scratchpad.py` — `build_scratchpad()` function
- `masking_benchmark/config/addition_2digit.py` — base config template
- `masking_benchmark/config/scratchpad_1_2digit.py` — scratchpad config template
- `comp560-nanoGPT/train_benchmark.py` — training with masking
- `comp560-nanoGPT/eval_generation.py` — AR eval
