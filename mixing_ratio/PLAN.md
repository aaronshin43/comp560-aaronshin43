# Experiment 7 — Mixing Ratio Ablation

**Question:** What is the optimal number of 3~4-digit curriculum samples (n) to maximize OOD generalization in scratchpad addition, given a fixed base of 10,100 exhaustive 1~2-digit samples?

---

## Background

Exp 5 Phase 4b discovered a non-monotonic relationship between curriculum data quantity and OOD accuracy:

| Variant | n (per digit) | Total | 3-digit OOD | 4-digit OOD |
|---|---|---|---|---|
| Min | 500 | 11,100 | 58.8% | 42.8% |
| Mid | 2,000 | 14,100 | 33.6% | 29.6% |
| Large | 5,000 | 20,100 | 98.2% | 95.8% |

Min (n=500) outperformed Mid (n=2,000) on all eval targets. This implies the accuracy curve is not monotonically increasing with n. Somewhere between n=500 and n=5,000, performance dips before recovering.

**Hypothesis:** There exists a mixing ratio sweet spot. At low n, the dominant 1~2-digit base keeps short-digit computation anchored while providing just enough depth signal. At medium n, the 3~4-digit samples begin to interfere with the well-learned short-digit patterns but do not yet provide enough coverage for reliable generalization. At high n, sheer coverage overcomes the interference.

**Goal:** Map the full n vs. OOD accuracy curve to identify the dip region, the recovery point, and the optimal ratio.

---

## Experimental Design

### Data Structure

Every condition uses the same base:
- **Base:** 1~2-digit exhaustive scratchpad pairs = 10,100 samples (100 1-digit + 10,000 2-digit)
- **Extension:** n random 3-digit pairs + n random 4-digit pairs

| n | Extension | Total | Base:Extension Ratio |
|---|---|---|---|
| 100 | 200 | 10,300 | 50.5:1 |
| 250 | 500 | 10,600 | 20.2:1 |
| 500 | 1,000 | 11,100 | 10.1:1 |
| 750 | 1,500 | 11,600 | 6.7:1 |
| 1,000 | 2,000 | 12,100 | 5.1:1 |
| 2,000 | 4,000 | 14,100 | 2.5:1 |
| 3,000 | 6,000 | 16,100 | 1.7:1 |
| 5,000 | 10,000 | 20,100 | 1.0:1 |

### Fixed Hyperparameters

Identical to Exp 5 Phase 4:
- `n_layer=6, n_head=4, n_embd=128`
- `block_size=128`
- `batch_size=128`
- `max_iters=10000`
- `learning_rate=3e-4, min_lr=3e-5, warmup_iters=400, lr_decay_iters=10000`
- `target_mask=True`
- `enable_tf_eval=True`

### Evaluation

For each condition, evaluate with `eval_scratchpad.py` on:

| Eval Set | Source | Purpose |
|---|---|---|
| Val (1~4-digit) | Train/val split | In-distribution generalization |
| Test 3-digit | `data/plain_3_4digit/3digit.jsonl` (3,000 samples) | Primary OOD metric |
| Test 4-digit | `data/plain_3_4digit/4digit.jsonl` (3,000 samples) | Primary OOD metric |
| Stretch 5-digit | `data/plain_5digit/5digit.jsonl` (3,000 samples) | Beyond-training-range probe |

### Replications

**Phase 1 — Shape mapping (1 seed):** Run all 8 conditions with `seed=1337` to map the curve shape. This identifies the dip region and recovery point.

**Phase 2 — Variance measurement (3 seeds):** For the 3 most interesting points on the curve (expected: the dip minimum, the recovery inflection, and one boundary), run 2 additional seeds (1338, 1339) to measure run-to-run variance.

Total runs: 8 (Phase 1) + 6 (Phase 2) = 14 runs.

---

## Implementation Plan

### Step 1 — Generate datasets

Create `gen_data.py` that generates all 8 dataset variants by calling `generate_combined_scratchpad` (reuse `build_scratchpad` from Exp 5) with varying n. Also generate OOD test sets (3-digit, 4-digit, 5-digit).

Output directories: `data/n0100/`, `data/n0250/`, ..., `data/n5000/`.

### Step 2 — Prepare binary files

Run `prepare.py` on each variant to produce `train.bin`, `val.bin`, `meta.pkl`, and JSONL splits.

### Step 3 — Create configs

One base config (`config/base.py`) with all shared hyperparameters. Override `dataset`, `out_dir`, `wandb_run_name` per condition via CLI.

### Step 4 — Run Phase 1 (8 conditions, 1 seed)

Train all 8 conditions. After each run, evaluate on val + 3 OOD test sets.

### Step 5 — Analyze Phase 1 results

Plot the n vs. OOD accuracy curve. Identify the dip region and select 3 points for Phase 2.

### Step 6 — Run Phase 2 (3 conditions, 2 additional seeds each)

Replicate selected conditions with seeds 1338 and 1339. Plot final curve with error bars.

### Step 7 — Report

Write `EXPERIMENT.md` with the full curve, analysis of the dip mechanism, and practical recommendations for mixing ratio selection.

---

## Expected Outcomes

| Result | Interpretation |
|---|---|
| Clear U-shaped dip between n=500 and n=2,000 | Confirms interference hypothesis; optimal ratio is identifiable |
| Monotonic increase above n=500 | Exp 5 Mid result was noise; n=2,000 just needed more iters |
| Dip extends beyond n=2,000 | Interference is more severe than expected; recovery requires very high n |
| Plateau at high n (diminishing returns above n=3,000) | Coverage saturates; more data beyond the plateau is wasteful |

The main deliverable is the **n vs. OOD accuracy curve** with the dip characterized (location, depth, width) and a practical recommendation for mixing ratio.

---

## Reusable Assets from Exp 5

- `addition_scratchpad/gen_addition.py` — `build_scratchpad()` function and `generate_combined_scratchpad()` can be imported or adapted
- `addition_scratchpad/prepare.py` — identical data prep pipeline
- `comp560-nanoGPT/eval_scratchpad.py` — AR eval with answer extraction
- OOD test data (`data/plain_3_4digit/`, `data/plain_5digit/`) can be reused directly or regenerated for consistency
