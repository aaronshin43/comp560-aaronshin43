# Experiment 5 — Addition Scratchpad (Length Generalization)

**Directory:** `addition_scratchpad/`
**Goal:** Investigate whether scratchpad-based chain-of-thought reasoning can enable a tiny Transformer to generalize the addition algorithm to digit lengths unseen during training.

Full results: see [addition_scratchpad/EXPERIMENT.md](../../addition_scratchpad/EXPERIMENT.md)

---

## Setup

- **Model:** `n_layer=6, n_head=4, n_embd=128` (identical across all phases for fair comparison)
- **Training data:** 1~2-digit addition (exhaustive: 100 + 10,000 = 10,100 samples), 90:10 shuffle split
- **OOD test data:** 3,000 random 3-digit samples (100–999), 3,000 random 4-digit samples (1000–9999)
- **Training flags:** `--target_mask=True --enable_tf_eval=True`
- **Eval metric:** AR generation — model prompted with `input=` only; final contiguous digit sequence extracted from output and compared to true answer

---

## Data Formats

**Plain** (used in Phase 1):
```json
{"input": "15+27", "output": "42"}
```

**Scratchpad** (used in Phase 2–4):
```json
{"input": "15+27", "output": "[5+7=12,C1][1+2+1=4,C0]42"}
```

Each bracket `[dA+dB(+carry)=sum,Cnew_carry]` encodes one digit position processed right-to-left. Overflow carry appended as `[1]`. The structure is identical regardless of operand length.

---

## Files

### `gen_addition.py`

Generates all datasets for all phases:

| Dataset | Contents | Phase |
|---|---|---|
| `data/plain_1_2digit/` | 10,100 plain 1~2-digit pairs | Phase 1 train |
| `data/scratchpad_1_2digit/` | 10,100 scratchpad 1~2-digit pairs | Phase 2 train |
| `data/plain_3_4digit/3digit.jsonl` | 3,000 random plain 3-digit pairs | OOD test (all phases) |
| `data/plain_3_4digit/4digit.jsonl` | 3,000 random plain 4-digit pairs | OOD test (all phases) |
| `data/scratchpad_1_4digit/` | 10,100 + 5,000 + 5,000 = 20,100 pairs | Phase 4a train |
| `data/scratchpad_1_4digit_min/` | 10,100 + 500 + 500 = 11,100 pairs | Phase 4b (Min) train |
| `data/scratchpad_1_4digit_mid/` | 10,100 + 2,000 + 2,000 = 14,100 pairs | Phase 4b (Mid) train |
| `data/plain_5digit/5digit.jsonl` | 3,000 random plain 5-digit pairs | Stretch OOD test (Phase 4a) |

### `prepare.py`

Same as `validation/prepare.py` — saves `train.bin`, `val.bin`, `meta.pkl`, `train.jsonl`, `val.jsonl`.

### `eval_scratchpad.py` (in `comp560-nanoGPT/`)

Unlike `eval_generation.py` (full exact match), this evaluator extracts only the final numeric answer. Supports both plain (`"42"`) and scratchpad (`"[...C0]42"`) output formats by finding the last digit sequence before `\n`.

---

## Configs

| Config | `dataset` | `block_size` | `max_iters` |
|---|---|---|---|
| `phase1_plain.py` | `plain_1_2digit` | 64 | 5,000 |
| `phase2_scratchpad.py` | `scratchpad_1_2digit` | 128 | 10,000 |
| `phase4_curriculum.py` | `scratchpad_1_4digit` | 128 | 10,000 |
| `phase4_min.py` | `scratchpad_1_4digit_min` | 128 | 10,000 |
| `phase4_mid.py` | `scratchpad_1_4digit_mid` | 128 | 10,000 |

Note: `phase2_scratchpad.py` uses `block_size=128` even though training data is 1~2-digit, because this checkpoint is reused at inference time for 3~4-digit OOD inputs (which can be ~70 chars in scratchpad format).

---

## Key Commands

Run from `addition_scratchpad/`:

```bash
# 0. Generate all datasets
python gen_addition.py

# ── Phase 1 ──────────────────────────────────────────────────────────
python prepare.py --file data/plain_1_2digit/plain_1_2digit.jsonl --shuffle --test_size=0.1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/phase1_plain.py \
  --target_mask=True --enable_tf_eval=True --benchmark_target=both

# Evaluate (in-dist val, OOD 3-digit, OOD 4-digit)
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py \
  --eval_file=data/plain_1_2digit/val.jsonl
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py \
  --eval_file=data/plain_3_4digit/3digit.jsonl

# ── Phase 2 ──────────────────────────────────────────────────────────
python prepare.py --file data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl --shuffle --test_size=0.1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/phase2_scratchpad.py \
  --target_mask=True --enable_tf_eval=True --benchmark_target=both

# ── Phase 3 (no training — reuse Phase 2 checkpoint) ─────────────────
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py \
  --eval_file=data/plain_3_4digit/3digit.jsonl

# ── Phase 4a ─────────────────────────────────────────────────────────
python prepare.py --file data/scratchpad_1_4digit/scratchpad_1_4digit.jsonl --shuffle --test_size=0.1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/phase4_curriculum.py \
  --target_mask=True --enable_tf_eval=True --benchmark_target=both

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py \
  --eval_file=data/plain_3_4digit/3digit.jsonl
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py \
  --eval_file=data/plain_5digit/5digit.jsonl
```

---

## Results

### Phase 1 — Baseline (Plain Model)

| Eval Target | Accuracy | Correct / Total |
|---|---|---|
| Val (1~2-digit, in-dist) | 82.9% | 837 / 1,010 |
| Test (3-digit, OOD) | 0.6% | 17 / 3,000 |
| Test (4-digit, OOD) | 0.0% | 0 / 3,000 |

Hypothesis confirmed: plain model fails completely outside its training digit range.

### Phase 2 — Scratchpad Training

| Eval Target | Accuracy | Correct / Total |
|---|---|---|
| Val (1~2-digit, in-dist) | 88.5% | 894 / 1,010 |

Model genuinely generates bracket-by-bracket carry chains (manually verified). It has learned the carry algorithm.

### Phase 3 — Zero-Shot OOD Test (Phase 2 checkpoint, no retraining)

| Eval Target | Phase 1 | Phase 3 |
|---|---|---|
| Test (3-digit, OOD) | 0.6% | **0.0%** |
| Test (4-digit, OOD) | 0.0% | **0.0%** |

Failure mode: model generates exactly **2 brackets** regardless of input length (bracket depth fixed at the training maximum). The carry arithmetic is correct within those 2 steps, but the depth decision never generalizes.

### Phase 4a — Curriculum Learning (n=5,000 per digit length)

| Eval Target | Accuracy | Correct / Total |
|---|---|---|
| Val (1~4-digit, in-dist) | 91.5% | 1,840 / 2,010 |
| Test (3-digit) | **98.2%** | 2,947 / 3,000 |
| Test (4-digit) | **95.8%** | 2,874 / 3,000 |
| Stretch (5-digit, OOD) | 0.0% | 0 / 3,000 |

Model learned that bracket count = f(operand length). 5-digit stretch: model attempts 5 brackets (correct depth) but arithmetic fails at unseen digit positions.

### Phase 4b — Minimum Data Ablation

| Eval Target | Phase 4a (n=5k) | Min (n=500) | Mid (n=2k) |
|---|---|---|---|
| Val (1~4-digit) | 91.5% | 83.2% | 67.3% |
| Test (3-digit) | 98.2% | 58.8% | **33.6%** |
| Test (4-digit) | 95.8% | 42.8% | **29.6%** |

Surprising finding: **Min (n=500) outperforms Mid (n=2k)** on all targets (replicated on fresh datasets).

Likely explanation: with n=500, the 1~2-digit base (10,100 samples) outnumbers 3~4-digit samples 10:1 — keeping short-digit computation well-anchored while providing just enough depth-variation signal. At n=2,000 the ratio drops to ~2.5:1, and the additional longer-digit samples disrupt the well-learned short-digit computation without providing enough coverage for reliable generalization.

---

## Key Findings

| Phase | Key Finding |
|---|---|
| Phase 1 | Plain model memorizes; fails completely OOD — no algorithm learned |
| Phase 2 | Scratchpad teaches the carry algorithm; model genuinely computes bracket chains |
| Phase 3 | Algorithm learned but **depth** is fixed at training maximum — zero-shot length gen fails |
| Phase 4a | Including 3~4-digit examples immediately solves depth generalization within range |
| Phase 4b | Non-monotonic mixing effect: Min (n=500) > Mid (n=2k); optimal ratio between 500 and 5,000 per digit length |

**Core insight:** Scratchpad training teaches the algorithm but not when to apply it to different-depth inputs. The fix is straightforward — expose the model to examples at the target depth. Length generalization remains bounded by training distribution; the bottleneck shifts from structural depth (fixed in Phase 3) to arithmetic accuracy at unseen digit positions (Phase 4a stretch).
