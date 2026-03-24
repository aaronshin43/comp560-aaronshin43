# Experiment 4 — Validation (Target Masking Study)

**Directory:** `validation/`
**Goal:** Validate the impact of **target masking** and **context window size (`block_size`)** on training efficiency and true autoregressive (AR) generation accuracy.

Full results: see [validation/EXPERIMENT.md](../../validation/EXPERIMENT.md)

---

## Setup

- **Dataset:** 2-digit addition, 10,000 samples, `{"input": "12+34", "output": "46"}`
- **Model:** `n_layer=6, n_head=4, n_embd=128`
- **Split:** 9,000 train / 1,000 val (9:1) or 8,000 / 2,000 (8:2 for Exp 3)
- **Training script:** `train_benchmark.py` (supports target masking + TF eval)
- **Eval metric:** Two types — Teacher Forcing (TF) exact-match and Autoregressive (AR) generation accuracy

### Evaluation Types

**Teacher Forcing (TF) exact-match:** feeds full sequence `input=output\n` in one forward pass; argmax predictions in the output region must match token-for-token. Run during training every `eval_interval` iters.

**Autoregressive (AR) generation:** prompts model with `input=` only; generation stops at `\n`; output string compared to ground truth. Run post-training via `eval_generation.py`.

---

## New in This Experiment

### Updated `prepare.py`

Unlike `framework/prepare.py`, this version also saves `train.jsonl` and `val.jsonl` alongside the `.bin` files. These are required by `train_benchmark.py --enable_tf_eval=True` and `eval_generation.py`.

Also fixes the shuffle bug: in `framework/prepare.py`, shuffle only reordered `samples_str` but not the paired `dataset` list (so JSONL splits were misaligned). This version shuffles both together with `zip`.

### `config/addition_2digit.py`

```
n_layer=6, n_head=4, n_embd=128, block_size=64
batch_size=128, max_iters=5000, lr_decay_iters=5000
learning_rate=3e-4, min_lr=3e-5, warmup_iters=200
dataset='addition_2digit', stop_token="\n"
```

---

## Key Commands

Run from `validation/`:

```bash
# Prepare data (9:1 split)
python prepare.py --file data/addition_2digit/addition_2digit.jsonl --shuffle --test_size=0.1

# Train — no mask
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --enable_tf_eval=True --benchmark_target=both --tf_eval_max_samples=1000

# Train — with target mask
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --enable_tf_eval=True --benchmark_target=both --tf_eval_max_samples=1000 --target_mask=True

# AR evaluation post-training
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_generation.py config/addition_2digit.py \
  --benchmark_target=both --eval_max_samples=1000
```

---

## Results

### Experiment 1: `block_size=16` (Anomaly)

| Config | Train Loss | Val Loss | Train TF | Val TF | Train AR | Val AR |
|---|---|---|---|---|---|---|
| No Mask | 0.989 | 1.160 | 89.3% | 92.2% | 87.9% | 89.8% |
| Target Mask | 0.531 | 0.541 | 83.1% | 84.1% | **78.8%** | **79.1%** |

**Finding:** Target masking *hurt* AR accuracy here. Root cause: with `block_size=16`, many random windows begin mid-equation (e.g., starting at the `4` in `12+34=46\n`). The masking logic sees no preceding stop token and treats the whole window as "input phase", silencing actual output tokens. The model trains on corrupted supervision and generalizes worse.

### Experiment 2: `block_size=64` (Resolution)

| Config | Train Loss | Val Loss | Train TF | Val TF | Train AR | Val AR |
|---|---|---|---|---|---|---|
| No Mask | 0.989 | 1.160 | 88.4% | 89.6% | 73.4% | 76.7% |
| **Target Mask** | **0.123** | **0.125** | **88.9%** | **90.5%** | **87.2%** | **88.0%** |

**Finding:** With `block_size=64` (fits multiple complete equations), target masking dramatically improves AR accuracy (76.7% → 88.0%). Without masking, the model wastes capacity predicting arbitrary first-tokens of the next equation in the continuous stream; at AR time it has no such context and destabilizes.

### Experiment 3: 8:2 Split — Edge Case Verification

| Config | Train Loss | Val Loss | Train TF | Val TF | Train AR | Val AR |
|---|---|---|---|---|---|---|
| No Mask | 0.982 | 1.170 | 88.0% | 86.6% | 80.1% | 78.4% |
| Target Mask | 0.121 | 0.125 | 89.2% | 88.4% | 87.4% | 86.9% |

**Finding:** With a larger val set (2,000 vs 1,000), Val TF Match dropped below Train TF Match — the opposite of Exp 2. Confirmed the "Val > Train on exact match" paradox from Exp 2 was a statistical artifact: the small 9:1 val set randomly contained fewer carry-heavy edge cases. Doubling the val set equalized the edge case distribution.

---

## Key Findings

1. **Target masking is critical for instruction-following tasks.** It restricts optimization to output tokens only, preventing the model from wasting capacity on input patterns and transitional noise between equations.
2. **`block_size` must comfortably fit at least one complete sample.** Truncation corrupts the masking logic. Rule of thumb: `block_size` ≥ 2× max sample length.
3. **Cross-entropy loss and exact-match accuracy measure different things.** Low loss reflects average confidence; exact match requires per-token perfection. A model can have moderate loss but high accuracy (Exp 1/2 no-mask) or very low loss but lower accuracy (Exp 1 target mask with truncation bug).
4. **Val > Train on exact match is a dataset artifact, not a model property.** It disappears when both splits contain proportional edge cases.
