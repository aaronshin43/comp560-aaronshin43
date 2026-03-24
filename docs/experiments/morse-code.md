# Experiment 2 — Morse Code

**Directory:** `morse-code/`
**Goal:** Verify that a tiny Transformer can learn associative mapping between English words and their Morse code representations.

---

## Dataset

**Format:** `WORD: morse-sequence\n`

```
SOS: ... --- ...
CAT: -.-. .- -
HELLO: .... . .-.. .-.. ---
```

Data is prepared by `data/basic/prepare.py` (character-level). Produces `train.bin`, `val.bin`, `meta.pkl` in `data/basic/`.

The vocabulary is small: uppercase letters, Morse characters (`.`, `-`, ` `), `:`, `\n`.

---

## Config (`config/basic.py`)

```
n_layer=3, n_head=3, n_embd=120, block_size=64
batch_size=12, max_iters=2000, lr_decay_iters=2000
learning_rate=1e-3, min_lr=1e-4, warmup_iters=0
dataset='basic', out_dir='out/basic'
```

---

## Key Commands

Run from `morse-code/`:

```bash
# Data preparation
python data/basic/prepare.py

# Training
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/basic.py

# Sampling
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/basic.py \
  --num_samples=1 --max_new_tokens=100 --seed=2
```

---

## Experiment Log

### Run 1 — Underfitting (max_iters=200)

Dataset: HELP, SOS, AI, CAT, HELLO, WORLD (6 words)

Sample output showed clear underfitting:
- Misspelled words ("HELO", "HELLP", "CAI")
- Incorrect Morse patterns

### Run 2 — Premature LR decay bug (max_iters=2000, lr_decay_iters=200)

- `lr_decay_iters` was left at 200 while `max_iters` was increased to 2000
- LR dropped to minimum after 200 steps and stayed there for 1800 more steps
- Accuracy improved but inconsistent (e.g., "WORLD" had errors)
- **Lesson:** `lr_decay_iters` should equal `max_iters`

### Run 3 — Final (max_iters=2000, lr_decay_iters=2000, expanded vocabulary)

Added words: CODE, DOG, LOVE, SKY, STAR, SUN, SEA, HI, BYE (9 words added → 15 total)

Result: **5 samples (~1000 tokens) with 0 errors.** Perfect memorization.

```
HELLO: .... . .-.. .-.. ---
CAT: -.-. .- -
SKY: ... -.- -.--
BYE: -... -.-- .
CODE: -.-. --- -.. .
```

---

## Key Findings

1. **Tiny Transformers can learn associative lookup tables** (English word → Morse code) with sufficient training.
2. **`lr_decay_iters` must equal `max_iters`** — setting it too small causes the LR to collapse too early, leaving the model undertrained for most of the run.
3. This experiment confirmed the associative learning capacity established in Exp 1 generalizes to non-numeric mappings.
