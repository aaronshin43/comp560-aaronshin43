# Experiment 1 — Addition

**Directory:** `addition/`
**Goal:** Verify that a tiny character-level Transformer can learn single-digit and two-digit addition through rote memorization of prompt-response pairs.

This was the first experiment and established the "short input → short output" framework used by all subsequent experiments.

---

## Dataset

| Variant | Samples | Format | Range |
|---|---|---|---|
| `basic` (1-digit) | 100 (all combinations) | `a+b=c\n` | 0–9 |
| `intermediate` (2-digit) | 10,000 (all combinations) | `a+b=c\n` | 0–99 |

Data is prepared by per-dataset `prepare.py` scripts (one inside each `data/<name>/` subdirectory). They produce `train.bin`, `val.bin`, and `meta.pkl`. No JSONL splits are saved — this predates the framework's standardized pipeline.

---

## Config

### `config/basic.py` (1-digit)

```
n_layer=3, n_head=3, n_embd=120, block_size=64
batch_size=32, max_iters=7000, lr_decay_iters=7000
learning_rate=1e-3, min_lr=1e-4, warmup_iters=0
dataset='basic', out_dir='out/basic'
```

### `config/intermediate.py` (2-digit)

```
n_layer=4, n_head=4(?), n_embd=128, block_size=64
max_iters=5000
dataset='intermediate', out_dir='out/intermediate'
```

---

## Key Commands

Run from `addition/`:

```bash
# Data preparation (run inside data/<name>/ directory)
python data/basic/prepare.py

# Training
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/basic.py

# Sampling
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/basic.py

# Evaluation (accuracy test)
python evaluate.py                           # intermediate (default)
python evaluate.py out_dir=out/basic dataset=basic
```

### `evaluate.py`

A standalone script that predates `eval_generation.py`. It:
- Adds `comp560-nanoGPT` to `sys.path` directly
- Randomly generates `num_samples` test cases
- Prompts the model with `"a+b="`, calls `model.generate()` stopping at `\n`
- Reports exact-match accuracy
- Supports config overrides as `key=value` positional args (no `--` prefix)

---

## Experiment Log

### Basic (1-digit) — 3 phases

**Phase 1 — 90/10 split (failed):**
- 2000 iters: val loss ~3.0 (severe overfitting — val set contained unseen arithmetic pairs)
- 5000 iters: val loss ~3.5 (more iterations made it worse)
- Insight: random split is wrong for exhaustive datasets; the 10 held-out pairs are never seen during training

**Phase 2 — No split (full dataset for train and val):**
- 2000 iters: val loss ~0.06 — memorization worked
- 5000 iters: stable at ~0.06, but specific errors persisted (e.g., `3+5=9`) due to local minima or insufficient batch size

**Phase 3 — No split, increased batch size:**
- 7000 iters: **99% accuracy**
- Larger batch size + more iterations eliminated persistent errors

### Intermediate (2-digit)

- Settings: 5000 iters, `n_layer=4`, `n_embd=128`
- Results: train/val loss ~1.01; **high accuracy** despite loss ~1.0
- Insight: loss ~1.0 (perplexity ~2.7) does not preclude functional accuracy — argmax consistently picks correct digits in structured tasks
- Continuous generation: model generates valid subsequent equations (e.g., `91+20=111`, `67+7=74`), suggesting it learned data structure perfectly

**Out-of-distribution test (3-digit inputs):**
- Initial manual tests appeared promising (`100+1=101` correct)
- Rigorous evaluation (N=100, range_max=1000): **~0% accuracy**
- Confirmed: model memorized the 0–99 domain; cannot extrapolate carry algorithm to unseen digit lengths

---

## Key Findings

1. **Train/val split is wrong for exhaustive arithmetic datasets.** Using the full dataset for both train and val (memorization) is correct when the goal is to verify rote learning of a fixed table.
2. **Loss ~1.0 ≠ failure.** Functional accuracy can be high even with moderate cross-entropy loss in structured tasks.
3. **No generalization beyond training domain.** The model learned input-output mappings, not the addition algorithm. 3-digit inputs produce ~0% accuracy.
4. **Framework validated.** The `Prompt=Response\n` format with nanoGPT works reliably for arithmetic tasks.
