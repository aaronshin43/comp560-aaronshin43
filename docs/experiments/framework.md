# Experiment 3 — Short I/O Framework

**Directory:** `framework/`
**Goal:** Standardize the data preparation and training workflow for short-input / short-output tasks so future experiments can be set up with minimal boilerplate.

---

## What This Experiment Introduced

Prior experiments (Exp 1–2) had ad-hoc `prepare.py` scripts embedded inside each `data/` subdirectory. This experiment extracted that logic into a **reusable JSONL-based pipeline**:

1. **JSONL input protocol** — all datasets are `{"input": "...", "output": "..."}` per line
2. **Generic `prepare.py`** — reads any JSONL file, produces `train.bin`, `val.bin`, `meta.pkl`
3. **Dataset generator scripts** — e.g., `gen_addition.py` creates JSONL files programmatically

---

## Data Protocol

Input format (JSONL):
```json
{"input": "12+34", "output": "46"}
{"input": "5+5", "output": "10"}
```

Each sample is stored in the binary stream as:
```
{input}{sep}{output}{stop_token}
```
Default: separator `=`, stop token `\n` → produces `12+34=46\n`.

---

## Files

### `gen_addition.py`

Generates exhaustive addition datasets at 1, 2, or 3-digit levels. Creates JSONL files under `data/`:
- `data/addition_1digit/addition_1digit.jsonl` — 100 samples (0–9 × 0–9)
- `data/addition_2digit/addition_2digit.jsonl` — 10,000 samples (0–99 × 0–99)
- `data/addition_3digit/addition_3digit.jsonl` — 10,000 random samples (0–999)

### `prepare.py`

Arguments:
```
--file        Path to JSONL file (required)
--out_dir     Output directory (default: same directory as input file)
--sep         Separator character (default: '=')
--stop_token  End-of-sample token (default: '\n')
--test_size   Val split fraction (default: 0.1; set to 0.0 for full memorization)
--shuffle     Shuffle before splitting
```

Outputs: `train.bin`, `val.bin`, `meta.pkl`

> **Important limitation:** This version of `prepare.py` does NOT save `train.jsonl` / `val.jsonl` splits. This means it is **incompatible with `train_benchmark.py --enable_tf_eval=True`** and `eval_generation.py` without first generating those files separately. The `validation/prepare.py` version (Exp 4) fixes this.

### `config/addition_1digit.py`

```
n_layer=3, n_head=3, n_embd=120, block_size=64
batch_size=32, max_iters=2000
learning_rate=1e-3, warmup_iters=0
dataset='addition_1digit', separator="=", stop_token="\n"
```

### `config/addition_2digit.py`

```
n_layer=4, n_head=4, n_embd=128, block_size=128
batch_size=32, max_iters=5000, warmup_iters=100
learning_rate=1e-3
dataset='addition_2digit'
```

---

## Workflow (Quick Start)

Run from `framework/`:

```bash
# 1. Generate dataset
python gen_addition.py

# 2. Prepare (converts JSONL → binary)
python prepare.py --file data/addition_2digit/addition_2digit.jsonl

# 3. Train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/addition_2digit.py

# 4. Sample
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/addition_2digit.py

# 5. Sample with early stopping (Git Bash)
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/addition_2digit.py \
  --start="12+34" --stop_token=$'\n'
```

**Shell escaping for `\n`:**
- Git Bash / Linux: `--stop_token=$'\n'`
- PowerShell: `` --stop_token="`n" ``

---

## Key Findings

1. The JSONL → binary pipeline (`prepare.py`) generalizes to any short I/O task — just supply a different JSONL file.
2. The framework was validated on addition but is designed for arbitrary tasks (logic gates, Q&A, etc.).
3. **Gap identified:** `prepare.py` doesn't save `.jsonl` splits — the next experiment (`validation/`) fixed this to enable Teacher Forcing and AR evaluation.
