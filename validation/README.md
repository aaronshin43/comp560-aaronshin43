# Validation: Target Masking & Context Window Experiments

This directory contains the experiment configuration and scripts for validating the impact of **target masking** and **context window size (`block_size`)** on training efficiency and true autoregressive (AR) generation performance.

For full experimental results, observations, and analysis, see [EXPERIMENT.md](EXPERIMENT.md).

---

## Overview

The experiments use a 2-digit arithmetic dataset (`input: "12+34"`, `output: "46"`) and evaluate two dimensions:

1. **Target Masking** — whether the cross-entropy loss is restricted to output tokens only (via `ignore_index=-1`), or computed over the full sequence.
2. **Context Window** — whether `block_size=16` (truncation-prone) or `block_size=64` (full-equation) is used.

Training is run via `train_benchmark.py`, a superset of the standard `train.py` that adds target masking, Teacher Forcing (TF) exact-match evaluation, and optional early stopping. AR generation accuracy is evaluated separately via `eval_generation.py`.

---

## Reproducing the Experiments

> **Note:** All commands should be run from the `validation/` directory.

### 0. Prerequisites

Ensure the dataset exists. If not, generate and prepare it first:

```bash
cd ../validation
python gen_addition.py
python prepare.py --file data/addition_2digit/addition_2digit.jsonl --shuffle --test_size=0.1
```

For the 8:2 split experiment (Experiment 3), use:

```bash
python prepare.py --file data/addition_2digit/addition_2digit.jsonl --shuffle --test_size=0.2
```

### 1. Teacher Forcing (TF) Evaluation — No Mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py --enable_tf_eval=True --benchmark_target="both" --tf_eval_max_samples=1000
```

### 2. Teacher Forcing (TF) Evaluation — With Target Mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py --enable_tf_eval=True --benchmark_target="both" --tf_eval_max_samples=1000 --target_mask=True
```

### 3. Autoregressive (AR) Generation Evaluation

Run after training is complete. Evaluates true AR accuracy by prompting the model with only the input portion and comparing the generated output token-by-token.

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_generation.py config/addition_2digit.py --benchmark_target="both" --eval_max_samples=1000
```

---

## Key Flags

| Flag | Description |
|---|---|
| `--enable_tf_eval=True` | Enables Teacher Forcing exact-match evaluation every `eval_interval` iterations |
| `--target_mask=True` | Applies target masking — loss computed on output tokens only |
| `--benchmark_target` | `"train"`, `"val"`, or `"both"` — which split(s) to evaluate |
| `--tf_eval_max_samples` | Max samples used for TF evaluation (default: all) |
| `--eval_max_samples` | Max samples used for AR evaluation (default: all) |

---

## Configuration

Experiment hyperparameters (model size, `block_size`, `learning_rate`, etc.) are defined in [`config/addition_2digit.py`](config/addition_2digit.py).