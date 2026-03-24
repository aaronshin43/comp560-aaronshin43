# CLAUDE.md — comp560-aaronshin43

Aaron Shin's COMP560 (Dickinson, Spring 2026) research project. Experiments teaching tiny Transformers (nanoGPT) to perform short-input / short-output tasks such as arithmetic and Morse code translation.

---

## Repo Layout

This project spans **two sibling repos** that must be checked out at the same level:

```
d:\03_Coding\
├── comp560-nanoGPT/     ← shared training/eval engine (nanoGPT fork)
└── comp560-aaronshin43/ ← this repo: data, configs, per-experiment scripts
```

Each experiment lives in its own subdirectory inside `comp560-aaronshin43/`:

| Directory | Experiment | Notes |
|---|---|---|
| `addition/` | Exp 1 — Addition (basic / intermediate) | Earliest; uses custom `evaluate.py`, raw `.bin` data prep |
| `morse-code/` | Exp 2 — English → Morse translation | Character-level associative learning |
| `framework/` | Exp 3 — Short I/O framework | Introduces JSONL-based `prepare.py`; no JSONL eval splits |
| `validation/` | Exp 4 — Target masking study | Extended `prepare.py` saves `.jsonl` splits for TF/AR eval |
| `addition_scratchpad/` | Exp 5 — Scratchpad length generalization | 4-phase curriculum; uses `train_benchmark.py` + `eval_scratchpad.py` |

---

## How Commands Work

All training/sampling commands follow this pattern (run from the **experiment directory**):

```bash
# Training
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/my_config.py

# Sampling
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/my_config.py --start="1+1="
```

- `NANOGPT_CONFIG` tells the engine where to find `configurator.py` so experiments can live outside the nanoGPT repo.
- `train_benchmark.py` is the extended trainer used from Exp 4 onward — adds target masking, Teacher Forcing (TF) exact-match eval, and optional early stopping.
- `eval_generation.py` runs standalone AR generation eval post-training.
- `eval_scratchpad.py` runs AR eval with answer extraction (used only in Exp 5).

---

## Experiment Order & Progression

```
addition → morse-code → framework → validation → addition_scratchpad
   ↑           ↑            ↑             ↑                ↑
 Exp 1       Exp 2        Exp 3         Exp 4            Exp 5
```

Each experiment built on the tooling of the previous one:
- **Exp 1–2**: raw binary data, manual config overrides, custom `evaluate.py`
- **Exp 3**: standardized JSONL → `.bin` pipeline (`framework/prepare.py`)
- **Exp 4**: adds `.jsonl` splits, `train_benchmark.py`, target masking
- **Exp 5**: adds scratchpad data format, `eval_scratchpad.py`, curriculum datasets

---

## Documentation Index

| Document | Contents |
|---|---|
| [docs/engine.md](docs/engine.md) | nanoGPT fork — model, training loop, config system, all eval scripts |
| [docs/experiments/addition.md](docs/experiments/addition.md) | Exp 1 detail |
| [docs/experiments/morse-code.md](docs/experiments/morse-code.md) | Exp 2 detail |
| [docs/experiments/framework.md](docs/experiments/framework.md) | Exp 3 detail |
| [docs/experiments/validation.md](docs/experiments/validation.md) | Exp 4 detail |
| [docs/experiments/addition_scratchpad.md](docs/experiments/addition_scratchpad.md) | Exp 5 detail |
