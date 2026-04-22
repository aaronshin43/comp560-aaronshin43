# CLAUDE.md — comp560-aaronshin43

Aaron Shin's COMP560 (Dickinson, Spring 2026) research project. Experiments using small Transformer models and related training engines.

---

## Repo Layout

This project spans **multiple sibling repos** checked out at the same level:

```
d:\03_Coding\
├── comp560-nanoGPT/       ← shared training/eval engine (nanoGPT fork)
├── autoresearch/          ← shared LM benchmarking engine (Karpathy's autoresearch)
└── comp560-aaronshin43/   ← this repo: plans, data, configs, experiment docs/results
```

Engine repos (`comp560-nanoGPT/`, `autoresearch/`) remain as sibling checkouts and are never copied into experiment folders. Experiment folders in this repo store plans, results, plots, and reports only.

Each experiment lives in its own subdirectory inside `comp560-aaronshin43/`:

| Directory | Experiment | Notes |
|---|---|---|
| `addition/` | Exp 1 — Addition (basic / intermediate) | Earliest; uses custom `evaluate.py`, raw `.bin` data prep |
| `morse-code/` | Exp 2 — English → Morse translation | Character-level associative learning |
| `framework/` | Exp 3 — Short I/O framework | Introduces JSONL-based `prepare.py`; no JSONL eval splits |
| `validation/` | Exp 4 — Target masking study | Extended `prepare.py` saves `.jsonl` splits for TF/AR eval |
| `addition_scratchpad/` | Exp 5 — Scratchpad length generalization | 4-phase curriculum; uses `train_benchmark.py` + `eval_scratchpad.py` |
| `masking_benchmark/` | Exp 6 — Target masking benchmark | Convergence curve study (A/B plain + C/D scratchpad); post-hoc AR eval on named snapshots |
| `masking_study/` | Exp 7 — Masking input fraction study | Two-phase validation of Exp 6 hypothesis: digit-length extension (Phase 1) + input fraction manipulation (Phase 2) |
| `autoresearch-preln/` | Exp 8 — Pre-LN vs Post-LN reproduction | Uses sibling `autoresearch/` repo; docs/results live here |
| `autoresearch-scaling/` | Planned — Small-scale scaling law study | Uses sibling `autoresearch/` repo; docs/results live here |

---

## How Commands Work

### nanoGPT-based Experiments (Exp 1–7)

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

### autoresearch-based Experiments (Exp 8+)

For autoresearch experiments:

- run code from the sibling `autoresearch/` repo
- keep experiment docs, plans, and results in this repo under the experiment folder
- record the exact engine branch and commit in the experiment's `PLAN.md` and final report

When an experiment requires changes to the engine:

- create an experiment-specific branch in the engine repo (e.g. `exp/preln`)
- record engine repo path, branch, upstream commit, and changed files in `PLAN.md`

---

## Experiment Order & Progression

```
addition → morse-code → framework → validation → addition_scratchpad → masking_benchmark → masking_study
```

Each experiment built on the tooling of the previous one:
- **Exp 1–2**: raw binary data, manual config overrides, custom `evaluate.py`
- **Exp 3**: standardized JSONL → `.bin` pipeline (`framework/prepare.py`)
- **Exp 4**: adds `.jsonl` splits, `train_benchmark.py`, target masking
- **Exp 5**: adds scratchpad data format, `eval_scratchpad.py`, curriculum datasets
- **Exp 6**: adds named checkpoint snapshots in `train_benchmark.py`, `--ckpt_path` in `eval_generation.py`; multi-seed convergence curve study

The autoresearch experiments (Exp 8+) are a new branch of the project using a different sibling engine repo.

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
| [docs/experiments/masking_benchmark.md](docs/experiments/masking_benchmark.md) | Exp 6 detail |
| [docs/experiments/masking_study.md](docs/experiments/masking_study.md) | Exp 7 detail |
| [docs/activitylog.md](docs/activitylog.md) | Chronological log of all research sessions, posted to the team's Microsoft Teams channel |

## Activity Log

`docs/activitylog.md` is a running log of every research session — what was worked on, how long it took, and what was found. Entries are also posted to the research team's Microsoft Teams channel.

To add a new entry, use the `/activity-log` skill. It will ask for the date and activities, format the entry as proper markdown, and append it to the file.
