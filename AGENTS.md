# AGENTS.md - comp560-aaronshin43

Aaron Shin's COMP560 (Dickinson, Spring 2026) research project. Experiments using small Transformer models and related training engines.

---

## Repo Layout

This project may span multiple sibling repos checked out at the same level:

```text
d:\03_Coding\
- comp560-nanoGPT/      shared training/eval engine (nanoGPT fork)
- autoresearch/         shared LM benchmarking engine (Karpathy's autoresearch)
- comp560-aaronshin43/  this repo: plans, data, configs, experiment docs/results
```

### Engine Repo Policy

- `comp560-aaronshin43/` is the experiment repo
- `comp560-nanoGPT/` and `autoresearch/` are engine repos
- engine repos should remain sibling checkouts, not be copied into experiment folders
- experiment folders in this repo should store:
  - `PLAN.md`
  - `EXPERIMENT.md`
  - `COMMANDS.md`
  - plots/tables/results
  - the engine repo path, branch, and commit used

### How To Track Engine Changes

When an experiment requires engine changes:

- create an experiment-specific branch in the engine repo
  - examples: `exp/preln`, `exp/scaling`
- preferably use a separate git worktree per active experiment if multiple experiments are in flight
- record the following in the experiment's `PLAN.md` and final report:
  - engine repo path
  - engine branch
  - upstream/base commit
  - experiment commit
  - changed files

This keeps experiment docs/results in `comp560-aaronshin43/` while preserving a clean, auditable code history in the engine repo.

---

## Experiment Directories

Each experiment lives in its own subdirectory inside `comp560-aaronshin43/`:

| Directory | Experiment | Notes |
|---|---|---|
| `addition/` | Exp 1 - Addition (basic / intermediate) | Earliest; uses custom `evaluate.py`, raw `.bin` data prep |
| `morse-code/` | Exp 2 - English to Morse translation | Character-level associative learning |
| `framework/` | Exp 3 - Short I/O framework | Introduces JSONL-based `prepare.py`; no JSONL eval splits |
| `validation/` | Exp 4 - Target masking study | Extended `prepare.py` saves `.jsonl` splits for TF/AR eval |
| `addition_scratchpad/` | Exp 5 - Scratchpad length generalization | 4-phase curriculum; uses `train_benchmark.py` + `eval_scratchpad.py` |
| `masking_benchmark/` | Exp 6 - Target masking benchmark | Convergence curve study (A/B plain + C/D scratchpad); post-hoc AR eval on named snapshots |
| `masking_study/` | Exp 7 - Masking input fraction study | Two-phase validation of Exp 6 hypothesis |
| `autoresearch-preln/` | Planned - Pre-LN vs Post-LN reproduction | Uses sibling `autoresearch/` repo; docs/results live here |
| `autoresearch-scaling/` | Planned - Small-scale scaling law study | Uses sibling `autoresearch/` repo; docs/results live here |

---

## How Commands Work

### nanoGPT-based Experiments

All nanoGPT training/sampling commands follow this pattern and are run from the experiment directory:

```bash
# Training
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/my_config.py

# Sampling
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/my_config.py --start="1+1="
```

- `NANOGPT_CONFIG` tells the engine where to find `configurator.py`
- `train_benchmark.py` is the extended trainer used from Exp 4 onward
- `eval_generation.py` runs standalone AR generation eval post-training
- `eval_scratchpad.py` runs AR eval with answer extraction (used only in Exp 5)

### autoresearch-based Experiments

For autoresearch experiments:

- run code from the sibling `autoresearch/` repo
- keep experiment docs, plans, and results in this repo
- record the exact engine branch and commit in the experiment folder

---

## Experiment Order And Progression

```text
addition -> morse-code -> framework -> validation -> addition_scratchpad -> masking_benchmark -> masking_study
```

Each experiment built on the tooling of the previous one:

- Exp 1-2: raw binary data, manual config overrides, custom evaluation
- Exp 3: standardized JSONL to `.bin` pipeline
- Exp 4: adds `.jsonl` splits, `train_benchmark.py`, target masking
- Exp 5: adds scratchpad data format, `eval_scratchpad.py`, curriculum datasets
- Exp 6: adds named checkpoint snapshots and post-hoc convergence studies
- Exp 7: uses the same benchmark machinery for a larger masking study

The planned autoresearch experiments are a new branch of the project using a different sibling engine repo.

---

## Documentation Index

| Document | Contents |
|---|---|
| [docs/engine.md](docs/engine.md) | nanoGPT fork: model, training loop, config system, eval scripts |
| [docs/experiments/addition.md](docs/experiments/addition.md) | Exp 1 detail |
| [docs/experiments/morse-code.md](docs/experiments/morse-code.md) | Exp 2 detail |
| [docs/experiments/framework.md](docs/experiments/framework.md) | Exp 3 detail |
| [docs/experiments/validation.md](docs/experiments/validation.md) | Exp 4 detail |
| [docs/experiments/addition_scratchpad.md](docs/experiments/addition_scratchpad.md) | Exp 5 detail |
| [docs/experiments/masking_benchmark.md](docs/experiments/masking_benchmark.md) | Exp 6 detail |
| [docs/experiments/masking_study.md](docs/experiments/masking_study.md) | Exp 7 detail |
| [docs/activitylog.md](docs/activitylog.md) | Chronological log of research sessions |

---

## Activity Log

`docs/activitylog.md` is a running log of research sessions, what was worked on, how long it took, and what was found. Entries are also posted to the research team's Microsoft Teams channel.

To add a new entry, use the `/activity-log` skill. It will ask for the date and activities, format the entry as markdown, and append it to the file.
