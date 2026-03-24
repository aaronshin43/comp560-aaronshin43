# Experiment 6 — Target Masking Benchmark

**Question:** Does target masking improve training efficiency — i.e., does it reach the same accuracy in fewer iterations?

---

## Background

Exp 4 (Validation) established that target masking is **necessary for accuracy** with adequate `block_size`. The key result:

| Config | Val AR Accuracy |
|---|---|
| No mask, block_size=64 | 76.7% |
| Target mask, block_size=64 | **88.0%** |

What Exp 4 did **not** measure:
- How quickly each condition converges (accuracy vs. iteration curve)
- Whether the unmasked model eventually catches up given more iterations

**Hypothesis:** Target masking converges faster because the gradient signal is concentrated on output tokens only. Without masking, each gradient step updates the model on input tokens and separator tokens — noise relative to the instruction-following objective. The masked model should reach any given accuracy threshold in fewer iterations.

**Secondary hypothesis:** The efficiency gap is larger for tasks with longer inputs relative to outputs, since more tokens are masked and the signal-to-noise ratio improvement is greater.

---

## Experimental Design

### Task

2-digit addition — same as Exp 4. Format: `{"input": "12+34", "output": "46"}`.

Rationale: results are directly comparable to Exp 4; task is fast to train; output is short relative to input, making masking effect non-trivial.

### Conditions

| Condition | `target_mask` | Notes |
|---|---|---|
| A — No mask | False | Baseline; matches Exp 4 Exp 2 no-mask run |
| B — Target mask | True | Matches Exp 4 Exp 2 mask run |

Both conditions use identical hyperparameters and the same prepared data files. Run each condition **3 times with different seeds** to measure variance.

### Fixed Hyperparameters

- `n_layer=6, n_head=4, n_embd=128` (same as Exp 4)
- `block_size=64` (confirmed good in Exp 4 Exp 2)
- `batch_size=128`
- `max_iters=10000` — doubled from Exp 4's 5000 to give unmasked condition room to converge
- `learning_rate=3e-4, min_lr=3e-5, warmup_iters=200, lr_decay_iters=10000`
- `eval_interval=500` — controls loss logging frequency and checkpoint save frequency; no TF eval overhead
- `enable_tf_eval=False` — disabled; convergence curves come from post-hoc AR eval on saved checkpoints (see below)

### Metrics

**Primary:**
- Val AR accuracy at each checkpoint (convergence curve — accuracy vs. iterations), collected post-hoc
- Iterations to reach 80%, 85%, 90% Val AR accuracy (if reached)

**Secondary:**
- Val loss curve (logged during training, zero overhead)
- Total end-to-end training time (single wall-clock measurement per run)

> **Why not TF eval during training?** TF eval iterates over ~1000 samples every `eval_interval`, pausing GPU computation for 10–20 seconds each time. This creates a regular sawtooth pattern in W&B GPU metrics and inflates end-to-end training time. Since the ground-truth metric for this task is AR generation accuracy (not TF exact-match), it is cleaner to collect accuracy post-hoc on saved checkpoints.

### Convergence Curves via Post-Hoc Eval

**Required changes to the engine (minor):**

1. `train_benchmark.py` — add one line to the checkpoint save block to write a named snapshot alongside `ckpt.pt`:
   ```python
   torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num:05d}.pt'))
   ```
   This is additive — existing behavior is unchanged.

2. `eval_generation.py` — add an optional `--ckpt_path` argument (~5 lines) so the script can load `ckpt_01000.pt`, `ckpt_02000.pt`, etc. instead of always defaulting to `ckpt.pt`.

After training, run `eval_generation.py` against each saved snapshot in iteration order to produce the accuracy-vs-iterations curve. Storage cost: ~4 MB per checkpoint × 10 snapshots × 6 runs ≈ 240 MB total.

### Stretch Condition (optional)

If the primary result is clear, add:

| Condition | Format | `target_mask` |
|---|---|---|
| C — Scratchpad, no mask | scratchpad | False |
| D — Scratchpad, target mask | scratchpad | True |

Scratchpad outputs are ~5× longer than plain outputs. If the secondary hypothesis holds, the masking efficiency gain should be larger here.

---

## Implementation Plan

### Step 1 — Sanity check

Reuse Exp 4's prepared data (`validation/data/addition_2digit/`). Run one unmasked and one masked training run for 2000 iterations and verify TF accuracy curves look reasonable before committing to full runs.

### Step 2 — Patch the engine

Make the two small changes described above:
- `train_benchmark.py`: save named snapshots alongside `ckpt.pt`
- `eval_generation.py`: add optional `--ckpt_path` argument

### Step 3 — Create config

Create `masking_benchmark/config/addition_2digit.py` based on `validation/config/addition_2digit.py` with:
- `max_iters=10000`
- `lr_decay_iters=10000`
- `eval_interval=500`
- `enable_tf_eval=False`
- `out_dir` pointing to `masking_benchmark/out/`

### Step 4 — Run main conditions (A and B)

Run each condition 3 times, varying only `--seed`. Log runs to W&B (loss curves only — no TF eval pulses).

### Step 5 — Post-hoc AR eval

For each run, iterate over saved snapshots (`ckpt_00500.pt` through `ckpt_10000.pt`) and run `eval_generation.py --ckpt_path=...` to collect AR accuracy at each iteration. This produces the convergence curve.

### Step 6 — (Optional) Stretch conditions C and D

Prepare scratchpad data from `addition_scratchpad/data/scratchpad_1_2digit/`. Create a second config with `block_size=128` (required for scratchpad length) and repeat.

---

## Expected Outcomes

| Result | Interpretation |
|---|---|
| Masked converges faster (fewer iters to threshold) | Hypothesis confirmed: masking improves gradient efficiency |
| Same convergence speed, higher final accuracy | Masking improves signal quality but not speed — different story |
| Unmasked catches up given enough iters | Masking is a correctness fix, not an efficiency gain |
| Larger gap on scratchpad than plain | Efficiency benefit scales with output-to-input ratio |

The main deliverable is a convergence curve plot (Val TF accuracy vs. iterations) for both conditions, with error bands across seeds.

---

## Open Questions Before Starting

1. Confirm that `validation/data/addition_2digit/` is already prepared and ready to reuse.
2. Does `eval_generation.py` currently accept any path overrides, or is `out_dir/ckpt.pt` fully hardcoded? (Determines how much work Step 2 is.)
