# Research Roadmap

Tracks the overall plan for the COMP560 research project — what has been completed and what comes next.

---

## Completed Experiments

| # | Experiment | Key Finding |
|---|---|---|
| 1 | **Addition** (`addition/`) | Transformer memorizes 1–2-digit addition (99%), fails completely OOD on 3-digit. No algorithm learned. |
| 2 | **Morse Code** (`morse-code/`) | Character-level associative mapping learned near-perfectly. Discovered `lr_decay_iters` bug in base nanoGPT. |
| 3 | **Short I/O Framework** (`framework/`) | Standardized JSONL pipeline for future experiments. `prepare.py` does not save `.jsonl` eval splits (fixed in Exp 4). |
| 4 | **Target Masking Validation** (`validation/`) | Target masking is critical for AR accuracy with adequate `block_size`. Without masking, model wastes capacity on input tokens and fails at AR time. |
| 5 | **Addition Scratchpad** (`addition_scratchpad/`) | Scratchpad teaches the carry algorithm; curriculum exposure solves bracket-depth generalization within range. Non-monotonic mixing effect discovered: Min (n=500) > Mid (n=2k). |

---

## Planned Experiments

### Exp 6 — Target Masking Benchmark

**Directory:** `masking_benchmark/`
**Status:** Planning

**Question:** Does target masking improve training efficiency — i.e., does it reach the same accuracy in fewer iterations and less wall-clock time?

Exp 4 established that masking is correct. This experiment quantifies the efficiency benefit by comparing convergence curves (accuracy vs. iteration) between masked and unmasked runs.

See [masking_benchmark/PLAN.md](masking_benchmark/PLAN.md) for full design.

---

### Exp 7 — TBD (decision after Exp 6)

Two candidate directions:

#### Option A — Mixing Ratio Ablation

**Question:** What is the optimal n-per-digit-length in Phase 4b curriculum training?

Phase 4b found a surprising non-monotonic effect: Min (n=500) outperformed Mid (n=2000) on OOD generalization. The optimal ratio between base data (1–2 digit, n=10,100) and longer-digit data is unknown. A systematic sweep n ∈ {100, 250, 500, 1000, 2000, 5000} would quantify the shape of this effect and potentially surface a principled mixing strategy.

**Scope:** Contained ablation on top of existing Phase 4 infrastructure. Low implementation cost.

#### Option B — Positional Encoding Swap

**Question:** Can relative positional encodings (RoPE or ALiBi) enable zero-shot length generalization that absolute learned embeddings cannot?

Phase 3 showed that the scratchpad model generates exactly 2 brackets regardless of input length — a direct symptom of absolute positional embeddings never seeing longer sequences. RoPE/ALiBi encode relative distances instead and may allow the bracket depth to scale with input length at inference time.

**Scope:** Requires modifying `comp560-nanoGPT/model.py`. Higher implementation cost, but the research question is more fundamental.

---

## Decision Criteria for Exp 7

After Exp 6, choose based on:

- If Exp 6 confirms a large efficiency gain → the mixing ratio ablation is a natural next step (same infrastructure, clean story about training budget).
- If Exp 6 raises questions about model capacity / generalization mechanism → the positional encoding swap addresses the deeper question.
- Both options can be pursued sequentially; they are not mutually exclusive.
