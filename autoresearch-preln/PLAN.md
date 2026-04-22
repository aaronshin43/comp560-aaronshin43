# Autoresearch Experiment A: Pre-LN vs Post-LN Reproduction

**Directory:** `autoresearch-preln/`
**Status:** Planning
**Priority:** High

---

## Motivation

This experiment adapts a well-known Transformer stability result to the `karpathy/autoresearch` setting:

> Pre-LayerNorm Transformers should train more stably than Post-LayerNorm Transformers, and should rely less on learning-rate warmup.

Goal: reproduce the **direction of effect** in a small language model under the fixed 5-minute `autoresearch` budget, not the original paper's exact scale or numbers.

Why this is a good first `autoresearch` experiment:

- famous result with a narrow claim
- clean two-variable intervention
- mostly a `train.py` change
- directly comparable conditions
- aligned with the repo's short fixed-budget benchmark style

---

## Reference Result

Primary reference:

- Xiong et al., **On Layer Normalization in the Transformer Architecture** (ICML 2020)

Claim to test at small scale:

- **Post-LN** training is less stable and depends more heavily on warmup
- **Pre-LN** training is more stable and can often train well with reduced or no warmup

---

## Research Question

In the `autoresearch` small-LM setup, does moving LayerNorm from **Post-LN** to **Pre-LN**:

1. improve training stability under the same 5-minute budget?
2. reduce dependence on learning-rate warmup?
3. improve final validation bits-per-byte (`val_bpb`)?

---

## Hypotheses

Under the same optimizer, model size, data, and wall-clock budget:

- **Pre-LN** will achieve lower or equal `val_bpb` than **Post-LN**
- **Pre-LN without warmup** will remain stable
- **Post-LN without warmup** will be the least stable condition and may underperform significantly
- if warmup helps both variants, it should help **Post-LN** more than **Pre-LN**

---

## Engine Tracking

This experiment uses the sibling engine repo:

```text
d:\03_Coding\
- comp560-aaronshin43\
- autoresearch\
```

Record the following before the main runs:

- Engine repo: `..\autoresearch`
- Experiment branch: `exp/preln`
- Worktree path if used:
- Upstream/base commit:
- Experiment commit:
- Changed files:

All code changes should live in the `autoresearch/` repo. This directory stores planning, commands, results, and writeup only.

---

## Design

This is a **2 x 2 factorial design**:

| Condition | Norm placement | Warmup | Purpose |
|---|---|---|---|
| A | Post-LN | On | baseline Transformer-style condition |
| B | Post-LN | Off | tests warmup dependence |
| C | Pre-LN | On | matched stability comparison |
| D | Pre-LN | Off | main test of reduced warmup dependence |

Keep fixed across all four conditions:

- dataset
- tokenizer
- training time budget
- parameter budget
- number of layers, heads, and embedding width
- batch size unless memory or throughput forces adjustment
- optimizer type
- learning-rate peak value
- evaluation procedure
- random seed set

Only vary:

- LayerNorm placement
- warmup schedule

---

## Implementation Scope

Expected code area to touch:

- `train.py`

Likely edits:

1. add a switch for **norm placement**
2. implement or expose **Pre-LN** and **Post-LN** residual block variants
3. add a switch for **warmup steps**
4. ensure logs clearly report:
   - condition name
   - norm type
   - warmup status
   - final `val_bpb`
   - any NaN/divergence event

Try to leave unchanged:

- `prepare.py`
- tokenizer/data prep code
- evaluation logic, unless logging needs a small extension

---

## Execution Plan

### Phase 0 - Read And Baseline

1. Read `README.md`, `program.md`, and `train.py` in `autoresearch`
2. identify the current residual block structure
3. determine whether the baseline is already closer to Pre-LN or Post-LN
4. run one untouched baseline training job
5. record:
   - wall-clock behavior
   - baseline `val_bpb`
   - tokens/sec if available
   - whether compile/startup overhead materially affects the 5-minute budget

### Phase 1 - Add Switches

Implement:

- `norm_mode = "pre"` or `"post"`
- `warmup_iters = 0` or baseline warmup count

The implementation should make it difficult to accidentally change any other variable between conditions.

### Phase 2 - Sanity Check

Run short smoke tests for all four conditions to confirm:

- no shape errors
- no NaNs at startup
- config switches actually change the intended behavior
- logging clearly identifies each condition

### Phase 3 - Main Benchmark

Run all four conditions across multiple seeds.

Minimum:

- 3 seeds per condition

Preferred:

- 5 seeds per condition

Total:

- 12 runs minimum
- 20 runs preferred

### Phase 4 - Analysis

Aggregate by condition:

- final `val_bpb`
- mean and std across seeds
- instability count
- best run
- worst run

Key comparisons:

- A vs C to isolate norm placement with warmup on
- B vs D to isolate norm placement with warmup off
- A vs B to measure warmup dependence in Post-LN
- C vs D to measure warmup dependence in Pre-LN

---

## Metrics

Primary metric:

- final validation bits per byte (`val_bpb`) at the end of the fixed 5-minute budget

Secondary metrics:

- training stability
- NaN/divergence incidence
- loss spikes during training
- throughput if exposed
- compile overhead if relevant

Treat a run as unstable if any of the following happen:

- NaN loss
- exploding loss that does not recover
- obviously broken optimization curve
- catastrophic final `val_bpb` far outside normal seed variance

---

## Expected Outcomes

Expected ranking:

- best: Pre-LN with warmup
- close second: Pre-LN without warmup
- weaker: Post-LN with warmup
- worst: Post-LN without warmup

If the effect is weaker than expected:

- Pre-LN and Post-LN may be similar with warmup on
- warmup removal may hurt Post-LN much more than Pre-LN

If no effect appears, likely explanations are:

- model too small
- training budget too short
- optimizer choice dominates norm-placement effects
- autoresearch defaults already stabilize training enough

---

## Risks And Confounds

- if the repo already uses Pre-LN, converting to Post-LN may require a careful rewrite
- optimizer defaults may shrink the visible gap
- a 5-minute budget may overemphasize early optimization rather than long-run convergence
- small-model effects may mute large-scale instability phenomena

---

## Success Criteria

This experiment is worth keeping if it shows at least one of the following:

- Post-LN without warmup is clearly less stable than Pre-LN without warmup
- Pre-LN consistently matches or beats Post-LN on final `val_bpb`
- warmup dependence is materially larger for Post-LN

Even a modest but clean directional result would be useful as a small-scale reproduction.

---

## Stretch Extensions

- RMSNorm vs LayerNorm
- warmup sweep instead of binary on/off
- depth sweep to test whether deeper small models amplify the effect
- optimizer interaction
- gradient norm logging to connect empirical behavior to the paper's theory

---

## Deliverables

- `PLAN.md`
- a patched `train.py` in `autoresearch`
- run table with all seeds and conditions
- summary plot of `val_bpb` by condition
- short experiment report explaining whether the classic Pre-LN result reproduced

---

## Decision

This is the recommended first `autoresearch` experiment because it has a narrow intervention, a famous reference result, a clean code diff, and interpretable failure modes.
