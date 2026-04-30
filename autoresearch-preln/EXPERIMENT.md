# Autoresearch Experiment A: Pre-LN vs Post-LN Reproduction

**Directory:** `autoresearch-preln/`
**Status:** In Progress

## Objective

Test, in the `autoresearch` fixed-budget setting, whether:

- Pre-LN is more stable than Post-LN
- Pre-LN depends less on warmup

This writeup summarizes the first two failed attempts on the Windows RTX fork before starting a third run.

## Engine Tracking

- Engine repo: `D:\03_Coding\autoresearch-win-rtx`
- Experiment branch: `exp/preln`
- Attempt 1 commit: `848259b`
- Attempt 2 commit: `5fd3877`

## Attempt 1: `exp1` Operational Failure

Artifacts:

- `D:\03_Coding\autoresearch-win-rtx\out\exp1\preln-vs-postln-reproduction-2026-04-28.md`
- `D:\03_Coding\autoresearch-win-rtx\out\exp1\results.tsv`

What happened:

- The 4-condition scaffold was added successfully.
- Smoke tests completed.
- All full runs were killed by the `>10 minute` wall-clock rule before final evaluation.
- No run produced final `val_bpb`.

Conclusion:

- This attempt failed operationally, not scientifically.
- It showed that the old protocol was incompatible with the laptop compatibility runtime path.

## Attempt 2: `exp2` Scientific Failure

Artifacts:

- `D:\03_Coding\autoresearch-win-rtx\out\exp2\preln-vs-postln-reproduction-2026-04-30.md`
- `D:\03_Coding\autoresearch-win-rtx\out\exp2\results.tsv`

Observed results:

| Condition | val_bpb |
|---|---:|
| pre, no warmup | 1.191507 |
| pre, warmup 0.1 | 1.316709 |
| post, no warmup | 1.152545 |
| post, warmup 0.1 | 1.288508 |

Why this attempt is not trustworthy:

1. Warmup was not a fair intervention.
   - The training loop only counts time after step 10 toward the 5-minute budget.
   - LR progress is computed from that delayed timer.
   - With `warmup_ratio = 0.1`, the first ~11 steps stayed at effectively zero LR.
   - In a 29-step run, this means warmup throws away a large fraction of the total budget.

2. Pre-LN vs Post-LN was confounded in the model definition.
   - `pre` mode applies extra normalization around the embedding/input and final stack output.
   - `post` mode does not.
   - This is not a pure residual-block norm-placement comparison.

3. The benchmark is very short on this hardware.
   - Only 29 optimizer steps fit in the fixed budget.
   - Single-run-per-condition results at this scale are too brittle for a strong claim.

Conclusion:

- `exp2` produced complete outputs, but the outputs are not strong evidence about the target claim.
- The main lesson is that the current warmup implementation and pre/post comparison are not clean enough.

## Current Assessment

The project has not yet produced a valid reproduction result.

The two failures are different:

- `exp1`: runtime protocol failure
- `exp2`: experimental validity failure

## Decision For `exp3`

The next attempt should run under `out/exp3/` and should behave more like real automated research:

- run the benchmark to completion
- evaluate the result
- review whether the result is scientifically valid
- if invalid, log why
- then launch a better follow-up experiment instead of stopping after one matrix

The key fixes to look for in `exp3` are:

- a fairer warmup definition
- a cleaner Pre-LN/Post-LN comparison
- explicit review of whether each completed result is actually interpretable
