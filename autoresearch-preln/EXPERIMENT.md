# Autoresearch Experiment A: Pre-LN vs Post-LN Reproduction

**Directory:** `autoresearch-preln/`
**Status:** Completed with limitation

## Objective

Test, in the `autoresearch` fixed-budget setting, whether:

- Pre-LN is more stable than Post-LN
- Pre-LN depends less on warmup

This writeup summarizes the three attempts run on the Windows RTX fork.

## Engine Tracking

- Engine repo: `D:\03_Coding\autoresearch-win-rtx`
- Experiment branch: `exp/preln`
- Attempt 1 commit: `848259b`
- Attempt 2 commit: `5fd3877`
- Attempt 3 commit: `8ed2c42`

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

## Attempt 3: `exp3` Automated Review Loop

Artifacts:

- `D:\03_Coding\autoresearch-win-rtx\out\exp3\preln-vs-postln-final-2026-04-30.md`
- `D:\03_Coding\autoresearch-win-rtx\out\exp3\results.tsv`
- `D:\03_Coding\autoresearch-win-rtx\out\exp3\review_log.md`
- `D:\03_Coding\autoresearch-win-rtx\out\exp3\research_log.md`

What changed in `exp3`:

1. The agent was instructed to run a review loop instead of stopping after one matrix.
2. Warmup accounting was corrected so actual optimizer-step time drove LR schedule progress.
3. The Pre-LN/Post-LN comparison was cleaned so stack-level input/final normalization was shared.
4. Follow-up batches were only allowed when a review identified a specific validity problem.

### Batch 1

- Completed corrected 4-condition matrix.
- Reviewer accepted the cleaned Pre-vs-Post comparison as weak evidence.
- Reviewer rejected the warmup claim because `warmup_ratio = 0.1` only covered about two optimizer steps on this machine.

### Batch 2

- Reran only warmup-on conditions with `warmup_ratio = 0.25`.
- Reviewer accepted that warmup duration was now meaningful.
- Reviewer still rejected the warmup conclusion because warmup remained entangled with the always-on `warmdown_ratio = 0.5`.

### Batch 3

- Added run-time `warmdown_ratio` control.
- Reran the full matrix with `warmdown_ratio = 0.0`.
- Reviewer accepted this as weak but legitimate evidence for norm-placement comparison.
- Reviewer still rejected it as valid evidence about warmup dependence.

Cleanest `exp3` matrix:

| Label | Norm mode | Warmup | Warmdown | val_bpb |
|---|---|---|---|---:|
| A3 | pre | 0.0 | 0.0 | 1.309152 |
| B3 | pre | 0.25 | 0.0 | 1.311382 |
| C3 | post | 0.0 | 0.0 | 1.251770 |
| D3 | post | 0.25 | 0.0 | 1.283440 |

What `exp3` established:

- On this repo state, hardware path, and seed `42`, `post` outperformed `pre` in matched 300-second runs.
- No immediate instability, NaN, or crash was observed for either mode.
- The original warmup-dependence claim still could not be tested cleanly.

Why the warmup claim still failed:

- Only about 18 optimizer steps fit inside the 300-second benchmark on this laptop compatibility runtime.
- Any warmup large enough to matter necessarily consumes a large fraction of the run.
- That makes warmup act partly like a reduced effective optimization budget rather than a clean stability probe.

Conclusion:

- `exp3` fixed the earlier implementation confounds.
- It still did not deliver a scientifically defensible answer to the full target claim.
- It did deliver a narrower, defensible observation: in this custom RMSNorm architecture, `post` beat `pre` in matched short runs.

## Final Assessment

The original reproduction target was not achieved.

The three attempts failed in different ways:

- `exp1`: runtime protocol failure
- `exp2`: experimental validity failure
- `exp3`: benchmark-regime limitation after major confounds were fixed

The strongest final conclusion worth keeping is narrow:

- in this repo's custom RMSNorm architecture, on the RTX 4060 Laptop compatibility path, at seed `42`, `post` outperformed `pre` in matched 300-second runs;
- neither mode showed immediate blow-up within the approximately 18 optimizer steps that fit in the benchmark.

The broader literature-style claim about Pre-LN stability and warmup dependence remains unresolved in this setup.
