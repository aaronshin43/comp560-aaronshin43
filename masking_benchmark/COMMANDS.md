# Masking Benchmark — Commands

Run all commands from `masking_benchmark/`.

---

## Step 1 — Data Preparation

```bash
python gen_addition.py

python prepare.py --file data/addition_2digit/addition_2digit.jsonl \
    --out_dir data/addition_2digit --shuffle
```

---

## Step 2 — Training

### Condition A — No mask (baseline)

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_A_s1 --wandb_run_name=cond_A_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_A_s2 --wandb_run_name=cond_A_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_A_s3 --wandb_run_name=cond_A_s3
```

### Condition B — Target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_B_s1 --wandb_run_name=cond_B_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_B_s2 --wandb_run_name=cond_B_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_B_s3 --wandb_run_name=cond_B_s3
```

---

## Step 3 — Post-hoc AR Eval (Step 5 in PLAN.md)

Run after all training is done. Evaluates each saved snapshot to produce the convergence curve.
Output is saved to `results/eval_log.txt` and printed to the terminal simultaneously.

```bash
mkdir -p results

for cond in cond_A_s1 cond_A_s2 cond_A_s3 cond_B_s1 cond_B_s2 cond_B_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=addition_2digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log.txt
```

The log format looks like:
```
=== cond_A_s1 @ iter 00500 ===
...
  Exact-match: 45.3%  (453/1000 correct)

=== cond_A_s1 @ iter 01000 ===
...
```

To extract just the accuracy numbers into a CSV for plotting:
```bash
grep -E "^===|Exact-match" results/eval_log.txt \
  | awk 'BEGIN { print "cond,iter,accuracy" }
    /^===/ { split($0, a, " "); cond=a[2]; iter=a[5] }
    /Exact-match/ { match($0, /[0-9]+\.[0-9]+/); print cond "," iter "," substr($0, RSTART, RLENGTH) }
  ' > results/accuracy.csv
```

Output: `cond,iter,accuracy` — one row per snapshot.

---

## Step 6 — Stretch Conditions (Scratchpad)

### Step 6a — Data Preparation

```bash
python gen_scratchpad.py

python prepare.py --file data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl \
    --out_dir data/scratchpad_1_2digit --shuffle
```

### Step 6b — Training

#### Condition C — Scratchpad, no mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_C_s1 --wandb_run_name=cond_C_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_C_s2 --wandb_run_name=cond_C_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_C_s3 --wandb_run_name=cond_C_s3
```

#### Condition D — Scratchpad, target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_D_s1 --wandb_run_name=cond_D_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_D_s2 --wandb_run_name=cond_D_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_1_2digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_D_s3 --wandb_run_name=cond_D_s3
```

### Step 6c — Post-hoc AR Eval

```bash
for cond in cond_C_s1 cond_C_s2 cond_C_s3 cond_D_s1 cond_D_s2 cond_D_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=scratchpad_1_2digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_scratchpad.txt
```

Extract to CSV:

```bash
grep -E "^===|Exact-match" results/eval_log_scratchpad.txt \
  | awk 'BEGIN { print "cond,iter,accuracy" }
    /^===/ { split($0, a, " "); cond=a[2]; iter=a[5] }
    /Exact-match/ { match($0, /[0-9]+\.[0-9]+/); print cond "," iter "," substr($0, RSTART, RLENGTH) }
  ' > results/accuracy_scratchpad.csv
```
