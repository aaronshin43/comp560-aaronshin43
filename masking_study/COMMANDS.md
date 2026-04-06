# Masking Study — Commands

Run all commands from `masking_study/`.

---

## Phase 1 — Digit Length Extension

### Step 1 — Data Preparation

Already done via `gen_data.py` and `prepare.py`. Outputs are in `data/`.

---

### Step 2 — Training

#### Condition E — Plain 3-digit, no mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_E_s1 --wandb_run_name=cond_E_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_E_s2 --wandb_run_name=cond_E_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_E_s3 --wandb_run_name=cond_E_s3
```

#### Condition F — Plain 3-digit, target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_F_s1 --wandb_run_name=cond_F_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_F_s2 --wandb_run_name=cond_F_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_3digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_F_s3 --wandb_run_name=cond_F_s3
```

#### Condition G — Plain 4-digit, no mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_G_s1 --wandb_run_name=cond_G_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_G_s2 --wandb_run_name=cond_G_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_G_s3 --wandb_run_name=cond_G_s3
```

#### Condition H — Plain 4-digit, target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_H_s1 --wandb_run_name=cond_H_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_H_s2 --wandb_run_name=cond_H_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/plain_4digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_H_s3 --wandb_run_name=cond_H_s3
```

#### Condition I — Scratchpad 3-digit, no mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_I_s1 --wandb_run_name=cond_I_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_I_s2 --wandb_run_name=cond_I_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_I_s3 --wandb_run_name=cond_I_s3
```

#### Condition J — Scratchpad 3-digit, target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_J_s1 --wandb_run_name=cond_J_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_J_s2 --wandb_run_name=cond_J_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_3digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_J_s3 --wandb_run_name=cond_J_s3
```

#### Condition K — Scratchpad 4-digit, no mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=False --seed=1337 --out_dir=out/cond_K_s1 --wandb_run_name=cond_K_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=False --seed=1338 --out_dir=out/cond_K_s2 --wandb_run_name=cond_K_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=False --seed=1339 --out_dir=out/cond_K_s3 --wandb_run_name=cond_K_s3
```

#### Condition L — Scratchpad 4-digit, target mask

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=True --seed=1337 --out_dir=out/cond_L_s1 --wandb_run_name=cond_L_s1

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=True --seed=1338 --out_dir=out/cond_L_s2 --wandb_run_name=cond_L_s2

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train_benchmark.py config/scratchpad_4digit.py \
  --target_mask=True --seed=1339 --out_dir=out/cond_L_s3 --wandb_run_name=cond_L_s3
```

---

### Step 3 — Post-hoc AR Eval

Run after all training is done. Evaluates each saved snapshot across all conditions.

#### Plain conditions (E, F, G, H)

```bash
mkdir -p results

for cond in cond_E_s1 cond_E_s2 cond_E_s3 cond_F_s1 cond_F_s2 cond_F_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=plain_3digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_plain3.txt

for cond in cond_G_s1 cond_G_s2 cond_G_s3 cond_H_s1 cond_H_s2 cond_H_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=plain_4digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_plain4.txt
```

#### Scratchpad conditions (I, J, K, L)

```bash
for cond in cond_I_s1 cond_I_s2 cond_I_s3 cond_J_s1 cond_J_s2 cond_J_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=scratchpad_3digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_scratch3.txt

for cond in cond_K_s1 cond_K_s2 cond_K_s3 cond_L_s1 cond_L_s2 cond_L_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=scratchpad_4digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_scratch4.txt
```

### Step 4 — Extract to CSV

```bash
for log in plain3 plain4 scratch3 scratch4; do
  grep -E "^===|Exact-match" results/eval_log_${log}.txt \
    | awk 'BEGIN { print "cond,iter,accuracy" }
      /^===/ { split($0, a, " "); cond=a[2]; iter=a[5] }
      /Exact-match/ { match($0, /[0-9]+\.[0-9]+/); print cond "," iter "," substr($0, RSTART, RLENGTH) }
    ' > results/accuracy_${log}.csv
done
```

Output: 4 CSV files — `accuracy_plain3.csv`, `accuracy_plain4.csv`, `accuracy_scratch3.csv`, `accuracy_scratch4.csv`.
