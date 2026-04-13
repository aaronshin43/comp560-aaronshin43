# Masking Study — Commands

Run all commands from `masking_study/`.

---

## Phase 1 — Digit Length Extension

### Step 1 — Data Preparation

Already done via `gen_data.py` and `prepare.py`. Outputs are in `data/`.

---

### Step 2 — Training

```bash
bash run_training.sh
```

---

### Step 3 — Post-hoc AR Eval

Run after all training is done. Evaluates each saved snapshot across all conditions.
Uses `--eval_max_samples=1000` to limit eval time (~±3.2% margin at 95% CI).

#### Plain conditions (E, F, G, H)

```bash
mkdir -p results

for cond in cond_E_s1 cond_E_s2 cond_E_s3 cond_E_s4 cond_E_s5 \
            cond_F_s1 cond_F_s2 cond_F_s3 cond_F_s4 cond_F_s5; do
  for iter in 01000 02000 03000 04000 05000 06000 07000 08000 09000 10000 \
               11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=plain_3digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val \
      --eval_max_samples=1000
  done
done 2>&1 | tee results/eval_log_plain3.txt

for cond in cond_G_s1 cond_G_s2 cond_G_s3 cond_G_s4 cond_G_s5 \
            cond_H_s1 cond_H_s2 cond_H_s3 cond_H_s4 cond_H_s5; do
  for iter in 01000 02000 03000 04000 05000 06000 07000 08000 09000 10000 \
               11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=plain_4digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val \
      --eval_max_samples=1000
  done
done 2>&1 | tee results/eval_log_plain4.txt
```

#### Scratchpad conditions (I, J, K, L)

```bash
for cond in cond_I_s1 cond_I_s2 cond_I_s3 cond_I_s4 cond_I_s5 \
            cond_J_s1 cond_J_s2 cond_J_s3 cond_J_s4 cond_J_s5; do
  for iter in 01000 02000 03000 04000 05000 06000 07000 08000 09000 10000 \
               11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=scratchpad_3digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val \
      --eval_max_samples=1000
  done
done 2>&1 | tee results/eval_log_scratch3.txt

for cond in cond_K_s1 cond_K_s2 cond_K_s3 cond_K_s4 cond_K_s5 \
            cond_L_s1 cond_L_s2 cond_L_s3 cond_L_s4 cond_L_s5; do
  for iter in 01000 02000 03000 04000 05000 06000 07000 08000 09000 10000 \
               11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=scratchpad_4digit \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val \
      --eval_max_samples=1000
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

---

## Phase 2 — Input Fraction Manipulation

### Step 5 — AR Eval (Phase 2)

Val set is 810 samples — evaluate all of them (no `--eval_max_samples` limit).

```bash
for cond in cond_M_s1 cond_M_s2 cond_M_s3 cond_N_s1 cond_N_s2 cond_N_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=phase2_1x \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_phase2_1x.txt

for cond in cond_O_s1 cond_O_s2 cond_O_s3 cond_P_s1 cond_P_s2 cond_P_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=phase2_2x \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_phase2_2xb.txt

for cond in cond_Q_s1 cond_Q_s2 cond_Q_s3 cond_R_s1 cond_R_s2 cond_R_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=phase2_3x \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_phase2_3x.txt

for cond in cond_S_s1 cond_S_s2 cond_S_s3 cond_T_s1 cond_T_s2 cond_T_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ==="
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=phase2_4x \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_phase2_4x.txt

for cond in cond_U_s1 cond_U_s2 cond_U_s3 cond_V_s1 cond_V_s2 cond_V_s3; do
  for iter in 00500 01000 01500 02000 02500 03000 03500 04000 04500 05000 \
               05500 06000 06500 07000 07500 08000 08500 09000 09500 10000; do
    echo "=== $cond @ iter $iter ===" 
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
      python ../../comp560-nanoGPT/eval_generation.py \
      --out_dir=out/$cond \
      --dataset=phase2_5x \
      --ckpt_path=out/$cond/ckpt_${iter}.pt \
      --benchmark_target=val
  done
done 2>&1 | tee results/eval_log_phase2_5x.txt
```

### Step 6 — Extract to CSV (Phase 2)

```bash
for log in phase2_1x phase2_2x phase2_3x phase2_4x phase2_5x; do
  grep -E "^===|Exact-match" results/eval_log_${log}.txt \
    | awk 'BEGIN { print "cond,iter,accuracy" }
      /^===/ { split($0, a, " "); cond=a[2]; iter=a[5] }
      /Exact-match/ { match($0, /[0-9]+\.[0-9]+/); print cond "," iter "," substr($0, RSTART, RLENGTH) }
    ' > results/accuracy_${log}.csv
done
```

Output: 5 CSV files — `accuracy_phase2_1x.csv` through `accuracy_phase2_5x.csv`.
