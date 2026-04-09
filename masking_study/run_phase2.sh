#!/bin/bash
# Phase 2 — Input Fraction Manipulation (10 conditions, 3 seeds each = 30 runs)
# Run from masking_study/ directory.
# Estimated total: ~30 runs × ~10 min = ~5 hours

set -e  # stop on first error

TRAIN=../../comp560-nanoGPT/train_benchmark.py
export NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py

echo "=========================================="
echo "Phase 2 Training — $(date)"
echo "=========================================="

# ── 1x (input fraction ~19%) ───────────────────
echo "[1/10] Condition M — 1x, no mask"
python $TRAIN config/phase2_1x.py --target_mask=False --seed=1337 --out_dir=out/cond_M_s1 --wandb_run_name=cond_M_s1
python $TRAIN config/phase2_1x.py --target_mask=False --seed=1338 --out_dir=out/cond_M_s2 --wandb_run_name=cond_M_s2
python $TRAIN config/phase2_1x.py --target_mask=False --seed=1339 --out_dir=out/cond_M_s3 --wandb_run_name=cond_M_s3

echo "[2/10] Condition N — 1x, target mask"
python $TRAIN config/phase2_1x.py --target_mask=True --seed=1337 --out_dir=out/cond_N_s1 --wandb_run_name=cond_N_s1
python $TRAIN config/phase2_1x.py --target_mask=True --seed=1338 --out_dir=out/cond_N_s2 --wandb_run_name=cond_N_s2
python $TRAIN config/phase2_1x.py --target_mask=True --seed=1339 --out_dir=out/cond_N_s3 --wandb_run_name=cond_N_s3

# ── 2x (input fraction ~32%) ───────────────────
echo "[3/10] Condition O — 2x, no mask"
python $TRAIN config/phase2_2x.py --target_mask=False --seed=1337 --out_dir=out/cond_O_s1 --wandb_run_name=cond_O_s1
python $TRAIN config/phase2_2x.py --target_mask=False --seed=1338 --out_dir=out/cond_O_s2 --wandb_run_name=cond_O_s2
python $TRAIN config/phase2_2x.py --target_mask=False --seed=1339 --out_dir=out/cond_O_s3 --wandb_run_name=cond_O_s3

echo "[4/10] Condition P — 2x, target mask"
python $TRAIN config/phase2_2x.py --target_mask=True --seed=1337 --out_dir=out/cond_P_s1 --wandb_run_name=cond_P_s1
python $TRAIN config/phase2_2x.py --target_mask=True --seed=1338 --out_dir=out/cond_P_s2 --wandb_run_name=cond_P_s2
python $TRAIN config/phase2_2x.py --target_mask=True --seed=1339 --out_dir=out/cond_P_s3 --wandb_run_name=cond_P_s3

# ── 3x (input fraction ~41%) ───────────────────
echo "[5/10] Condition Q — 3x, no mask"
python $TRAIN config/phase2_3x.py --target_mask=False --seed=1337 --out_dir=out/cond_Q_s1 --wandb_run_name=cond_Q_s1
python $TRAIN config/phase2_3x.py --target_mask=False --seed=1338 --out_dir=out/cond_Q_s2 --wandb_run_name=cond_Q_s2
python $TRAIN config/phase2_3x.py --target_mask=False --seed=1339 --out_dir=out/cond_Q_s3 --wandb_run_name=cond_Q_s3

echo "[6/10] Condition R — 3x, target mask"
python $TRAIN config/phase2_3x.py --target_mask=True --seed=1337 --out_dir=out/cond_R_s1 --wandb_run_name=cond_R_s1
python $TRAIN config/phase2_3x.py --target_mask=True --seed=1338 --out_dir=out/cond_R_s2 --wandb_run_name=cond_R_s2
python $TRAIN config/phase2_3x.py --target_mask=True --seed=1339 --out_dir=out/cond_R_s3 --wandb_run_name=cond_R_s3

# ── 4x (input fraction ~48%) ───────────────────
echo "[7/10] Condition S — 4x, no mask"
python $TRAIN config/phase2_4x.py --target_mask=False --seed=1337 --out_dir=out/cond_S_s1 --wandb_run_name=cond_S_s1
python $TRAIN config/phase2_4x.py --target_mask=False --seed=1338 --out_dir=out/cond_S_s2 --wandb_run_name=cond_S_s2
python $TRAIN config/phase2_4x.py --target_mask=False --seed=1339 --out_dir=out/cond_S_s3 --wandb_run_name=cond_S_s3

echo "[8/10] Condition T — 4x, target mask"
python $TRAIN config/phase2_4x.py --target_mask=True --seed=1337 --out_dir=out/cond_T_s1 --wandb_run_name=cond_T_s1
python $TRAIN config/phase2_4x.py --target_mask=True --seed=1338 --out_dir=out/cond_T_s2 --wandb_run_name=cond_T_s2
python $TRAIN config/phase2_4x.py --target_mask=True --seed=1339 --out_dir=out/cond_T_s3 --wandb_run_name=cond_T_s3

# ── 5x (input fraction ~54%) ───────────────────
echo "[9/10] Condition U — 5x, no mask"
python $TRAIN config/phase2_5x.py --target_mask=False --seed=1337 --out_dir=out/cond_U_s1 --wandb_run_name=cond_U_s1
python $TRAIN config/phase2_5x.py --target_mask=False --seed=1338 --out_dir=out/cond_U_s2 --wandb_run_name=cond_U_s2
python $TRAIN config/phase2_5x.py --target_mask=False --seed=1339 --out_dir=out/cond_U_s3 --wandb_run_name=cond_U_s3

echo "[10/10] Condition V — 5x, target mask"
python $TRAIN config/phase2_5x.py --target_mask=True --seed=1337 --out_dir=out/cond_V_s1 --wandb_run_name=cond_V_s1
python $TRAIN config/phase2_5x.py --target_mask=True --seed=1338 --out_dir=out/cond_V_s2 --wandb_run_name=cond_V_s2
python $TRAIN config/phase2_5x.py --target_mask=True --seed=1339 --out_dir=out/cond_V_s3 --wandb_run_name=cond_V_s3

echo "=========================================="
echo "Phase 2 complete — $(date)"
echo "=========================================="
