#!/bin/bash
# Phase 1 — Full training run (all 8 conditions, 5 seeds each)
# Run from masking_study/ directory.
# Estimated total: ~40 runs × ~20 min = ~13 hours

set -e  # stop on first error

TRAIN=../../comp560-nanoGPT/train_benchmark.py
export NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py

echo "=========================================="
echo "Phase 1 Training — $(date)"
echo "=========================================="

# ── Plain 3-digit ──────────────────────────────
echo "[1/8] Condition E — Plain 3-digit, no mask"
python $TRAIN config/plain_3digit.py --target_mask=False --seed=1337 --out_dir=out/cond_E_s1 --wandb_run_name=cond_E_s1
python $TRAIN config/plain_3digit.py --target_mask=False --seed=1338 --out_dir=out/cond_E_s2 --wandb_run_name=cond_E_s2
python $TRAIN config/plain_3digit.py --target_mask=False --seed=1339 --out_dir=out/cond_E_s3 --wandb_run_name=cond_E_s3
python $TRAIN config/plain_3digit.py --target_mask=False --seed=1340 --out_dir=out/cond_E_s4 --wandb_run_name=cond_E_s4
python $TRAIN config/plain_3digit.py --target_mask=False --seed=1341 --out_dir=out/cond_E_s5 --wandb_run_name=cond_E_s5

echo "[2/8] Condition F — Plain 3-digit, target mask"
python $TRAIN config/plain_3digit.py --target_mask=True --seed=1337 --out_dir=out/cond_F_s1 --wandb_run_name=cond_F_s1
python $TRAIN config/plain_3digit.py --target_mask=True --seed=1338 --out_dir=out/cond_F_s2 --wandb_run_name=cond_F_s2
python $TRAIN config/plain_3digit.py --target_mask=True --seed=1339 --out_dir=out/cond_F_s3 --wandb_run_name=cond_F_s3
python $TRAIN config/plain_3digit.py --target_mask=True --seed=1340 --out_dir=out/cond_F_s4 --wandb_run_name=cond_F_s4
python $TRAIN config/plain_3digit.py --target_mask=True --seed=1341 --out_dir=out/cond_F_s5 --wandb_run_name=cond_F_s5

# ── Plain 4-digit ──────────────────────────────
echo "[3/8] Condition G — Plain 4-digit, no mask"
python $TRAIN config/plain_4digit.py --target_mask=False --seed=1337 --out_dir=out/cond_G_s1 --wandb_run_name=cond_G_s1
python $TRAIN config/plain_4digit.py --target_mask=False --seed=1338 --out_dir=out/cond_G_s2 --wandb_run_name=cond_G_s2
python $TRAIN config/plain_4digit.py --target_mask=False --seed=1339 --out_dir=out/cond_G_s3 --wandb_run_name=cond_G_s3
python $TRAIN config/plain_4digit.py --target_mask=False --seed=1340 --out_dir=out/cond_G_s4 --wandb_run_name=cond_G_s4
python $TRAIN config/plain_4digit.py --target_mask=False --seed=1341 --out_dir=out/cond_G_s5 --wandb_run_name=cond_G_s5

echo "[4/8] Condition H — Plain 4-digit, target mask"
python $TRAIN config/plain_4digit.py --target_mask=True --seed=1337 --out_dir=out/cond_H_s1 --wandb_run_name=cond_H_s1
python $TRAIN config/plain_4digit.py --target_mask=True --seed=1338 --out_dir=out/cond_H_s2 --wandb_run_name=cond_H_s2
python $TRAIN config/plain_4digit.py --target_mask=True --seed=1339 --out_dir=out/cond_H_s3 --wandb_run_name=cond_H_s3
python $TRAIN config/plain_4digit.py --target_mask=True --seed=1340 --out_dir=out/cond_H_s4 --wandb_run_name=cond_H_s4
python $TRAIN config/plain_4digit.py --target_mask=True --seed=1341 --out_dir=out/cond_H_s5 --wandb_run_name=cond_H_s5

# ── Scratchpad 3-digit ─────────────────────────
echo "[5/8] Condition I — Scratchpad 3-digit, no mask"
python $TRAIN config/scratchpad_3digit.py --target_mask=False --seed=1337 --out_dir=out/cond_I_s1 --wandb_run_name=cond_I_s1
python $TRAIN config/scratchpad_3digit.py --target_mask=False --seed=1338 --out_dir=out/cond_I_s2 --wandb_run_name=cond_I_s2
python $TRAIN config/scratchpad_3digit.py --target_mask=False --seed=1339 --out_dir=out/cond_I_s3 --wandb_run_name=cond_I_s3
python $TRAIN config/scratchpad_3digit.py --target_mask=False --seed=1340 --out_dir=out/cond_I_s4 --wandb_run_name=cond_I_s4
python $TRAIN config/scratchpad_3digit.py --target_mask=False --seed=1341 --out_dir=out/cond_I_s5 --wandb_run_name=cond_I_s5

echo "[6/8] Condition J — Scratchpad 3-digit, target mask"
python $TRAIN config/scratchpad_3digit.py --target_mask=True --seed=1337 --out_dir=out/cond_J_s1 --wandb_run_name=cond_J_s1
python $TRAIN config/scratchpad_3digit.py --target_mask=True --seed=1338 --out_dir=out/cond_J_s2 --wandb_run_name=cond_J_s2
python $TRAIN config/scratchpad_3digit.py --target_mask=True --seed=1339 --out_dir=out/cond_J_s3 --wandb_run_name=cond_J_s3
python $TRAIN config/scratchpad_3digit.py --target_mask=True --seed=1340 --out_dir=out/cond_J_s4 --wandb_run_name=cond_J_s4
python $TRAIN config/scratchpad_3digit.py --target_mask=True --seed=1341 --out_dir=out/cond_J_s5 --wandb_run_name=cond_J_s5

# ── Scratchpad 4-digit ─────────────────────────
echo "[7/8] Condition K — Scratchpad 4-digit, no mask"
python $TRAIN config/scratchpad_4digit.py --target_mask=False --seed=1337 --out_dir=out/cond_K_s1 --wandb_run_name=cond_K_s1
python $TRAIN config/scratchpad_4digit.py --target_mask=False --seed=1338 --out_dir=out/cond_K_s2 --wandb_run_name=cond_K_s2
python $TRAIN config/scratchpad_4digit.py --target_mask=False --seed=1339 --out_dir=out/cond_K_s3 --wandb_run_name=cond_K_s3
python $TRAIN config/scratchpad_4digit.py --target_mask=False --seed=1340 --out_dir=out/cond_K_s4 --wandb_run_name=cond_K_s4
python $TRAIN config/scratchpad_4digit.py --target_mask=False --seed=1341 --out_dir=out/cond_K_s5 --wandb_run_name=cond_K_s5

echo "[8/8] Condition L — Scratchpad 4-digit, target mask"
python $TRAIN config/scratchpad_4digit.py --target_mask=True --seed=1337 --out_dir=out/cond_L_s1 --wandb_run_name=cond_L_s1
python $TRAIN config/scratchpad_4digit.py --target_mask=True --seed=1338 --out_dir=out/cond_L_s2 --wandb_run_name=cond_L_s2
python $TRAIN config/scratchpad_4digit.py --target_mask=True --seed=1339 --out_dir=out/cond_L_s3 --wandb_run_name=cond_L_s3
python $TRAIN config/scratchpad_4digit.py --target_mask=True --seed=1340 --out_dir=out/cond_L_s4 --wandb_run_name=cond_L_s4
python $TRAIN config/scratchpad_4digit.py --target_mask=True --seed=1341 --out_dir=out/cond_L_s5 --wandb_run_name=cond_L_s5

echo "=========================================="
echo "All training complete — $(date)"
echo "=========================================="
