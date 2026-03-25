"""
Plot convergence curves for the masking benchmark experiment.

Generates two figures saved to results/:
  - AR_accuracy_convergence_curve.png  (AR accuracy vs iterations)
  - val_loss_curve.png                 (Validation loss vs iterations)

Run from masking_benchmark/:
    python plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_A = '#E05C5C'   # red  — Condition A (no mask)
COLOR_B = '#4C8EDA'   # blue — Condition B (target mask)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── 1. AR Accuracy Convergence Curve ─────────────────────────────────────────
acc = pd.read_csv('results/accuracy.csv')
acc['iter'] = acc['iter'].astype(int)
acc['cond_label'] = acc['cond'].str.extract(r'(cond_[AB])')

fig, ax = plt.subplots(figsize=(8, 5))

for cond_label, color, name in [
    ('cond_A', COLOR_A, 'Condition A — No Mask'),
    ('cond_B', COLOR_B, 'Condition B — Target Mask'),
]:
    subset = acc[acc['cond_label'] == cond_label]
    grouped = subset.groupby('iter')['accuracy']
    mean = grouped.mean()
    lo   = grouped.min()
    hi   = grouped.max()

    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=name)
    ax.fill_between(mean.index, lo.values, hi.values,
                    color=color, alpha=0.15, linewidth=0)

ax.axhline(90, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax.axhline(85, color='gray', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(80, color='gray', linestyle=':', linewidth=1, alpha=0.4)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Val AR Accuracy (%)', fontsize=12)
ax.set_title('AR Accuracy vs. Iterations\n(mean ± seed range, 3 seeds per condition)', fontsize=13)
ax.set_xlim(0, 10000)
ax.set_ylim(0, 100)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('results/AR_accuracy_convergence_curve.png', dpi=150)
plt.close()
print("Saved results/AR_accuracy_convergence_curve.png")

# ── 2. Validation Loss Curve ──────────────────────────────────────────────────
loss_raw = pd.read_csv('results/wandb_val_loss.csv')

# Extract only the mean columns (exclude __MIN / __MAX)
mean_cols = [c for c in loss_raw.columns if 'val/loss' in c
             and '__MIN' not in c and '__MAX' not in c]

# Map W&B step → iteration (eval_interval = 500)
loss_raw['iter'] = loss_raw['Step'].astype(int) * 500

# Build tidy dataframe: iter, run, loss
rows = []
for col in mean_cols:
    run = col.split(' - ')[0].strip()   # e.g. "cond_A_s1"
    cond = run[:6]                       # e.g. "cond_A"
    tmp = loss_raw[['iter', col]].copy()
    tmp.columns = ['iter', 'loss']
    tmp['run']  = run
    tmp['cond'] = cond
    rows.append(tmp.dropna())

loss = pd.concat(rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 5))

for cond_label, color, name in [
    ('cond_A', COLOR_A, 'Condition A — No Mask'),
    ('cond_B', COLOR_B, 'Condition B — Target Mask'),
]:
    subset = loss[loss['cond'] == cond_label]
    grouped = subset.groupby('iter')['loss']
    mean = grouped.mean()
    lo   = grouped.min()
    hi   = grouped.max()

    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=name)
    ax.fill_between(mean.index, lo.values, hi.values,
                    color=color, alpha=0.15, linewidth=0)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss vs. Iterations\n(mean ± seed range, 3 seeds per condition)', fontsize=13)
ax.set_xlim(0, 10000)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('results/val_loss_curve.png', dpi=150)
plt.close()
print("Saved results/val_loss_curve.png")
