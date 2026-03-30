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
COLOR_A = '#E05C5C'   # red    — Condition A (plain, no mask)
COLOR_B = '#4C8EDA'   # blue   — Condition B (plain, target mask)
COLOR_C = '#E09A3C'   # orange — Condition C (scratchpad, no mask)
COLOR_D = '#4CAD72'   # green  — Condition D (scratchpad, target mask)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def plot_accuracy(csv_path, conditions, out_path, title):
    acc = pd.read_csv(csv_path)
    acc['iter'] = acc['iter'].astype(int)
    acc['cond_label'] = acc['cond'].str.extract(
        r'(cond_[' + ''.join(c[0][-1] for c in conditions) + r'])'
    )

    _, ax = plt.subplots(figsize=(8, 5))
    for cond_label, color, name in conditions:
        subset = acc[acc['cond_label'] == cond_label]
        grouped = subset.groupby('iter')['accuracy']
        mean = grouped.mean()
        lo   = grouped.min()
        hi   = grouped.max()
        ax.plot(mean.index, mean.values, color=color, linewidth=2, label=name)
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=color, alpha=0.15, linewidth=0)

    for level, alpha in [(90, 0.6), (85, 0.4), (80, 0.4)]:
        ax.axhline(level, color='gray', linestyle=':', linewidth=1, alpha=alpha)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Val AR Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_loss(csv_path, conditions, out_path, title):
    loss_raw = pd.read_csv(csv_path)
    mean_cols = [c for c in loss_raw.columns if 'val/loss' in c
                 and '__MIN' not in c and '__MAX' not in c]
    loss_raw['iter'] = loss_raw['Step'].astype(int) * 500

    rows = []
    for col in mean_cols:
        run = col.split(' - ')[0].strip()
        cond = run[:6]
        tmp = loss_raw[['iter', col]].copy()
        tmp.columns = ['iter', 'loss']
        tmp['run']  = run
        tmp['cond'] = cond
        rows.append(tmp.dropna())
    loss = pd.concat(rows, ignore_index=True)

    _, ax = plt.subplots(figsize=(8, 5))
    for cond_label, color, name in conditions:
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
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 10000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── 1. Plain addition: A vs B ─────────────────────────────────────────────────
AB_CONDITIONS = [
    ('cond_A', COLOR_A, 'Condition A — No Mask'),
    ('cond_B', COLOR_B, 'Condition B — Target Mask'),
]

plot_accuracy(
    'results/accuracy_ab.csv', AB_CONDITIONS,
    'results/AR_accuracy_convergence_curve_ab.png',
    'AR Accuracy vs. Iterations — Plain Addition\n(mean ± seed range, 3 seeds per condition)',
)
plot_loss(
    'results/val_loss_ab.csv', AB_CONDITIONS,
    'results/val_loss_curve_ab.png',
    'Validation Loss vs. Iterations — Plain Addition\n(mean ± seed range, 3 seeds per condition)',
)

# ── 2. Scratchpad: C vs D ─────────────────────────────────────────────────────
CD_CONDITIONS = [
    ('cond_C', COLOR_C, 'Condition C — Scratchpad, No Mask'),
    ('cond_D', COLOR_D, 'Condition D — Scratchpad, Target Mask'),
]

plot_accuracy(
    'results/accuracy_scratchpad.csv', CD_CONDITIONS,
    'results/AR_accuracy_convergence_curve_cd.png',
    'AR Accuracy vs. Iterations — Scratchpad Addition\n(mean ± seed range, 3 seeds per condition)',
)
plot_loss(
    'results/val_loss_cd.csv', CD_CONDITIONS,
    'results/val_loss_curve_cd.png',
    'Validation Loss vs. Iterations — Scratchpad Addition\n(mean ± seed range, 3 seeds per condition)',
)
