"""
Plot convergence curves and summary bar chart for Exp 7 (masking_study).

Generates plots saved to results/plots/:
  - accuracy_plain3.png
  - accuracy_plain4.png
  - accuracy_scratch3.png
  - accuracy_scratch4.png
  - loss_plain3.png
  - loss_plain4.png
  - loss_scratch3.png
  - loss_scratch4.png
  - summary_bar.png

Run from masking_study/:
    python plot_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "D:/03_Coding/comp560-aaronshin43/masking_study/results"
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_NOMASK = '#4878CF'   # blue  — no mask
COLOR_MASK   = '#E8775A'   # orange — target mask

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

EVAL_INTERVAL = 500   # step * eval_interval = iteration number (for loss CSVs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_accuracy(csv_path):
    acc = pd.read_csv(csv_path)
    acc['iter'] = acc['iter'].astype(int)
    # Extract cond_X part (e.g. cond_E from cond_E_s1)
    acc['cond_label'] = acc['cond'].str.extract(r'(cond_[A-Z])')
    return acc


def plot_accuracy(acc, cond_pair, out_path, title):
    """
    cond_pair: [(cond_label, color, display_name), ...]
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_label, color, name in cond_pair:
        subset = acc[acc['cond_label'] == cond_label]
        grouped = subset.groupby('iter')['accuracy']
        mean = grouped.mean()
        std  = grouped.std()
        lo   = mean - std
        hi   = mean + std
        ax.plot(mean.index, mean.values, color=color, linewidth=2, label=name)
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=color, alpha=0.15, linewidth=0)

    for level, alpha in [(90, 0.6), (85, 0.4), (80, 0.4)]:
        ax.axhline(level, color='gray', linestyle=':', linewidth=1, alpha=alpha)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('AR Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(1000, 20000)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def load_loss(csv_path, metric_hint='val/loss'):
    """
    Load a W&B-exported loss CSV. Handles both val/loss and train/loss columns.
    Returns a long-form DataFrame with columns: iter, loss, run, cond.
    metric_hint: what substring to filter on ('val/loss' or 'train/loss').
    """
    loss_raw = pd.read_csv(csv_path)
    # Detect metric type present in file
    if any('val/loss' in c for c in loss_raw.columns):
        metric = 'val/loss'
    else:
        metric = 'train/loss'

    mean_cols = [c for c in loss_raw.columns
                 if metric in c and '__MIN' not in c and '__MAX' not in c]
    loss_raw['iter'] = loss_raw['Step'].astype(int) * EVAL_INTERVAL

    rows = []
    for col in mean_cols:
        run = col.split(' - ')[0].strip()
        cond = run[:6]  # e.g. cond_E
        tmp = loss_raw[['iter', col]].copy()
        tmp.columns = ['iter', 'loss']
        tmp['run']  = run
        tmp['cond'] = cond
        rows.append(tmp.dropna())
    return pd.concat(rows, ignore_index=True), metric


def plot_loss(loss_df, metric, cond_pair, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_label, color, name in cond_pair:
        subset = loss_df[loss_df['cond'] == cond_label]
        grouped = subset.groupby('iter')['loss']
        mean = grouped.mean()
        std  = grouped.std()
        lo   = mean - std
        hi   = mean + std
        ax.plot(mean.index, mean.values, color=color, linewidth=2, label=name)
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=color, alpha=0.15, linewidth=0)

    ylabel = 'Validation Loss' if metric == 'val/loss' else 'Training Loss'
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 20000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── 1. AR Accuracy Convergence Curves ─────────────────────────────────────────

acc_plain3   = load_accuracy(os.path.join(BASE, 'accuracy_plain3.csv'))
acc_plain4   = load_accuracy(os.path.join(BASE, 'accuracy_plain4.csv'))
acc_scratch3 = load_accuracy(os.path.join(BASE, 'accuracy_scratch3.csv'))
acc_scratch4 = load_accuracy(os.path.join(BASE, 'accuracy_scratch4.csv'))

PLAIN3_CONDITIONS = [
    ('cond_E', COLOR_NOMASK, 'Cond E — Plain 3-digit, No Mask'),
    ('cond_F', COLOR_MASK,   'Cond F — Plain 3-digit, Target Mask'),
]
PLAIN4_CONDITIONS = [
    ('cond_G', COLOR_NOMASK, 'Cond G — Plain 4-digit, No Mask'),
    ('cond_H', COLOR_MASK,   'Cond H — Plain 4-digit, Target Mask'),
]
SCRATCH3_CONDITIONS = [
    ('cond_I', COLOR_NOMASK, 'Cond I — Scratchpad 3-digit, No Mask'),
    ('cond_J', COLOR_MASK,   'Cond J — Scratchpad 3-digit, Target Mask'),
]
SCRATCH4_CONDITIONS = [
    ('cond_K', COLOR_NOMASK, 'Cond K — Scratchpad 4-digit, No Mask'),
    ('cond_L', COLOR_MASK,   'Cond L — Scratchpad 4-digit, Target Mask'),
]

plot_accuracy(acc_plain3, PLAIN3_CONDITIONS,
    os.path.join(OUT, 'accuracy_plain3.png'),
    'AR Accuracy vs. Iterations — Plain 3-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_accuracy(acc_plain4, PLAIN4_CONDITIONS,
    os.path.join(OUT, 'accuracy_plain4.png'),
    'AR Accuracy vs. Iterations — Plain 4-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_accuracy(acc_scratch3, SCRATCH3_CONDITIONS,
    os.path.join(OUT, 'accuracy_scratch3.png'),
    'AR Accuracy vs. Iterations — Scratchpad 3-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_accuracy(acc_scratch4, SCRATCH4_CONDITIONS,
    os.path.join(OUT, 'accuracy_scratch4.png'),
    'AR Accuracy vs. Iterations — Scratchpad 4-digit Addition\n(mean ± 1 std, 5 seeds per condition)')


# ── 2. Val Loss Convergence Curves ────────────────────────────────────────────

loss_ef, metric_ef = load_loss(os.path.join(BASE, 'loss', 'val_loss_EF.csv'))
loss_gh, metric_gh = load_loss(os.path.join(BASE, 'loss', 'val_loss_GH.csv'))
loss_ij, metric_ij = load_loss(os.path.join(BASE, 'loss', 'val_loss_IJ.csv'))
loss_kl, metric_kl = load_loss(os.path.join(BASE, 'loss', 'val_loss_KL.csv'))

plot_loss(loss_ef, metric_ef, PLAIN3_CONDITIONS,
    os.path.join(OUT, 'loss_plain3.png'),
    f'{"Validation" if metric_ef == "val/loss" else "Training"} Loss vs. Iterations — Plain 3-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_loss(loss_gh, metric_gh, PLAIN4_CONDITIONS,
    os.path.join(OUT, 'loss_plain4.png'),
    f'{"Validation" if metric_gh == "val/loss" else "Training"} Loss vs. Iterations — Plain 4-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_loss(loss_ij, metric_ij, SCRATCH3_CONDITIONS,
    os.path.join(OUT, 'loss_scratch3.png'),
    f'{"Validation" if metric_ij == "val/loss" else "Training"} Loss vs. Iterations — Scratchpad 3-digit Addition\n(mean ± 1 std, 5 seeds per condition)')

plot_loss(loss_kl, metric_kl, SCRATCH4_CONDITIONS,
    os.path.join(OUT, 'loss_scratch4.png'),
    f'{"Validation" if metric_kl == "val/loss" else "Training"} Loss vs. Iterations — Scratchpad 4-digit Addition\n(mean ± 1 std, 5 seeds per condition)')


# ── 3. Summary Bar Chart ──────────────────────────────────────────────────────

def peak_stats(acc_df, cond_label):
    """Return (mean, std) of per-seed peak accuracy for a condition."""
    subset = acc_df[acc_df['cond_label'] == cond_label]
    # isolate individual seeds
    seed_peaks = subset.groupby('cond')['accuracy'].max()
    return seed_peaks.mean(), seed_peaks.std()

groups = [
    ('Plain\n3-digit',   acc_plain3,   [('cond_E', 'No Mask'),   ('cond_F', 'Mask')]),
    ('Plain\n4-digit',   acc_plain4,   [('cond_G', 'No Mask'),   ('cond_H', 'Mask')]),
    ('Scratch\n3-digit', acc_scratch3, [('cond_I', 'No Mask'),   ('cond_J', 'Mask')]),
    ('Scratch\n4-digit', acc_scratch4, [('cond_K', 'No Mask'),   ('cond_L', 'Mask')]),
]

group_labels = [g[0] for g in groups]
no_mask_means, no_mask_stds = [], []
mask_means, mask_stds = [], []

for _, acc_df, conds in groups:
    nm_label, m_label = conds[0][0], conds[1][0]
    nm_mean, nm_std = peak_stats(acc_df, nm_label)
    m_mean,  m_std  = peak_stats(acc_df, m_label)
    no_mask_means.append(nm_mean)
    no_mask_stds.append(nm_std)
    mask_means.append(m_mean)
    mask_stds.append(m_std)

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, no_mask_means, width, yerr=no_mask_stds,
               color=COLOR_NOMASK, capsize=4, label='No Mask', alpha=0.9)
bars2 = ax.bar(x + width/2, mask_means,    width, yerr=mask_stds,
               color=COLOR_MASK,   capsize=4, label='Target Mask', alpha=0.9)

ax.set_xlabel('Format × Digit Length', fontsize=12)
ax.set_ylabel('Peak AR Accuracy (%)', fontsize=12)
ax.set_title('Peak AR Accuracy by Condition\n(mean ± 1 std across 5 seeds)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=11)
ax.set_ylim(0, 110)
ax.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate masking gap above each group
for i, (nm, m, nm_s, m_s) in enumerate(zip(no_mask_means, mask_means, no_mask_stds, mask_stds)):
    gap = m - nm
    bar_top = max(nm + nm_s, m + m_s) + 1.5
    ax.text(i, bar_top, f'gap={gap:+.1f}pp', ha='center', va='bottom', fontsize=9,
            color='#333333')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'summary_bar.png'), dpi=150)
plt.close()
print(f"Saved {os.path.join(OUT, 'summary_bar.png')}")


# ── 4. Print Summary Table ────────────────────────────────────────────────────

def first_iter_above(acc_df, cond_label, threshold=80.0):
    """Mean iteration (across seeds) at which accuracy first exceeds threshold."""
    subset = acc_df[acc_df['cond_label'] == cond_label].copy()
    results = []
    for seed_run, grp in subset.groupby('cond'):
        grp_sorted = grp.sort_values('iter')
        above = grp_sorted[grp_sorted['accuracy'] > threshold]
        if len(above) == 0:
            results.append(None)
        else:
            results.append(above['iter'].iloc[0])
    valid = [r for r in results if r is not None]
    if not valid:
        return 'never'
    return f'{int(np.mean(valid)):,}'

all_conditions = [
    ('E', 'Plain 3-digit',   'No',  acc_plain3,   'cond_E'),
    ('F', 'Plain 3-digit',   'Yes', acc_plain3,   'cond_F'),
    ('G', 'Plain 4-digit',   'No',  acc_plain4,   'cond_G'),
    ('H', 'Plain 4-digit',   'Yes', acc_plain4,   'cond_H'),
    ('I', 'Scratch 3-digit', 'No',  acc_scratch3, 'cond_I'),
    ('J', 'Scratch 3-digit', 'Yes', acc_scratch3, 'cond_J'),
    ('K', 'Scratch 4-digit', 'No',  acc_scratch4, 'cond_K'),
    ('L', 'Scratch 4-digit', 'Yes', acc_scratch4, 'cond_L'),
]

print()
print('| Condition | Format | Mask | Peak Accuracy (mean+/-std) | First iter >80% (mean) |')
print('|---|---|---|---|---|')
for cond, fmt, mask, acc_df, label in all_conditions:
    mean, std = peak_stats(acc_df, label)
    first = first_iter_above(acc_df, label, threshold=80.0)
    print(f'| {cond} | {fmt} | {mask} | {mean:.1f}+/-{std:.1f}% | {first} |')

print()

# Masking gaps
print('Masking gaps (masked minus unmasked peak accuracy):')
gap_pairs = [
    ('Plain 3-digit',   acc_plain3,   'cond_E', 'cond_F'),
    ('Plain 4-digit',   acc_plain4,   'cond_G', 'cond_H'),
    ('Scratch 3-digit', acc_scratch3, 'cond_I', 'cond_J'),
    ('Scratch 4-digit', acc_scratch4, 'cond_K', 'cond_L'),
]
for fmt, acc_df, no_m, with_m in gap_pairs:
    nm_mean, _ = peak_stats(acc_df, no_m)
    m_mean,  _ = peak_stats(acc_df, with_m)
    print(f'  {fmt}: {m_mean - nm_mean:+.1f}pp')
