"""
Phase 2 analysis and plotting script — Exp 7, Input Fraction Manipulation.

Generates four plots and a summary table for conditions M–V.
Run from masking_study/:
    python plot_phase2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_NOMASK  = '#4C8EDA'   # blue
COLOR_MASKED  = '#E09A3C'   # orange

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS_DIR = 'D:/03_Coding/comp560-aaronshin43/masking_study/results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Multiplier metadata ───────────────────────────────────────────────────────
MULTS = [
    {'label': '1x',  'frac': '~19%', 'no_cond': 'M', 'mask_cond': 'N', 'file': 'accuracy_phase2_1x.csv', 'loss_file': 'val_loss_MN.csv'},
    {'label': '2x',  'frac': '~32%', 'no_cond': 'O', 'mask_cond': 'P', 'file': 'accuracy_phase2_2x.csv', 'loss_file': 'val_loss_OP.csv'},
    {'label': '3x',  'frac': '~41%', 'no_cond': 'Q', 'mask_cond': 'R', 'file': 'accuracy_phase2_3x.csv', 'loss_file': 'val_loss_QR.csv'},
    {'label': '4x',  'frac': '~48%', 'no_cond': 'S', 'mask_cond': 'T', 'file': 'accuracy_phase2_4x.csv', 'loss_file': 'val_loss_ST.csv'},
    {'label': '5x',  'frac': '~54%', 'no_cond': 'U', 'mask_cond': 'V', 'file': 'accuracy_phase2_5x.csv', 'loss_file': 'val_loss_UV.csv'},
]

# ── Load accuracy data ────────────────────────────────────────────────────────
def load_acc(mult):
    path = os.path.join(RESULTS_DIR, mult['file'])
    df = pd.read_csv(path)
    df['iter'] = df['iter'].astype(int)
    # extract the single-letter condition identifier
    df['base_cond'] = df['cond'].str.extract(r'cond_([A-Z])_s\d')
    return df

# ── Load loss data ────────────────────────────────────────────────────────────
def load_loss(mult):
    path = os.path.join(RESULTS_DIR, 'loss', mult['loss_file'])
    raw = pd.read_csv(path)
    raw['iter'] = raw['Step'].astype(int) * 500
    # keep only plain val/loss cols (no __MIN / __MAX)
    mean_cols = [c for c in raw.columns
                 if 'val/loss' in c and '__MIN' not in c and '__MAX' not in c]
    rows = []
    for col in mean_cols:
        run = col.split(' - ')[0].strip()      # e.g. "cond_M_s1"
        base = run[5]                           # single letter
        tmp = raw[['iter', col]].copy()
        tmp.columns = ['iter', 'loss']
        tmp['base_cond'] = base
        rows.append(tmp.dropna())
    return pd.concat(rows, ignore_index=True)

# ── Helper: mean ± std band ───────────────────────────────────────────────────
def plot_band(ax, df, cond_letter, color, label):
    sub = df[df['base_cond'] == cond_letter]
    grp = sub.groupby('iter')['accuracy']
    mean = grp.mean()
    std  = grp.std(ddof=1).fillna(0)
    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=label)
    ax.fill_between(mean.index,
                    (mean - std).values,
                    (mean + std).values,
                    color=color, alpha=0.15, linewidth=0)

def plot_loss_band(ax, df, cond_letter, color, label):
    sub = df[df['base_cond'] == cond_letter]
    grp = sub.groupby('iter')['loss']
    mean = grp.mean()
    std  = grp.std(ddof=1).fillna(0)
    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=label)
    ax.fill_between(mean.index,
                    (mean - std).values,
                    (mean + std).values,
                    color=color, alpha=0.15, linewidth=0)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Accuracy convergence curves (1×5 grid)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

for ax, m in zip(axes, MULTS):
    df = load_acc(m)
    plot_band(ax, df, m['no_cond'],   COLOR_NOMASK, 'No mask')
    plot_band(ax, df, m['mask_cond'], COLOR_MASKED,  'Target mask')

    for level, alpha in [(90, 0.6), (85, 0.4)]:
        ax.axhline(level, color='gray', linestyle=':', linewidth=1, alpha=alpha)

    ax.set_title(f"{m['label']} ({m['frac']} input)", fontsize=11)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

axes[0].set_ylabel('AR Accuracy (%)', fontsize=11)

fig.suptitle(
    'Accuracy vs. Iterations — Phase 2 Input Fraction Manipulation\n'
    '(mean ± std across 3 seeds)',
    fontsize=13, y=1.01
)
plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, 'accuracy_phase2_curves.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Masking gap bar chart
# ─────────────────────────────────────────────────────────────────────────────
gap_vals, gap_errs = [], []
no_peaks_all, mask_peaks_all = [], []

for m in MULTS:
    df = load_acc(m)

    no_df   = df[df['base_cond'] == m['no_cond']]
    mask_df = df[df['base_cond'] == m['mask_cond']]

    # per-seed peak
    no_peaks   = no_df.groupby('cond')['accuracy'].max().values
    mask_peaks = mask_df.groupby('cond')['accuracy'].max().values

    no_peaks_all.append(no_peaks)
    mask_peaks_all.append(mask_peaks)

    gap_per_seed = mask_peaks - no_peaks          # same seed order within cond
    # If seeds aren't guaranteed paired we use mean difference
    gap = mask_peaks.mean() - no_peaks.mean()
    # std of per-seed gaps as error bar
    err = np.std(gap_per_seed, ddof=1)
    gap_vals.append(gap)
    gap_errs.append(err)

x = np.arange(len(MULTS))
labels = [m['label'] for m in MULTS]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, gap_vals, yerr=gap_errs, capsize=5,
              color=[COLOR_MASKED if g >= 0 else '#E05C5C' for g in gap_vals],
              alpha=0.85, width=0.5, error_kw={'ecolor': 'black', 'linewidth': 1.5})
ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([f"{m['label']}\n({m['frac']})" for m in MULTS], fontsize=11)
ax.set_xlabel('Input Repetition Multiplier (input fraction)', fontsize=12)
ax.set_ylabel('Masking gap (pp)\n[masked peak − no-mask peak]', fontsize=11)
ax.set_title(
    'Peak Accuracy Masking Gap vs. Input Fraction\n'
    '(positive = masking helps; error bars = std across seeds)',
    fontsize=12
)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for bar, val, err in zip(bars, gap_vals, gap_errs):
    ypos = val + err + 0.05 if val >= 0 else val - err - 0.1
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f'{val:+.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, 'accuracy_phase2_gap.png')
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Val loss convergence curves (1×5 grid)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

for ax, m in zip(axes, MULTS):
    df_loss = load_loss(m)
    plot_loss_band(ax, df_loss, m['no_cond'],   COLOR_NOMASK, 'No mask')
    plot_loss_band(ax, df_loss, m['mask_cond'], COLOR_MASKED,  'Target mask')

    ax.set_title(f"{m['label']} ({m['frac']} input)", fontsize=11)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_xlim(0, 10000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

axes[0].set_ylabel('Validation Loss', fontsize=11)

fig.suptitle(
    'Validation Loss vs. Iterations — Phase 2 Input Fraction Manipulation\n'
    '(mean ± std across 3 seeds)',
    fontsize=13, y=1.01
)
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, 'val_loss_phase2_curves.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out3}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Summary line plot — peak accuracy vs multiplier
# ─────────────────────────────────────────────────────────────────────────────
no_means   = [p.mean() for p in no_peaks_all]
no_stds    = [p.std(ddof=1) for p in no_peaks_all]
mask_means = [p.mean() for p in mask_peaks_all]
mask_stds  = [p.std(ddof=1) for p in mask_peaks_all]

x_pos = np.arange(1, 6)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x_pos, no_means,   yerr=no_stds,   fmt='o-', color=COLOR_NOMASK,
            linewidth=2, markersize=7, capsize=4, label='No mask')
ax.errorbar(x_pos, mask_means, yerr=mask_stds, fmt='s-', color=COLOR_MASKED,
            linewidth=2, markersize=7, capsize=4, label='Target mask')

ax.set_xticks(x_pos)
ax.set_xticklabels([f"{m['label']}\n({m['frac']})" for m in MULTS], fontsize=11)
ax.set_xlabel('Input Repetition Multiplier (input fraction)', fontsize=12)
ax.set_ylabel('Mean Peak AR Accuracy (%)', fontsize=12)
ax.set_title(
    'Peak Accuracy vs. Input Fraction — Phase 2\n'
    '(mean ± std across 3 seeds)',
    fontsize=12
)
ax.set_ylim(95, 101)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
out4 = os.path.join(PLOTS_DIR, 'accuracy_phase2_summary.png')
plt.savefig(out4, dpi=150)
plt.close()
print(f"Saved {out4}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
header = (
    "| Multiplier | Input fraction | No-mask mean peak | No-mask std "
    "| Masked mean peak | Masked std | Gap (masked − no-mask) |"
)
sep = (
    "|---|---|---|---|---|---|---|"
)
rows_md = [header, sep]
for i, m in enumerate(MULTS):
    nm = no_means[i];  ns = no_stds[i]
    mm = mask_means[i]; ms = mask_stds[i]
    gap = mm - nm
    rows_md.append(
        f"| {m['label']} | {m['frac']} | {nm:.2f}% | ±{ns:.2f} | {mm:.2f}% | ±{ms:.2f} | {gap:+.2f} pp |"
    )

table_str = "\n".join(rows_md)
print("\n" + table_str)

summary_path = os.path.join(RESULTS_DIR, 'phase2_summary.txt')
with open(summary_path, 'w') as f:
    f.write(table_str + "\n")
print(f"\nSummary written to {summary_path}")
print(f"\nPlots:\n  {out1}\n  {out2}\n  {out3}\n  {out4}")
