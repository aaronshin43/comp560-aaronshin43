"""
Figure 3 for the final report: masking gap by input token fraction.

Bar chart of final autoregressive accuracy for plain (A/B) vs scratchpad (C/D),
with mean +/- std over 3 seeds and gap annotations above each pair.

Saves results/masking_gap_by_input_fraction.png. Run from masking_benchmark/.
"""

import numpy as np
import matplotlib.pyplot as plt

COLOR_NOMASK = '#E05C5C'
COLOR_MASKED = '#4C8EDA'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Per-seed final accuracies at iter 10,000.
A = np.array([83.9, 82.5, 84.2])
B = np.array([89.8, 90.2, 89.8])
C = np.array([86.3, 87.7, 86.4])
D = np.array([86.7, 84.8, 86.6])

means = np.array([A.mean(), B.mean(), C.mean(), D.mean()])
stds = np.array([A.std(ddof=0), B.std(ddof=0), C.std(ddof=0), D.std(ddof=0)])

group_centers = np.array([0.0, 1.2])
bar_offset = 0.22
positions = np.array([
    group_centers[0] - bar_offset,
    group_centers[0] + bar_offset,
    group_centers[1] - bar_offset,
    group_centers[1] + bar_offset,
])
colors = [COLOR_NOMASK, COLOR_MASKED, COLOR_NOMASK, COLOR_MASKED]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    positions, means, width=0.4,
    yerr=stds, color=colors, edgecolor='black', linewidth=0.6,
    capsize=4, error_kw={'elinewidth': 1.0, 'ecolor': '#444'},
)

for bar, m in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 4.0,
        f'{m:.1f}%',
        ha='center', va='top', color='white', fontsize=10, fontweight='bold',
    )

gap_ab = B.mean() - A.mean()
gap_cd = D.mean() - C.mean()

for center, gap, left_mean, right_mean in [
    (group_centers[0], gap_ab, A.mean(), B.mean()),
    (group_centers[1], gap_cd, C.mean(), D.mean()),
]:
    bracket_y = max(left_mean, right_mean) + 1.5
    ax.plot(
        [center - bar_offset, center - bar_offset, center + bar_offset, center + bar_offset],
        [bracket_y, bracket_y + 0.3, bracket_y + 0.3, bracket_y],
        color='black', linewidth=1.0,
    )
    sign = '+' if gap >= 0 else ''
    ax.text(
        center, bracket_y + 0.6,
        f'gap = {sign}{gap:.1f}pp',
        ha='center', va='bottom', fontsize=10,
    )

ax.set_xticks(group_centers)
ax.set_xticklabels([
    'Plain addition\n(~70% input)',
    'Scratchpad addition\n(~22% input)',
], fontsize=11)
ax.set_ylabel('Final autoregressive accuracy (%)', fontsize=11)
ax.set_ylim(75, 95)
ax.set_title(
    'Target masking benefit collapses as input token fraction shrinks',
    fontsize=12, pad=12,
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_NOMASK, ec='black', lw=0.6),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_MASKED, ec='black', lw=0.6),
]
ax.legend(
    handles, ['No mask (Cond A / C)', 'Target mask (Cond B / D)'],
    loc='upper right', frameon=False, fontsize=10,
)

ax.yaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = 'results/masking_gap_by_input_fraction.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'wrote {out_path}')
