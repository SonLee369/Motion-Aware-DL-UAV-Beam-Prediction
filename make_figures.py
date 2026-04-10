"""
One-time script: generate 4 figures for advisor report.
Run: python make_figures.py
Delete after use.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from pathlib import Path

OUT = Path('figures'); OUT.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = sns.color_palette('Set2', 8)

# ── Data from report.md (already validated) ──────────────────────────

# Fig 1 — E2 Finalist bar chart
fig1_labels = ['E2.0\n(5D Pos)', 'E2.2\n(+v_rad)', 'E2.13\n(+dir v_tan)', 'E2.14\n(+all vel+ω)']
fig1_means  = [73.74, 74.60, 74.54, 73.94]
fig1_stds   = [0.95,  0.41,  1.09,  1.15]
fig1_pvals  = [None,  0.032, 0.132, 0.686]

# Fig 2 — Line plot by horizon
fig2_horizons = ['v0', 'v1', 'v2', 'v3']
fig2_data = {
    'E2.0 (5D)':               [74.21, 74.64, 74.24, 71.86],
    'E2.2 (+v_rad)':           [75.19, 75.73, 74.86, 72.61],
    'E2.14 (+all vel+ω)':     [75.34, 75.05, 73.57, 71.81],
    'Bridge-Trans (6D_vrad)':  [75.69, 75.78, 75.38, 73.50],
}

# Fig 3 — Speed × Horizon heatmaps
fig3_configs = {
    'E2.0 (5D)': np.array([
        [79.01, 79.16, 78.25, 77.03],
        [66.46, 66.14, 65.81, 62.08],
        [61.05, 63.40, 65.11, 59.05],
    ]),
    'E2.2 (+v_rad)': np.array([
        [80.05, 80.01, 79.12, 78.36],
        [68.08, 68.11, 67.36, 62.25],
        [60.99, 64.55, 63.62, 57.69],
    ]),
    'E2.14 (+all vel+ω)': np.array([
        [78.74, 78.72, 77.59, 76.70],
        [71.04, 69.35, 67.17, 64.17],
        [64.68, 64.65, 62.31, 57.98],
    ]),
    'Bridge-Trans': np.array([
        [80.10, 80.06, 79.47, 78.88],
        [70.49, 69.19, 68.73, 63.00],
        [61.54, 63.59, 64.04, 60.32],
    ]),
}

# Fig 4 — Scatter Top-1 vs Params
fig4_data = [
    # (name, params_K, top1, feature_set)
    ('E4.0 Baseline\n8D',      260, 73.85, '8D'),
    ('E4.5 2L-GRU\n8D',        360, 74.24, '8D'),
    ('E4.7 Transformer\n8D',   559, 74.02, '8D'),
    ('Bridge Baseline\n6D_vrad', 260, 74.70, '6D_vrad'),
    ('Bridge 2L-GRU\n6D_vrad',  360, 74.91, '6D_vrad'),
    ('Bridge Trans\n6D_vrad',   558, 75.09, '6D_vrad'),
    ('E2.2 Baseline\n6D_vrad',  260, 74.60, '6D_vrad (E2)'),
]

# ── Figure 1: Bar Chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(fig1_labels))
bars = ax.bar(x, fig1_means, yerr=fig1_stds, capsize=6, width=0.55,
              color=[COLORS[1] if i == 1 else COLORS[0] for i in range(4)],
              edgecolor='gray', linewidth=0.8)

for i, (m, p) in enumerate(zip(fig1_means, fig1_pvals)):
    label = f'{m:.2f}%'
    if p is not None:
        label += f'\np={p:.3f}' + (' *' if p < 0.05 else ' n.s.')
    ax.text(i, m + fig1_stds[i] + 0.15, label, ha='center', va='bottom', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(fig1_labels)
ax.set_ylabel('Overall Top-1 Accuracy (%)')
ax.set_title('Fig 1: Feature Ablation — E2 Finalist (10 seeds)')
ax.set_ylim(71.5, 77)
ax.axhline(fig1_means[0], ls='--', color='gray', alpha=0.5, label='Baseline')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(OUT / 'fig1_e2_finalist_bar.png', dpi=200)
plt.close(fig)
print(f'Saved {OUT}/fig1_e2_finalist_bar.png')

# ── Figure 2: Line Plot by Horizon ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
markers = ['o', 's', '^', 'D']
for i, (name, vals) in enumerate(fig2_data.items()):
    ax.plot(fig2_horizons, vals, marker=markers[i], label=name,
            linewidth=2, markersize=7, color=COLORS[i])

ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('Fig 2: Accuracy by Prediction Horizon')
ax.set_ylim(70, 77)
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(OUT / 'fig2_horizon_line.png', dpi=200)
plt.close(fig)
print(f'Saved {OUT}/fig2_horizon_line.png')

# ── Figure 3: Speed × Horizon Heatmaps ──────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
speed_labels = ['Slow', 'Medium', 'Fast']
horizon_labels = ['v0', 'v1', 'v2', 'v3']

vmin = 56; vmax = 82
for ax, (name, data) in zip(axes, fig3_configs.items()):
    sns.heatmap(data, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd_r',
                vmin=vmin, vmax=vmax, cbar=False,
                xticklabels=horizon_labels, yticklabels=speed_labels if ax == axes[0] else False)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Horizon')

# Shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='YlOrRd_r', norm=plt.Normalize(vmin, vmax))
fig.colorbar(sm, cax=cbar_ax, label='Top-1 (%)')

fig.suptitle('Fig 3: Speed × Horizon Top-1 Accuracy', fontsize=12, y=1.02)
fig.tight_layout(rect=[0, 0, 0.92, 1])
fig.savefig(OUT / 'fig3_speed_horizon_heatmap.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT}/fig3_speed_horizon_heatmap.png')

# ── Figure 4: Scatter Top-1 vs Params ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
feat_colors = {'8D': COLORS[3], '6D_vrad': COLORS[1], '6D_vrad (E2)': COLORS[2]}
feat_markers = {'8D': 'o', '6D_vrad': 's', '6D_vrad (E2)': 'D'}

for name, params, top1, feat in fig4_data:
    ax.scatter(params, top1, s=120, c=[feat_colors[feat]], marker=feat_markers[feat],
               edgecolors='black', linewidth=0.8, zorder=5)
    offset_x = 12 if 'Trans' not in name else -12
    ha = 'left' if 'Trans' not in name else 'right'
    ax.annotate(name, (params, top1), textcoords='offset points',
                xytext=(offset_x, 8), fontsize=7.5, ha=ha)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=feat_colors['8D'],
           markersize=10, markeredgecolor='black', label='8D features'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=feat_colors['6D_vrad'],
           markersize=10, markeredgecolor='black', label='6D_vrad (Bridge)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=feat_colors['6D_vrad (E2)'],
           markersize=10, markeredgecolor='black', label='6D_vrad (E2)'),
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('Parameters (K)')
ax.set_ylabel('Overall Top-1 Accuracy (%)')
ax.set_title('Fig 4: Top-1 vs Model Size — Feature Dominates Architecture')
ax.set_xlim(200, 620)
ax.set_ylim(73.2, 75.6)

# Draw horizontal band showing 6D_vrad cluster is ABOVE 8D cluster
ax.axhspan(74.5, 75.2, alpha=0.08, color=COLORS[1], label='_')
ax.axhspan(73.5, 74.4, alpha=0.08, color=COLORS[3], label='_')
ax.text(580, 74.85, '6D_vrad\ncluster', fontsize=8, color=COLORS[1], ha='right', style='italic')
ax.text(580, 73.95, '8D\ncluster', fontsize=8, color=COLORS[3], ha='right', style='italic')

fig.tight_layout()
fig.savefig(OUT / 'fig4_top1_vs_params.png', dpi=200)
plt.close(fig)
print(f'Saved {OUT}/fig4_top1_vs_params.png')

print('\nDone! All 4 figures in figures/')
