import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv(r'D:\Research\ExtremeHeat\Preprocess\Journal\Code\data\buffer_analysis.csv', encoding='utf-8-sig')

cluster_order = ['Heat-Sensitive MSAs', 'Temperature-Resilient MSAs', 'Adaptive (Baseline) MSAs']

cluster_colors = {
    'Heat-Sensitive MSAs':        '#E64B35',
    'Temperature-Resilient MSAs': '#00A087',
    'Adaptive (Baseline) MSAs':   '#0053A0',
}
COLOR_NO_HEAT = '#6C7EB7'
COLOR_HEAT    = '#E07B54'

# Sort: Heat-Sensitive at top → invert y, so build bottom-to-top as Adaptive first
rows = []
for cl in reversed(cluster_order):
    sub = df[df['Cluster'] == cl].sort_values('NoHeat_Ratio_In_Out', ascending=True)
    rows.append(sub)
df_sorted = pd.concat(rows, ignore_index=True)

cities      = df_sorted['City'].tolist()
noheat_vals = df_sorted['NoHeat_Ratio_In_Out'].tolist()
heat_vals   = df_sorted['Heat_Ratio_In_Out'].tolist()
clusters    = df_sorted['Cluster'].tolist()
n           = len(cities)
y_pos       = np.arange(n)

fig, ax = plt.subplots(figsize=(12, 10))

for i, (noh, h) in enumerate(zip(noheat_vals, heat_vals)):
    ax.plot([noh, h], [i, i], color='gray', linewidth=1.2, alpha=0.6, zorder=1)
    ax.scatter(noh, i, color=COLOR_NO_HEAT, s=60, zorder=3, alpha=0.9)
    ax.scatter(h,   i, color=COLOR_HEAT,    s=60, zorder=3, alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(cities, fontsize=11)
ax.set_xlim(0, None)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel('The Percentage of Walkable POIs', fontsize=12)
ax.tick_params(axis='x', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.5)

# Cluster separator lines
prev_cl = None
for i, cl in enumerate(clusters):
    if cl != prev_cl and prev_cl is not None:
        ax.axhline(i - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    prev_cl = cl


legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NO_HEAT,
           markersize=9, label='No Extreme Heat'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HEAT,
           markersize=9, label='Extreme Heat'),
]
ax.legend(handles=legend_elements, title='Condition', fontsize=11,
          title_fontsize=11, loc='lower right', frameon=True,
          framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig(r'D:\Research\ExtremeHeat\Preprocess\Journal\Code\Fig 3c. Walkable POIs.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved")
