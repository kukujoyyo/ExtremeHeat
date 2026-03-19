import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
INPUT_FILE = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\CBG_ALLcity_Week_NAICS.csv'
OUTPUT_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"

CITY_CLUSTERS = {
    'Heat-Sensitive':        ['Chicago', 'Dallas', 'Houston'],
    'Temperature-Resilient': ['LA', 'NewYork', 'phoenix'],
    'Adaptive (Baseline)':   ['SanFrancisco', 'SanDiego', 'SanAntonio', 'philadelphia',
                              'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
                              'Indianapolis', 'Denver', 'Columbus', 'Charlotte', 'Nashville', 'Seattle'],
}
TARGET_NAICS = {6221: 'Healthcare facilities', 4451: 'Grocery stores', 8129: 'Service establishments'}
VARIABLES    = ['Temperature', 'distance/person/day', 'visitation_ratio']
Y_LABELS     = {
    'distance/person/day': 'Daily Distance/Person (km)',
    'Temperature':         'Temperature (°C)',
    'visitation_ratio':    'Visitation Frequency',
}

# ── Data loading ────────────────────────────────────────────────────────────
def load_and_prepare_data():
    data = pd.read_csv(INPUT_FILE)
    data['date_range_start'] = pd.to_datetime(data['date_range_start'])

    total_visits  = (data.groupby(['City', 'date_range_start'])['cbgs_visitor_7day_avg']
                         .sum().reset_index().rename(columns={'cbgs_visitor_7day_avg': 'total_visits'}))
    target_data   = data[data['naics_code_unique'].isin(TARGET_NAICS)].copy()
    target_visits = (target_data.groupby(['City', 'date_range_start'])['cbgs_visitor_7day_avg']
                                .sum().reset_index().rename(columns={'cbgs_visitor_7day_avg': 'target_visits'}))
    agg_vars      = (target_data.groupby(['City', 'date_range_start'])
                                .agg({'distance/person/day': 'mean', 'Temperature': 'mean'})
                                .reset_index())

    merged = (agg_vars
              .merge(target_visits, on=['City', 'date_range_start'])
              .merge(total_visits,  on=['City', 'date_range_start']))
    merged['visitation_ratio'] = merged['target_visits'] / merged['total_visits']

    city_map = {city: cluster for cluster, cities in CITY_CLUSTERS.items() for city in cities}
    merged['city_cluster'] = merged['City'].map(city_map)
    merged.dropna(subset=['city_cluster'], inplace=True)
    print(f"Data prepared: {len(merged)} records")
    return merged

# ── Violin plots (combined) ─────────────────────────────────────────────────
def create_violin_plots(data):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    cluster_order = list(CITY_CLUSTERS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(36, 8))

    for ax, var, panel in zip(axes, VARIABLES, 'cde'):
        sns.violinplot(x='city_cluster', y=var, data=data, order=cluster_order,
                       palette='muted', ax=ax)

        for i, cluster in enumerate(cluster_order):
            grp = data[data['city_cluster'] == cluster][var].dropna()
            if grp.empty:
                continue
            for val, label, va, color in [
                (grp.median(), f'Median: {grp.median():.2f}', 'center', ('black', 'white')),
                (grp.min(),    f'Min: {grp.min():.2f}',       'top',    ('white', 'black')),
                (grp.max(),    f'Max: {grp.max():.2f}',       'bottom', ('white', 'black')),
            ]:
                ax.text(i, val, label, ha='center', va=va, size=24,
                        color=color[1], weight='normal',
                        bbox=dict(facecolor=color[0], alpha=0.5, boxstyle='round,pad=0.2'))

        ax.set_xlabel('City Cluster', fontsize=24)
        ax.set_ylabel(Y_LABELS.get(var, var), fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.set_title(f'({panel})', fontsize=24, fontweight='bold', loc='left')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig 2c-e. violin_plots_combined.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    data = load_and_prepare_data()
    create_violin_plots(data)
    print("Analysis complete.")

if __name__ == "__main__":
    main()