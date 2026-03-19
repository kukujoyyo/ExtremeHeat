import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_PATH  = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Traffic_NAICS_Temp_processed"
OUTPUT_DIR  = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"

CITY_CLUSTERS = {
    'Heat-sensitive MSAs':        ['Chicago', 'Dallas', 'Houston'],
    'Temperature-resilient MSAs': ['LA', 'Phoenix', 'NewYork'],
    'Adaptive/baseline MSAs':     ['SanFrancisco', 'SanDiego', 'SanAntonio', 'Philadelphia',
                                   'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
                                   'Indianapolis', 'Denver', 'Columbus', 'Charlotte',
                                   'Nashville', 'Seattle'],
}
CITY_TO_CLUSTER = {city: cluster for cluster, cities in CITY_CLUSTERS.items() for city in cities}

LEGACY_MAP = {
    'High Temperature Suppressed': 'Heat-sensitive MSAs',
    'Mid Temperature Stable':      'Temperature-resilient MSAs',
    'Large Metro Baseline':        'Adaptive/baseline MSAs',
}

NAICS_MAP = {
    4451: 'Grocery stores',         6111: 'Educational institutions',
    6221: 'Healthcare facilities',  7121: 'Greenspaces',
    7139: 'Recreational areas',     7225: 'Restaurants',
    8129: 'Service establishments', 8131: 'Religious institutions',
}
NAICS_ORDER = ['Grocery stores', 'Restaurants', 'Healthcare facilities',
               'Educational institutions', 'Greenspaces', 'Recreational areas',
               'Service establishments', 'Religious institutions']

PALETTE = {
    'Heat-sensitive MSAs':        '#FF1E00',
    'Adaptive/baseline MSAs':     '#FFE600',
    'Temperature-resilient MSAs': '#0C57C9',
}

# ── Data processing ──────────────────────────────────────────────────────────
def compute_cluster_visit_change():
    excel_files = [f for f in glob.glob(os.path.join(INPUT_PATH, "*.xlsx"))
                   if not os.path.basename(f).startswith('~$')]
    if not excel_files:
        print(f"No Excel files found in {INPUT_PATH}")
        return None

    all_results = []
    for file_path in excel_files:
        city_name = os.path.basename(file_path).replace('_temp_visits.xlsx', '').replace('_temp_visits', '')
        try:
            df = pd.read_excel(file_path)
            if any(c not in df.columns for c in ['Status', 'naics_code_unique', 'date', 'SUM_visit']):
                continue
            df['SUM_visit'] = pd.to_numeric(df['SUM_visit'], errors='coerce')
            df.dropna(subset=['SUM_visit'], inplace=True)
            if df.empty:
                continue

            daily_stats = []
            for (date, status), group in df.groupby(['date', 'Status']):
                total = group['SUM_visit'].sum()
                if total > 0:
                    for naics, visits in group.groupby('naics_code_unique')['SUM_visit'].sum().items():
                        daily_stats.append({'Status': status, 'naics_code_unique': naics, 'visits': visits})
            if not daily_stats:
                continue

            status_naics = pd.DataFrame(daily_stats).groupby(['Status', 'naics_code_unique'])['visits'].mean().reset_index()
            normal  = status_naics[status_naics['Status'] == 'Normal']
            extreme = status_naics[status_naics['Status'] == 'Extreme']
            change  = normal.merge(extreme, on='naics_code_unique', suffixes=('_n', '_e'), how='outer')
            change['visits_change_pct'] = np.where(
                change['visits_n'] > 0,
                (change['visits_e'].fillna(0) - change['visits_n'].fillna(0)) / change['visits_n'] * 100, 0
            )
            cluster = CITY_TO_CLUSTER.get(city_name, 'Unknown')
            for _, row in change.iterrows():
                all_results.append({'City_Cluster': cluster,
                                    'NAICS_Code': row['naics_code_unique'],
                                    'Visits_Change_Percentage': round(row['visits_change_pct'], 4)})
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_results:
        return None

    df_all = pd.DataFrame(all_results)
    df_all['City_Cluster'] = df_all['City_Cluster'].replace(LEGACY_MAP)

    # Cluster_VisitChange_Summary
    summary = (df_all.groupby(['City_Cluster', 'NAICS_Code'])['Visits_Change_Percentage']
               .mean().reset_index())
    summary['NAICS_Name'] = summary['NAICS_Code'].map(NAICS_MAP)
    return summary

# ── Visualization ─────────────────────────────────────────────────────────────
def plot_visit_changes(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(data=df, x='NAICS_Name', y='Visits_Change_Percentage',
                hue='City_Cluster', palette=PALETTE, order=NAICS_ORDER, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=13)

    ax.set_xlabel('NAICS Category', fontsize=14)
    ax.set_ylabel('Visits Change Percentage (%)', fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='City Cluster', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig 3d. City_Weather_NAICS_Analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = compute_cluster_visit_change()
    if summary is not None:
        plot_visit_changes(summary)
