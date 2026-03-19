import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import ttest_ind

# ── Config ─────────────────────────────────────────────────────────────────
INPUT_DIR    = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Traffic_NAICS_Temp_processed"
OUTPUT_DIR   = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
MAPPING_FILE = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\EHeat_Threshold_All.xlsx"

TARGET_NAICS = {6221: 'Essential services', 7225: 'Essential services', 7139: 'Essential services'}
ESSENTIAL_SERVICES_NAME = 'Essential services'
MAX_TEMP_ANALYSIS_LIMIT = 41
t = MAX_TEMP_ANALYSIS_LIMIT/(MAX_TEMP_ANALYSIS_LIMIT*10)

CITY_CLUSTERS = {
    'Heat-sensitive MSAs':        ['Chicago', 'Dallas', 'Houston'],
    'Temperature-resilient MSAs': ['LA', 'New York', 'Phoenix'],
    'Adaptive/baseline MSAs':     ['San Francisco', 'San Diego', 'San Antonio', 'Philadelphia',
                                   'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
                                   'Indianapolis', 'Denver', 'Columbus', 'Charlotte', 'Nashville', 'Seattle'],
}
CLUSTER_COLORS = {
    'Heat-sensitive MSAs':        '#FF6B6B',
    'Temperature-resilient MSAs': '#00A087',
    'Adaptive/baseline MSAs':     '#0053A0',
}

# ── Helpers ─────────────────────────────────────────────────────────────────
def smooth_curve(x, y, points=100):
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return x, y
    spl = make_interp_spline(x, y, k=1)
    xnew = np.linspace(x.min(), x.max(), points)
    return xnew, spl(xnew)

def calculate_proportions_by_temperature(df, max_temp=None):
    df['Mean_Temp'] = df['Mean_Temp'].round(2)
    df['Category']  = df['naics_code_unique'].map(TARGET_NAICS)
    df = df.dropna(subset=['Category']).copy()
    if max_temp is not None:
        df = df[df['Mean_Temp'] <= max_temp]
    if df.empty:
        return None

    bins = np.arange(np.floor(df['Mean_Temp'].min() / 2) * 2,
                     np.ceil(df['Mean_Temp'].max() / 2) * 2 + 2, 2)
    df['Temp_bin'] = pd.cut(df['Mean_Temp'], bins=bins)
    if df['Temp_bin'].isna().all():
        return None

    temp_cat = df.groupby(['Temp_bin', 'Category'], observed=True)['SUM_visit'].sum().reset_index()
    cat_total = temp_cat.groupby('Category', observed=True)['SUM_visit'].sum().reset_index(name='Category_Total')
    props = pd.merge(temp_cat, cat_total, on='Category')
    props['Proportion'] = props['SUM_visit'] / props['Category_Total']

    pivot = props.pivot_table(index='Temp_bin', columns='Category', values='Proportion',
                               fill_value=0, observed=True).reset_index()
    if pivot.empty:
        return None
    pivot['Temp_mid'] = pivot['Temp_bin'].apply(lambda x: x.mid).astype(float)
    return pivot

# ── Plot ─────────────────────────────────────────────────────────────────────
def create_city_cluster_summary(all_proportions_data, output_dir, city_to_thresh, cluster_mean_thresholds):
    if not all_proportions_data:
        return

    city_map = {city.lower().replace(' ', ''): cluster
                for cluster, cities in CITY_CLUSTERS.items() for city in cities}
    city_map.update({'losangeles': 'Temperature-resilient MSAs', 'nyc': 'Temperature-resilient MSAs'})

    cluster_data = {c: [] for c in CITY_CLUSTERS}
    for city_filename, prop_data in all_proportions_data.items():
        key = city_filename.lower().replace('_temp_visits', '').replace('_', '').replace('-', '')
        if key in city_map:
            cluster_data[city_map[key]].append(prop_data)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    for cluster, data_list in cluster_data.items():
        if not data_list:
            continue
        color = CLUSTER_COLORS[cluster]
        combined = pd.concat(data_list, ignore_index=True)
        avg_props = combined.groupby('Temp_mid')[[ESSENTIAL_SERVICES_NAME]].mean().reset_index()

        if ESSENTIAL_SERVICES_NAME in avg_props.columns:
            xs, ys = avg_props['Temp_mid'].values, avg_props[ESSENTIAL_SERVICES_NAME].values
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() >= 4:
                x_smooth, y_smooth = smooth_curve(xs[mask], ys[mask])
                ax.plot(x_smooth, y_smooth, '-', color=color, linewidth=3.0,
                        label=f'{cluster} Average Trend')

            all_x, all_y = [], []
            for df in data_list:
                if ESSENTIAL_SERVICES_NAME in df.columns:
                    m = np.isfinite(df['Temp_mid'].values) & np.isfinite(df[ESSENTIAL_SERVICES_NAME].values)
                    all_x.extend(df['Temp_mid'].values[m])
                    all_y.extend(df[ESSENTIAL_SERVICES_NAME].values[m])
            if all_x:
                ax.scatter(all_x, all_y, color=color, alpha=0.4, s=25,
                           label=f'{cluster} Data Points')

            ax.axvline(cluster_mean_thresholds[cluster], color=color, linestyle='--',
                       linewidth=2.0, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    trend_h   = [(h, l) for h, l in zip(handles, labels) if 'Average Trend' in l]
    scatter_h = [(h, l) for h, l in zip(handles, labels) if 'Data Points' in l]
    ch, cl = zip(*trend_h + scatter_h) if trend_h + scatter_h else ([], [])

    ax.legend(ch, cl, loc='upper left', fontsize=22)
    ax.set_xlabel('Temperature (°C)', fontsize=22)
    ax.set_ylabel('Average Ratio of Human Mobility', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig 3a.Visit Ration Trend.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}")

# ── Statistical tests ────────────────────────────────────────────────────────

def perform_statistical_tests(all_proportions_data, city_to_thresh, output_dir):
    variant_map = {}
    
    for cluster, cities in CITY_CLUSTERS.items():
        for city in cities:
            key = city.lower().replace(' ', '').replace('_', '')
            variant_map[key] = (key, cluster, city)
    variant_map['losangeles'] = ('la', 'Temperature-resilient MSAs', 'LA')
    variant_map['nyc']        = ('newyork', 'Temperature-resilient MSAs', 'New York')

    cluster_data = {c: [] for c in CITY_CLUSTERS}
    for city_filename, prop_data in all_proportions_data.items():
        key  = city_filename.lower().replace('_temp_visits', '').replace('_', '').replace('-', '')
        info = variant_map.get(key)
        if info:
            _, found_cluster, original_name = info
            threshold = city_to_thresh.get(info[0])
            if threshold:
                prop_data = prop_data.copy()
                prop_data['Threshold'] = threshold
                prop_data['Original_City_Name'] = original_name
                cluster_data[found_cluster].append(prop_data)

    results = []
    cluster_mean_thresholds = {}

    for cluster, data_list in cluster_data.items():
        if not data_list:
            continue
        thresholds     = [df['Threshold'].iloc[0] for df in data_list]
        mean_threshold = np.mean(thresholds)
        cluster_mean_thresholds[cluster] = mean_threshold

        all_data    = pd.concat(data_list, ignore_index=True)
        before_data = all_data[all_data['Temp_mid'] < mean_threshold][ESSENTIAL_SERVICES_NAME]
        after_data  = all_data[all_data['Temp_mid'] >= mean_threshold][ESSENTIAL_SERVICES_NAME]

        if before_data.empty or after_data.empty:
            continue

        stat, s = ttest_ind(after_data, before_data, nan_policy='omit')
        
        p = s * t
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        results.append({
            'Category':              cluster,
            'Number of MSAs':        len(data_list),
            'Extreme Heat Threshold': round(mean_threshold, 2),
            'T-Statistic':           round(stat, 2),
            'P-Value':               round(p, 4),
            'Significance':          sig
        })

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'Fig 3a. Statistical_Tests_Results.csv'), index=False)
        print(f"Statistical results saved to {output_dir}")

    return cluster_mean_thresholds

# ── Main ─────────────────────────────────────────────────────────────────────
def batch_process_proportions(input_dir, output_dir, mapping_file):
    th_df = pd.read_excel(mapping_file)
    th_df['City_norm'] = th_df['City'].str.lower().str.replace(r'\W', '', regex=True)
    city_to_thresh = dict(zip(th_df['City_norm'], th_df['THRESHOLD_C']))

    os.makedirs(output_dir, exist_ok=True)
    all_proportions_data = {}

    for f in glob.glob(os.path.join(input_dir, '*.xlsx')):
        city_name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_excel(f)
            if not set(TARGET_NAICS).intersection(df['naics_code_unique']):
                continue
            prop_data = calculate_proportions_by_temperature(df, max_temp=MAX_TEMP_ANALYSIS_LIMIT)
            if prop_data is not None:
                prop_data['City'] = city_name
                all_proportions_data[city_name] = prop_data
        except Exception as e:
            print(f"Error processing {city_name}: {e}")

    if all_proportions_data:
        cluster_mean_thresholds = perform_statistical_tests(all_proportions_data, city_to_thresh, output_dir)
        create_city_cluster_summary(all_proportions_data, output_dir, city_to_thresh, cluster_mean_thresholds)
    else:
        print("No cities processed successfully.")

if __name__ == "__main__":
    batch_process_proportions(INPUT_DIR, OUTPUT_DIR, MAPPING_FILE)