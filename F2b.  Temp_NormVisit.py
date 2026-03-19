import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

# ── Style ──────────────────────────────────────────────────────────────────
def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

# ── Config ─────────────────────────────────────────────────────────────────
input_path  = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Daily_Traffic_ALLCity_Statistics_Temp"
output_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
os.makedirs(output_path, exist_ok=True)

city_clusters = {
    'Heat-Sensitive MSAs':        ['Chicago', 'Dallas', 'Houston'],
    'Temperature-Resilient MSAs': ['LA', 'Phoenix', 'NewYork'],
    'Adaptive/Baseline MSAs':     ['SanFrancisco', 'SanDiego', 'SanAntonio', 'Philadelphia',
                                     'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
                                     'Indianapolis', 'Denver', 'Columbus', 'Charlotte',
                                     'Nashville', 'Seattle'],
}
cluster_colors = {
    'Heat-Sensitive MSAs':        '#E64B35',
    'Temperature-Resilient MSAs': '#00A087',
    'Adaptive/Baseline MSAs':     '#0053A0',
}
MANUAL_OFFSETS = {
    'SanJose':       {'x':  1.0, 'y':  0.0},
    'Columbus':      {'x': -1.0, 'y':  0.0},
    'Austin':        {'x': -1.0, 'y': -0.005},
    'SanFrancisco':  {'x':  1.0, 'y':  0.0},
    'SanDiego':      {'x': -1.0, 'y':  0.0},
}

# ── Load data ──────────────────────────────────────────────────────────────
all_city_data = {}
for fp in glob.glob(os.path.join(input_path, "*.xlsx")):
    city = os.path.basename(fp).split('_')[0]
    try:
        df = pd.read_excel(fp)
        if {'Mean_Temp', 'Essential_visit'}.issubset(df.columns):
            df['Mean_Temp_Rounded'] = np.round(df['Mean_Temp'])
            all_city_data[city] = (df.groupby('Mean_Temp_Rounded')['Essential_visit']
                                     .mean().reset_index()
                                     .sort_values('Mean_Temp_Rounded'))
            print(f"Processed {city}: {len(all_city_data[city])} temperature points")
        else:
            print(f"Warning: {city} missing required columns")
    except Exception as e:
        print(f"Error processing {city}: {e}")

set_nature_style()

# ── Helpers ────────────────────────────────────────────────────────────────
def smooth(arr, win):
    if len(arr) > win:
        return np.clip(savgol_filter(arr, win, 2), 0, 1)
    return arr

def extend_to_12_5(temp_valid, mean_s, q25_s, q75_s, cluster_cities, min_t=12.5):
    if temp_valid[0] <= min_t:
        return temp_valid, mean_s, q25_s, q75_s
    nearby = []
    for city in cluster_cities:
        if city in all_city_data:
            d = all_city_data[city]
            nb = d[(d['Mean_Temp_Rounded'] >= min_t - 1) & (d['Mean_Temp_Rounded'] <= min_t + 1)]
            if not nb.empty:
                nearby.extend(nb['Essential_visit'].values / d['Essential_visit'].max())
    if nearby:
        tm = np.mean(nearby)
        tq25 = np.percentile(nearby, 25) if len(nearby) >= 4 else tm * 0.9
        tq75 = np.percentile(nearby, 75) if len(nearby) >= 4 else tm * 1.1
    elif len(temp_valid) >= 2:
        slope = (mean_s[1] - mean_s[0]) / (temp_valid[1] - temp_valid[0])
        tm = np.clip(mean_s[0] - slope * (temp_valid[0] - min_t), 0.7, 1.0)
        tq25, tq75 = tm * 0.9, tm * 1.1
    else:
        tm, tq25, tq75 = mean_s[0], q25_s[0], q75_s[0]
    return (np.insert(temp_valid, 0, min_t), np.insert(mean_s, 0, tm),
            np.insert(q25_s, 0, tq25), np.insert(q75_s, 0, tq75))

def bin_city(data, bin_size=5):
    x = data['Mean_Temp_Rounded'].values
    df = data[(data['Mean_Temp_Rounded'] >= 10) & (data['Mean_Temp_Rounded'] <= 43)]
    if df.empty:
        return [], []
    x = df['Mean_Temp_Rounded'].values
    bins = np.arange(max(10, int(np.floor(x.min() / bin_size)) * bin_size),
                     min(43, int(np.ceil(x.max() / bin_size)) * bin_size) + bin_size, bin_size)
    bx = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins)-1)
          if ((x >= bins[j]) & (x < bins[j+1])).any()]
    by = [df['Essential_visit'].values[(x >= bins[j]) & (x < bins[j+1])].mean()
          / data['Essential_visit'].max() for j in range(len(bins)-1)
          if ((x >= bins[j]) & (x < bins[j+1])).any()]
    return bx, by

# ── Main plot ──────────────────────────────────────────────────────────────
def create_temperature_quantiles_with_mean():
    plt.figure(figsize=(12, 8))
    plt.xlim(10, 43)
    plt.ylim(0.6, 1.02)

    temp_range = np.arange(
        np.floor(min(d['Mean_Temp_Rounded'].min() for d in all_city_data.values())),
        np.ceil(max(d['Mean_Temp_Rounded'].max() for d in all_city_data.values())) + 1, 1.0
    )

    # Background scatter per city
    for cluster_name, cities in city_clusters.items():
        for city in cities:
            if city not in all_city_data:
                continue
            d = all_city_data[city]
            mask = (d['Mean_Temp_Rounded'] >= 10) & (d['Mean_Temp_Rounded'] <= 43)
            if mask.any():
                plt.plot(d.loc[mask, 'Mean_Temp_Rounded'],
                         d.loc[mask, 'Essential_visit'] / d['Essential_visit'].max(),
                         'o', color=cluster_colors[cluster_name], markersize=1.5, alpha=0.15)

    # Cluster trend lines
    for cluster_name, cities in city_clusters.items():
        valid_cities = [c for c in cities if c in all_city_data]
        if not valid_cities:
            continue
        mat = np.full((len(valid_cities), len(temp_range)), np.nan)
        for i, city in enumerate(valid_cities):
            d = all_city_data[city].copy()
            max_v = d['Essential_visit'].max()
            if max_v > 0:
                d['Normalized_Visits'] = d['Essential_visit'] / max_v
                for j, t in enumerate(temp_range):
                    row = d[d['Mean_Temp_Rounded'] == t]
                    if not row.empty:
                        mat[i, j] = row['Normalized_Visits'].values[0]

        cmean = np.nanmean(mat, axis=0)
        cq25  = np.nanquantile(mat, 0.25, axis=0)
        cq75  = np.nanquantile(mat, 0.75, axis=0)
        valid = (np.sum(~np.isnan(mat), axis=0) >= 2) & ~np.isnan(cmean)

        if valid.sum() <= 3:
            continue
        tv, mv, q25v, q75v = temp_range[valid], cmean[valid], cq25[valid], cq75[valid]
        tf = tv >= 10
        tv, mv, q25v, q75v = tv[tf], mv[tf], q25v[tf], q75v[tf]

        win = min(7, len(mv) - 2 if len(mv) % 2 == 0 else len(mv) - 1)
        ms, q25s, q75s = smooth(mv, win), smooth(q25v, win), smooth(q75v, win)
        tv, ms, q25s, q75s = extend_to_12_5(tv, ms, q25s, q75s, cities)

        mask43 = tv <= 43
        plt.plot(tv[mask43], ms[mask43], linewidth=3, color=cluster_colors[cluster_name],
                 label=f'{cluster_name} (n={len(valid_cities)})')
        plt.fill_between(tv[mask43], q25s[mask43], q75s[mask43],
                         color=cluster_colors[cluster_name], alpha=0.2)

    # City labels and binned lines
    city_labels = {}
    cluster_city_data = {}
    for cluster_name, cities in city_clusters.items():
        prefix = cluster_name[0]
        idx = 0
        cluster_city_data[cluster_name] = {}
        for city in cities:
            if city in all_city_data:
                city_labels[city] = f"{prefix}{idx+1}"
                idx += 1
                bx, by = bin_city(all_city_data[city])
                if bx:
                    plt.plot(bx, by, 'o-', color=cluster_colors[cluster_name],
                             markersize=2.5, linewidth=0.8, alpha=0.4)
                    cluster_city_data[cluster_name][city] = (bx, by)

    # Label placement
    total_labels = sum(len([c for c in v if c in all_city_data]) for v in city_clusters.values())
    label_x_positions = np.linspace(11.0, 42.0, total_labels)
    label_index = 0
    label_assignments = {}
    for cities in city_clusters.values():
        for city in cities:
            if city in all_city_data:
                label_assignments[city] = label_x_positions[label_index]
                label_index += 1

    for cluster_name, cities in city_clusters.items():
        for city in cities:
            if city not in all_city_data or city not in cluster_city_data[cluster_name]:
                continue
            bx, by = cluster_city_data[cluster_name][city]
            if not bx:
                continue
            valid_pts = [(x, y) for x, y in zip(bx, by) if 0.6 <= y <= 1.0]
            if not valid_pts:
                continue
            ax = label_assignments[city]
            x, y = min(valid_pts, key=lambda p: abs(p[0] - ax))
            if city in MANUAL_OFFSETS:
                x = np.clip(x + MANUAL_OFFSETS[city]['x'], 10.5, 42.5)
                y = np.clip(y + MANUAL_OFFSETS[city]['y'], 0.62, 0.98)
            plt.text(x, y, city_labels[city], fontsize=8, fontweight='bold', color='white',
                     bbox=dict(facecolor=cluster_colors[cluster_name], alpha=0.9,
                               edgecolor='none', pad=0.8, boxstyle='circle'),
                     ha='center', va='center', zorder=10)

    # Axes and legend
    plt.minorticks_on()
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray', which='both')
    plt.xlabel('Temperature ($^\\circ$C)', fontsize=10)
    plt.ylabel('Normalized Visits', fontsize=10)
    plt.yticks(np.arange(0.6, 1.05, 0.1))

    legend_elements = [
        Line2D([0], [0], color=cluster_colors[cn], linewidth=3,
               label=f'{cn} (n={len([c for c in city_clusters[cn] if c in all_city_data])})')
        for cn in city_clusters
    ] + [
        Patch(facecolor='gray', alpha=0.2, label='Interquartile range'),
        Line2D([0], [0], marker='o', color='gray', linewidth=0.8,
               markersize=2.5, alpha=0.4, label='Individual city 5°C bin averages'),
    ]

    city_legend = [Line2D([0], [0], color='white', linewidth=0,
                          label=f"{city_labels[city]}: {city}")
                   for cities in city_clusters.values()
                   for city in cities if city in all_city_data and city in city_labels]
    while len(city_legend) < 20:
        city_legend.append(Line2D([0], [0], color='white', linewidth=0, label=''))

    cl = plt.figlegend(handles=city_legend, ncol=5, fontsize=10, frameon=True,
                       title="City Legend", title_fontsize=11,
                       columnspacing=0.5, handletextpad=0.3,
                       loc='upper left', bbox_to_anchor=(0.15, 0.23))
    cl.get_title().set_fontweight('bold')

    plt.figlegend(handles=legend_elements, frameon=False,
                  loc='upper right', bbox_to_anchor=(0.98, 0.23), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    out = os.path.join(output_path, "Fig 2b. Visit_Trend_MSAs.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")


try:
    create_temperature_quantiles_with_mean()
except Exception as e:
    print(f"Error: {e}")
