import os
import glob
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'figure.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

nature_contrast = ['#E64B35', '#4DBBD5', '#3C5488', '#00A087', '#F39B7F']

# ── Paths ──────────────────────────────────────────────────────────────────
input_path  = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Daily_Traffic_ALLCity_Statistics_Temp"
output_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
os.makedirs(output_path, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
all_city_data = {}
for file_path in glob.glob(os.path.join(input_path, "*.xlsx")):
    city_name = os.path.basename(file_path).split('_')[0]
    try:
        df = pd.read_excel(file_path)
        if {'Mean_Temp', 'Essential_visit'}.issubset(df.columns):
            df['Mean_Temp_Rounded'] = np.round(df['Mean_Temp'])
            all_city_data[city_name] = (df.groupby('Mean_Temp_Rounded')['Essential_visit']
                                          .mean().reset_index()
                                          .sort_values('Mean_Temp_Rounded'))
            print(f"Processed {city_name}: {len(all_city_data[city_name])} temperature points")
        else:
            print(f"Warning: {city_name} missing required columns")
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

cities_to_use   = list(all_city_data.keys())[:20]
city_avg_temps  = {c: all_city_data[c]['Mean_Temp_Rounded'].mean() for c in cities_to_use}
sorted_city_names = [c for c, _ in sorted(city_avg_temps.items(), key=lambda x: x[1])]

# ── Figure 1 ───────────────────────────────────────────────────────────────
def create_temperature_quantiles_with_mean():
    set_nature_style()
    plt.figure(figsize=(12, 7))

    temp_range = np.arange(
        np.floor(min(d['Mean_Temp_Rounded'].min() for d in all_city_data.values())),
        np.ceil(max(d['Mean_Temp_Rounded'].max() for d in all_city_data.values())) + 1, 1.0
    )

    visit_matrix = np.full((len(sorted_city_names), len(temp_range)), np.nan)
    for i, city in enumerate(sorted_city_names):
        data = all_city_data[city]
        if len(data) > 0:
            max_v = data['Essential_visit'].max()
            if max_v > 0:
                data = data.copy()
                data['Normalized_Visits'] = data['Essential_visit'] / max_v
                for j, temp in enumerate(temp_range):
                    row = data[data['Mean_Temp_Rounded'] == temp]
                    if not row.empty:
                        visit_matrix[i, j] = row['Normalized_Visits'].values[0]

    mean   = np.nanmean(visit_matrix, axis=0)
    q25    = np.nanquantile(visit_matrix, 0.25, axis=0)
    q75    = np.nanquantile(visit_matrix, 0.75, axis=0)
    valid  = (np.sum(~np.isnan(visit_matrix), axis=0) >= 3) & ~np.isnan(mean)

    if valid.sum() <= 5:
        print("Not enough valid data points.")
        return

    tv, mv, q25v, q75v = temp_range[valid], mean[valid], q25[valid], q75[valid]
    tf = tv >= 10
    tv, mv, q25v, q75v = tv[tf], mv[tf], q25v[tf], q75v[tf]

    win = min(7, len(mv) - 2 if len(mv) % 2 == 0 else len(mv) - 1)
    if len(mv) > win:
        ms   = np.clip(savgol_filter(mv,   win, 2), 0, 1)
        q25s = np.clip(savgol_filter(q25v, win, 2), 0, 1)
        q75s = np.clip(savgol_filter(q75v, win, 2), 0, 1)
    else:
        ms, q25s, q75s = mv, q25v, q75v

    min_city_line_temp = 12.5
    if tv[0] > min_city_line_temp:
        nearby = []
        for city in sorted_city_names:
            d = all_city_data[city]
            nb = d[(d['Mean_Temp_Rounded'] >= min_city_line_temp - 1) &
                   (d['Mean_Temp_Rounded'] <= min_city_line_temp + 1)]
            if not nb.empty:
                nearby.extend(nb['Essential_visit'].values / d['Essential_visit'].max())

        if nearby:
            t_mean = np.mean(nearby)
            t_q25  = np.percentile(nearby, 25) if len(nearby) >= 4 else np.clip(t_mean - (ms[0] - q25s[0]), 0.6, 1.0)
            t_q75  = np.percentile(nearby, 75) if len(nearby) >= 4 else np.clip(t_mean + (q75s[0] - ms[0]),  t_q25, 1.0)
        else:
            slope  = (ms[1] - ms[0]) / (tv[1] - tv[0]) if len(tv) >= 2 else 0
            t_mean = np.clip(ms[0] - slope * (tv[0] - min_city_line_temp), 0.7, 1.0)
            t_q25  = np.clip(t_mean - (ms[0] - q25s[0]), 0.6, 1.0)
            t_q75  = np.clip(t_mean + (q75s[0] - ms[0]),  t_q25, 1.0)

        tv   = np.insert(tv,   0, min_city_line_temp)
        ms   = np.insert(ms,   0, t_mean)
        q25s = np.insert(q25s, 0, t_q25)
        q75s = np.insert(q75s, 0, t_q75)

    plt.subplots_adjust(right=0.75)
    plt.xlim(10, 43)
    plt.ylim(0.6, 1.02)

    # Individual city scatter
    for city in sorted_city_names:
        data = all_city_data[city]
        if len(data) > 0:
            df = data[(data['Mean_Temp_Rounded'] >= 10) & (data['Mean_Temp_Rounded'] <= 43)]
            if not df.empty:
                plt.plot(df['Mean_Temp_Rounded'],
                         df['Essential_visit'] / data['Essential_visit'].max(),
                         'o', color='lightgray', markersize=1.5, alpha=0.3)

    mask = tv <= 43
    plt.plot(tv[mask], ms[mask], linewidth=2.5, color=nature_contrast[2], label='Mean')
    plt.fill_between(tv[mask], q25s[mask], q75s[mask],
                     color=nature_contrast[2], alpha=0.3,
                     label='Interquartile range (25th–75th percentile)')

    # City labels & binned lines
    city_labels    = {c: string.ascii_uppercase[i % 26] for i, c in enumerate(sorted_city_names)}
    city_line_data = {}
    for city in sorted_city_names:
        data = all_city_data[city]
        if len(data) <= 5:
            continue
        df = data[(data['Mean_Temp_Rounded'] >= 10) & (data['Mean_Temp_Rounded'] <= 43)]
        if df.empty:
            continue
        x = df['Mean_Temp_Rounded'].values
        bin_size = 5
        bins = np.arange(max(10, int(np.floor(x.min() / bin_size)) * bin_size),
                         min(43, int(np.ceil(x.max() / bin_size)) * bin_size) + bin_size, bin_size)
        bx, by = [], []
        for j in range(len(bins) - 1):
            mask2 = (x >= bins[j]) & (x < bins[j+1])
            if mask2.any():
                bx.append((bins[j] + bins[j+1]) / 2)
                by.append(df['Essential_visit'].values[mask2].mean() / data['Essential_visit'].max())
        if bx:
            plt.plot(bx, by, 'o-', color='gray', markersize=3, linewidth=0.8, alpha=0.5)
            city_line_data[city] = (bx, by)

    # Label placement
    grid_x, grid_y = 6, 5
    x_min, x_max, y_min, y_max = 10, 43, 0.6, 1.0
    cw = (x_max - x_min) / grid_x
    ch = (y_max - y_min) / grid_y
    occupied = np.zeros((grid_y, grid_x), dtype=bool)

    for city, (bx, by) in city_line_data.items():
        valid_pts = [(x, y) for x, y in zip(bx, by) if y_min <= y <= y_max]
        if not valid_pts:
            continue
        cx_mean, cy_mean = np.mean(bx), np.mean(by)
        best, best_score = None, float('inf')
        for x, y in valid_pts:
            gi = min(grid_y-1, int((y - y_min) / ch))
            gj = min(grid_x-1, int((x - x_min) / cw))
            if not occupied[gi, gj]:
                score = np.sqrt((x - cx_mean)**2 + (y - cy_mean)**2)
                if score < best_score:
                    best_score, best = score, (x, y, gi, gj)

        label_kw = dict(fontsize=10, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black',
                                  pad=0.3, boxstyle='circle'),
                        ha='center', va='center')
        if best:
            x, y, gi, gj = best
            occupied[gi, gj] = True
            plt.text(x, y, city_labels[city], **label_kw)
        else:
            valid_pts.sort(key=lambda p: p[0])
            mx, my = valid_pts[len(valid_pts)//2]
            gi = min(grid_y-1, int((my - y_min) / ch))
            gj = min(grid_x-1, int((mx - x_min) / cw))
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = gi+di, gj+dj
                    if 0 <= ni < grid_y and 0 <= nj < grid_x and not occupied[ni, nj]:
                        occupied[ni, nj] = True
                        lx = x_min + (nj + 0.5) * cw
                        ly = y_min + (ni + 0.5) * ch
                        plt.text(lx, ly, city_labels[city], **label_kw)
                        nn = min(valid_pts, key=lambda p: (p[0]-lx)**2 + (p[1]-ly)**2)
                        plt.plot([lx, nn[0]], [ly, nn[1]], '--', color='gray',
                                 linewidth=0.8, alpha=0.6, dashes=(2, 2))
                        break
                else:
                    continue
                break

    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray', which='both')
    plt.minorticks_on()
    plt.xlabel('Temperature (°C)', fontsize=15)
    plt.ylabel('Normalized Visit Frequency', fontsize=15)

    plt.legend(handles=[
        Line2D([0], [0], color=nature_contrast[2], linewidth=2.5, label='Mean'),
        Patch(facecolor=nature_contrast[2], alpha=0.3, label='Interquartile range (25th–75th percentile)'),
        Line2D([0], [0], marker='o', color='gray', linewidth=0.8, markersize=3, alpha=0.5, label='City 5°C bin averages'),
    ], frameon=False, loc='lower right', fontsize=12)

    city_legend = [Line2D([0], [0], color='white', linewidth=0, label=f"{city_labels[c]}: {c}")
                   for c in sorted_city_names if c in city_line_data]
    if city_legend:
        h = min(0.9, len(city_legend) * 0.025)
        ax2 = plt.gcf().add_axes([0.78, 0.5 - h/2, 0.2, h])
        ax2.axis('off')
        ax2.legend(handles=city_legend, loc='center', ncol=1, fontsize=13,
                   frameon=True, title="City Legend")

    plt.yticks(np.arange(0.6, 1.05, 0.1), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    out = os.path.join(output_path, "S2.Temperature_Response_Mean.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")


try:
    create_temperature_quantiles_with_mean()
except Exception as e:
    print(f"Error: {e}")
