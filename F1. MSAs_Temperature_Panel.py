import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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

cities_to_use  = list(all_city_data.keys())[:20]
city_avg_temps = {c: all_city_data[c]['Mean_Temp_Rounded'].mean() for c in cities_to_use}
sorted_city_names = [c for c, _ in sorted(city_avg_temps.items(), key=lambda x: x[1])]

# ── Figure 2 ───────────────────────────────────────────────────────────────
city_table_avg_temps = {
    'Austin': 34.08, 'Charlotte': 30.18, 'Chicago': 27.11, 'Columbus': 28.44,
    'Dallas': 33.64, 'Denver': 26.53, 'Houston': 32.85, 'Indianapolis': 27.94,
    'Jacksonville': 32.43, 'LA': 28.13, 'Nashville': 31.43, 'New York': 26.90,
    'Oklahoma': 32.60, 'Philadelphia': 28.49, 'Phoenix': 40.29, 'SanAntonio': 35.15,
    'SanDiego': 25.39, 'SanFrancisco': 25.46, 'SanJose': 27.24, 'Seattle': 23.65,
}

def create_city_panel_visualization():
    set_nature_style()
    n_cities = min(20, len(sorted_city_names))
    n_cols   = min(4, n_cities)
    n_rows   = int(np.ceil(n_cities / n_cols))

    min_t = min(city_table_avg_temps.values())
    max_t = max(city_table_avg_temps.values())
    norm  = plt.Normalize(min_t, max_t)

    plt.figure(figsize=(12, 10))

    for i, city in enumerate(sorted_city_names[:n_cities]):
        ax   = plt.subplot(n_rows, n_cols, i + 1)
        data = all_city_data[city]
        avg_temp = city_table_avg_temps.get(city, city_avg_temps.get(city, 0))
        color    = plt.cm.viridis(norm(avg_temp))

        if len(data) > 0:
            data = data.copy()
            data['Normalized_Visits'] = data['Essential_visit'] / data['Essential_visit'].max()
            data = data.sort_values('Mean_Temp_Rounded')
            x, y = data['Mean_Temp_Rounded'].values, data['Normalized_Visits'].values

            ax.fill_between(x, 0.5, y, color=color, alpha=0.2)

            if len(x) > 5 and len(np.unique(x)) > 3:
                try:
                    xs = np.linspace(x.min(), x.max(), 100)
                    ys = np.clip(make_interp_spline(x, y, k=min(3, len(np.unique(x))-1))(xs), 0.5, 1)
                    ax.plot(xs, ys, '-', color=color, linewidth=1.5)
                except Exception:
                    ax.plot(x, y, '-', color=color, linewidth=1.5)
            else:
                ax.plot(x, y, '-', color=color, linewidth=1.5)

            bin_size = 5
            bins = np.arange(int(np.floor(x.min() / bin_size)) * bin_size,
                             int(np.ceil(x.max() / bin_size)) * bin_size + bin_size, bin_size)
            bx = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins)-1)
                  if ((x >= bins[j]) & (x < bins[j+1])).any()]
            by = [y[(x >= bins[j]) & (x < bins[j+1])].mean() for j in range(len(bins)-1)
                  if ((x >= bins[j]) & (x < bins[j+1])).any()]
            if bx:
                ax.plot(bx, by, 'o-', color='black', markersize=3, linewidth=1, alpha=0.8)

        ax.set_title(f"{city} ({avg_temp:.2f}°C)", fontsize=10)
        ax.set_xlim(10, 45)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        if i % n_cols == 0:
            ax.set_ylabel('Normalized Visits', fontsize=10)
        else:
            ax.set_yticklabels([])
        if i >= n_cities - n_cols:
            ax.set_xlabel('Temperature (°C)', fontsize=10)

    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                 cax=cbar_ax).set_label('City Average Temperature (°C)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    out = os.path.join(output_path, "Fig1. City_Panel_Visualization.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")


try:
    create_city_panel_visualization()
except Exception as e:
    print(f"Error: {e}")
