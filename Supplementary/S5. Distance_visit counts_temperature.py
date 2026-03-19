import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np

# 1. File paths (Input & Output)
input_csv = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\CBG_ALLcity_Week_NAICS.csv"
output_dir = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
os.makedirs(output_dir, exist_ok=True)

# 2. NAICS code to name mapping
naics_mapping = {
    7225: 'Restaurants',
    8129: 'Service establishments',
    8131: 'Religious institutions',
    4451: 'Grocery stores',
    7139: 'Recreational areas',
    6221: 'Healthcare facilities',
    7121: 'Greenspaces',
    6111: 'Educational institutions'
}

# 3. Read and clean data
df = pd.read_csv(input_csv)
df = df.dropna(subset=[
    'Temperature',
    'cbgs_visitor_7day_avg',
    'distance_from_home_7day_avg',
    'naics_code_unique'
])
df['NAICS_name'] = df['naics_code_unique'].map(naics_mapping)

# 4. Create temperature bins (10 equal-width bins), format labels to 1 decimal place
df['Temp_bins'] = pd.cut(df['Temperature'], bins=10) \
    .apply(lambda x: f"({x.left:.1f}, {x.right:.1f}]")

# 5. Configure a more vibrant Nature-style color scheme
sns.set_style("whitegrid")
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['grid.alpha'] = 0.3  # Faint grid lines to make colors pop

# Ultra-high saturation Nature-style color palette
enhanced_nature_palette = [
    '#0072B2',  # Vibrant blue
    '#FF2222',  # Vibrant red
    '#009E73',  # Vibrant green
    '#E69F00',  # Vibrant orange
    '#CC79A7',  # Vibrant pink
    '#56B4E9',  # Vibrant sky blue
    '#D55E00',  # Vibrant deep orange
    '#9467BD',  # Vibrant purple
]

# 6. Define outlier style (small dots) - remove outline, use solid color points
flierprops = dict(
    marker='o',
    markerfacecolor='#555555',
    markeredgecolor='none',  # Remove point outline
    markersize=3,
    linestyle='none',
    alpha=0.7
)

# 7. Plot & Save -- Average Visitors Boxplot
plt.figure(figsize=(14, 6), dpi=100, facecolor='white')
ax = sns.boxplot(
    data=df,
    x='Temp_bins',
    y=df['cbgs_visitor_7day_avg'] / 1e5,
    hue='NAICS_name',
    palette=enhanced_nature_palette,
    flierprops=flierprops,
    linewidth=0.6,  # Thinner box borders
    saturation=1.0,  # Maximum saturation
    width=0.7        # Slightly wider boxes
)

# Enhance chart style
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel("Temperature Bins (°C)", fontsize=12, fontweight='bold')
plt.ylabel("Average Visitors (×10⁵)", fontsize=12, fontweight='bold')
plt.title("Visitor Count by Temperature and NAICS Category", fontsize=14, fontweight='bold')

# Add light background to axes to make colors pop
ax.set_facecolor('#F8F8F8')

# Optimize legend
legend = plt.legend(title="NAICS Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('#333333')
legend.get_title().set_fontweight('bold')

# Save high-quality chart
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "S5a.Boxplot_Visitors.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8. Plot & Save -- Average Travel Distance Boxplot
plt.figure(figsize=(14, 6), dpi=100, facecolor='white')
ax = sns.boxplot(
    data=df,
    x='Temp_bins',
    y=df['distance_from_home_7day_avg'] / 1e5,
    hue='NAICS_name',
    palette=enhanced_nature_palette,
    flierprops=flierprops,
    linewidth=0.6,  # Thinner box borders
    saturation=1.0,  # Maximum saturation
    width=0.7        # Slightly wider boxes
)

# Enhance chart style
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel("Temperature Bins (°C)", fontsize=12, fontweight='bold')
plt.ylabel("Average Distance (km ×10⁵)", fontsize=12, fontweight='bold')
plt.title("Travel Distance by Temperature and NAICS Category", fontsize=14, fontweight='bold')

# Add light background to axes to make colors pop
ax.set_facecolor('#F8F8F8')

# Optimize legend
legend = plt.legend(title="NAICS Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('#333333')
legend.get_title().set_fontweight('bold')

# Add clearer legend background color to enhance contrast
legend.get_frame().set_facecolor('#F8F8F8')

# Save high-quality chart
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "S5a. Boxplot_Distance.png"), dpi=300, bbox_inches='tight')
plt.close()