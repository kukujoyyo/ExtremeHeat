import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Read data from Excel file
input_file = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Classified_city_metrics.xlsx'
output_dir = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code'


df = pd.read_excel(input_file)

# Rename columns to match the code requirements
df = df.rename(columns={
    'H_o': 'H_o',
    'φ': 'phi',
    'node_density_per_km2': 'Node_Density',
    'avg_clustering_coeff': 'Clustering_Coefficient'
})

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Probability Density Distributions by Category', fontsize=16, fontweight='bold')

# Define colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, cyan-green, blue
categories = df['category'].unique()

# Variable list and corresponding x-axis labels
variables = ['H_o', 'phi', 'Node_Density', 'Clustering_Coefficient']
x_labels = ['Ho (Entropy)', ' φ (Orientation-Order Measure)', ' N (Node Density)', 'C (Clustering Coefficient)']

# Plot each subplot
for idx, (var, x_label) in enumerate(zip(variables, x_labels)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Collect data from all categories to determine x-axis range
    all_data = []
    kde_list = []
    
    for i, category in enumerate(categories):
        data_subset = df[df['category'] == category][var]
        all_data.extend(data_subset.values)
        kde_list.append((stats.gaussian_kde(data_subset), category, i))
    
    # Determine x-axis range based on all data
    data_min = min(all_data)
    data_max = max(all_data)
    data_range = data_max - data_min
    
    # Calculate initial x-axis range - expand the range
    x_min_initial = data_min - 1.2 * data_range
    x_max_initial = data_max + 1.2 * data_range
    x_range_initial = np.linspace(x_min_initial, x_max_initial, 600)
    
    # If x-axis range contains negative values, apply compression
    if x_min_initial < 0:
        # Compression method: shift negative values to above 0
        shift = abs(x_min_initial)
        x_range = x_range_initial + shift
        
        # Readjust range to keep data in reasonable position
        # Find where the actual data range is in the new coordinate system
        data_start_new = data_min + shift
        data_end_new = data_max + shift
        
        # Optimize display range: start from 0, extend to larger range after data ends
        x_min_final = 0
        x_max_final = data_end_new + 0.8 * data_range
        x_range = np.linspace(x_min_final, x_max_final, 600)
        
        # Plot density curves for each category (using compressed coordinates)
        for kde, category, i in kde_list:
            # Calculate density in original coordinate system
            x_original = x_range - shift  # Convert back to original coordinates
            density = kde(x_original)
            
            # Plot in compressed coordinate system
            ax.plot(x_range, density, color=colors[i], linewidth=2.5, 
                    label=category, alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.3, color=colors[i])
            
    else:
        # If no negative values, proceed normally
        x_range = x_range_initial
        for kde, category, i in kde_list:
            density = kde(x_range)
            ax.plot(x_range, density, color=colors[i], linewidth=2.5, 
                    label=category, alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.3, color=colors[i])
    
    # Set axes
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # Show legend only in the first subplot
    if idx == 0:
        ax.legend(loc='upper left', fontsize=10.5, framealpha=0.9)

# Adjust layout
plt.tight_layout()

# Create output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)

# Save the figure
output_file = os.path.join(output_dir, 'S11. Probability_Density.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_file}")

plt.show()

# Print basic statistics
print("\nBasic Statistics:")
print("="*50)
for category in categories:
    print(f"\n{category}:")
    subset = df[df['category'] == category]
    for var in variables:
        mean_val = subset[var].mean()
        std_val = subset[var].std()
        print(f"  {var}: Mean={mean_val:.3f}, Std Dev={std_val:.3f}")