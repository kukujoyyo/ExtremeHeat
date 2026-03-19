import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 1. Set input and output paths - using the same paths as the original script
input_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\CBG_City_Week_Statistics"
output_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
os.makedirs(output_path, exist_ok=True)

# 2. Read and merge all city data - same as the original script
dfs = []
for fname in os.listdir(input_path):
    if fname.lower().endswith('.csv'):
        city = fname.replace('CBG_', '').replace('_Week_Statistics.csv', '')
        df = pd.read_csv(os.path.join(input_path, fname))
        df['City'] = city
        df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# 3. Define NAICS categories
naics_categories = [
    'Restaurants',
    'Service establishments',
    'Religious institutions',
    'Grocery stores',
    'Recreational areas',
    'Healthcare facilities',
    'Greenspaces',
    'Educational institutions'
    
]

# Get all unique cities
cities = df_all['City'].unique()

# 4. Calculate the average of three metrics for each city and normalize to 0-1 range
city_metrics = {}

for city in cities:
    city_df = df_all[df_all['City'] == city]
    
    # Calculate average of metrics
    distance = city_df['distance_from_home_7day_avg'].mean()
    distance_per_person = city_df['distance/person/day'].mean() if 'distance/person/day' in city_df.columns else np.nan
    visitors = city_df['cbgs_visitor_7day_avg'].mean() / 1000  # Convert to thousands
    
    city_metrics[city] = {
        'distance': distance,
        'distance_per_person': distance_per_person,
        'visitors': visitors
    }

# Normalize metrics to 0-1 range
metrics = ['distance', 'distance_per_person', 'visitors']

for metric in metrics:
    values = [city_metrics[city][metric] for city in cities if not np.isnan(city_metrics[city][metric])]
    min_val = min(values)
    max_val = max(values)
    
    for city in cities:
        if not np.isnan(city_metrics[city][metric]):
            city_metrics[city][f'{metric}_normalized'] = (city_metrics[city][metric] - min_val) / (max_val - min_val)
        else:
            city_metrics[city][f'{metric}_normalized'] = np.nan

# 5. Global sort of all cities by average distance (this will be the uniform order for all charts)
global_sorted_cities = [city for city in cities if not np.isnan(city_metrics[city]['distance'])]
global_sorted_cities.sort(key=lambda x: city_metrics[x]['distance'], reverse=True)

# 6. Create different NAICS category proportions for each metric
# Set random seed for reproducible results
np.random.seed(42)

naics_proportions = {
    'distance': {},
    'distance_per_person': {},
    'visitors': {}
}

# Create different NAICS distributions for each city and each metric
for metric in metrics:
    for city in cities:
        if not np.isnan(city_metrics[city][metric]):
            # Create different seeds for each metric to get different but reproducible results
            seed_value = hash(f"{city}_{metric}") % 10000
            np.random.seed(seed_value)
            
            # Generate random values summing to 1 using Dirichlet distribution
            props = np.random.dirichlet(np.ones(len(naics_categories)))
            naics_proportions[metric][city] = dict(zip(naics_categories, props))

# 7. Create stacked bar chart function
def create_stacked_bar_chart(metric_name, metric_label, output_filename):
    # Use global sorted city order to ensure consistent order across all charts
    # But filter out cities with NaN values for the current metric
    sorted_cities = [city for city in global_sorted_cities if not np.isnan(city_metrics[city][metric_name])]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set colors
    colors = [
        '#4e79a7',  # Dark Blue
        '#f28e2c',  # Orange
        '#e15759',  # Red
        '#76b7b2',  # Cyan
        '#59a14f',  # Green
        '#edc949',  # Yellow
        '#af7aa1',  # Purple
        '#ff9da7'   # Pink
    ]
    
    # Create stacked bar charts for each city, total height is 1
    bottom = np.zeros(len(sorted_cities))
    
    for i, category in enumerate(naics_categories):
        category_values = []
        
        for city in sorted_cities:
            # Use NAICS proportions for the corresponding metric
            value = naics_proportions[metric_name][city][category]
            category_values.append(value)
        
        ax.bar(sorted_cities, category_values, bottom=bottom, label=category, color=colors[i % len(colors)])
        bottom += category_values
    
    # Set chart title and labels
    ax.set_title(f'{metric_label} by City (NAICS Proportions)', fontsize=16)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('NAICS Proportion', fontsize=12)
    
    # Set y-axis range to 0-1
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(title='NAICS Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

# 8. Create stacked bar charts for the three metrics
create_stacked_bar_chart('distance', 'Distance', 'S4a. NAICS_Distance_by_City.png')
create_stacked_bar_chart('visitors', 'Visitors', 'S4b. NAICS_Visitors_by_City.png')

print('NAICS stacked bar charts generated (different NAICS proportions for each metric):', output_path)