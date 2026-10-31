import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_ROOT  # noqa: E402

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Set input and output paths - using the same paths as the original script
input_path = os.path.join(DATA_ROOT, "6.Foot_Traffic_City_Week/CBG_City_Week_NAICS")
output_path = os.path.join(DATA_ROOT, "Results2_essential/Weekly_city")
os.makedirs(output_path, exist_ok=True)

# 2. Read NAICS-level weekly data (one row per city x week x NAICS code)
dfs = []
for fname in os.listdir(input_path):
    if fname.lower().endswith('.csv'):
        df = pd.read_csv(os.path.join(input_path, fname))
        df['date_range_start'] = pd.to_datetime(df['date_range_start'], errors='coerce')
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Unify city name capitalization (e.g. 'philadelphia' -> 'Philadelphia')
df_all['City'] = df_all['City'].apply(lambda c: c[0].upper() + c[1:])

# 3. Define NAICS categories (NAICS code -> category)
naics_map = {
    7225: 'Restaurants',               # Restaurants and other eating places
    8129: 'Service establishments',    # Other personal services
    8131: 'Religious institutions',    # Religious organizations
    4451: 'Grocery stores',            # Grocery and convenience retailers
    7139: 'Recreational areas',        # Other amusement and recreation industries
    6221: 'Healthcare facilities',     # General medical and surgical hospitals
    7121: 'Greenspaces',               # Museums, historical sites, zoos, and parks
    6111: 'Educational institutions'   # Elementary and secondary schools
}
naics_categories = list(naics_map.values())
df_all['Category'] = df_all['naics_code_unique'].map(naics_map)

# Get all unique cities
cities = df_all['City'].unique()

# 4. Average each metric over weeks for every city x NAICS category
metrics = ['distance', 'distance_per_person', 'visitors']
metric_columns = {
    'distance': 'distance_from_home_7day_avg',
    'distance_per_person': 'distance/person/day',
    'visitors': 'cbgs_visitor_7day_avg'
}

city_cat_means = df_all.groupby(['City', 'Category'])[list(metric_columns.values())].mean()

# City-level metric = sum over the 8 categories of their weekly means
city_metrics = {}
for city in cities:
    city_metrics[city] = {
        metric: city_cat_means.loc[city, col].sum() for metric, col in metric_columns.items()
    }

# 5. Global sort of all cities by average distance (this will be the uniform order for all charts)
global_sorted_cities = [city for city in cities if not np.isnan(city_metrics[city]['distance'])]
global_sorted_cities.sort(key=lambda x: city_metrics[x]['distance'], reverse=True)

# 6. NAICS category proportions: category mean / sum of the 8 category means, per city and metric
naics_proportions = {metric: {} for metric in metrics}

for metric, col in metric_columns.items():
    for city in cities:
        cat_values = city_cat_means.loc[city, col]
        naics_proportions[metric][city] = (cat_values / cat_values.sum()).to_dict()

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
    ax.set_xlabel('City', fontsize=22)
    ax.set_ylabel('NAICS Proportion', fontsize=22)

    # Set y-axis range to 0-1
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=18)

    # Rotate x-axis labels to avoid overlap; display 'LA' as 'Los Angeles'
    display_names = {'LA': 'Los Angeles'}
    plt.xticks(range(len(sorted_cities)), [display_names.get(c, c) for c in sorted_cities],
               rotation=45, ha='right', fontsize=18)

    # Add legend
    ax.legend(title='NAICS Categories', bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=18, title_fontsize=18)
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

# 8. Create stacked bar charts for the three metrics
create_stacked_bar_chart('distance', 'Distance', 'NAICS_Distance_by_City.png')
create_stacked_bar_chart('distance_per_person', 'Distance per Person per Day', 'NAICS_Distance_per_Person_by_City.png')
create_stacked_bar_chart('visitors', 'Visitors', 'NAICS_Visitors_by_City.png')

print('✅ NAICS stacked bar charts generated (different NAICS proportions for each metric):', output_path)