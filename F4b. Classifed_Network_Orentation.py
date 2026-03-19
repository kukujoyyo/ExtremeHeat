import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
import os

# Define output directory
OUTPUT_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

weight_by_length = False

# 1. Define three categories of cities
categories = {
    "High Temperature Suppressed": ["Chicago, IL, USA", "Dallas, TX, USA", "Houston, TX, USA"],
    "Mid Temperature Stable": ["Los Angeles, CA, USA", "New York, NY, USA", "Phoenix, AZ, USA"],
    "Large Metro Baseline": [
        "San Francisco, CA, USA", "San Diego, CA, USA", "San Antonio, TX, USA",
        "Philadelphia, PA, USA", "Oklahoma City, OK, USA", "Austin, TX, USA",
        "San Jose, CA, USA", "Jacksonville, FL, USA", "Indianapolis, IN, USA",
        "Denver, CO, USA", "Columbus, OH, USA", "Charlotte, NC, USA",
        "Nashville, TN, USA", "Seattle, WA, USA"
    ]
}

# 2. Store network metrics for each city
city_metrics = {
    "city": [],
    "category": [],
    "H_o": [],
    "φ": [],
    "node_density_per_km2": [],
    "avg_clustering_coeff": []
}

# 3. Prepare a list for each category to collect all "bearing" values later
bins = 36
bin_edges = np.linspace(0, 360, bins + 1)
category_bearings = {cat: [] for cat in categories}

# 4. Iterate through each category and city, calculate metrics, and collect bearings
for category, city_list in categories.items():
    for place in city_list:
        print("Processing", place)

        # 4.1 Download unprojected street network and project to meters
        try:
            G = ox.graph_from_place(place, network_type="drive")
            G_proj = ox.project_graph(G)

            # 4.2 Calculate "bearing", needs to be done on the "unprojected" undirected graph
            Gu_unproj = ox.bearing.add_edge_bearings(G.to_undirected())
            bearings = []
            for u, v, key, data in Gu_unproj.edges(keys=True, data=True):
                if "bearing" in data:
                    bearings.append(data["bearing"] % 360)
            bearings = np.array(bearings)
            category_bearings[category].append(bearings)

            # 4.3 Calculate orientation entropy H_o (unweighted)
            weight = "length" if weight_by_length else None
            H_o = ox.bearing.orientation_entropy(Gu_unproj, weight=weight)

            # 4.4 Calculate normalized order φ
            H_g = np.log(2)
            H_max = np.log(bins)
            φ = 1 - ((H_o - H_g) / (H_max - H_g)) ** 2

            # 4.5 Calculate node density (nodes per km²)
            city_poly = ox.geocode_to_gdf(place).to_crs(G_proj.graph["crs"])
            area_m2 = city_poly.geometry.union_all().convex_hull.area
            area_km2 = area_m2 / 1e6
            node_density = len(G_proj.nodes) / area_km2

            # 4.6 Calculate Average Clustering Coefficient
            G_simple = nx.Graph(G_proj)
            clustering_dict = nx.clustering(G_simple)
            avg_clust = np.mean(list(clustering_dict.values()))

            # 4.8 Save to city_metrics dictionary
            city_metrics["city"].append(place)
            city_metrics["category"].append(category)
            city_metrics["H_o"].append(H_o)
            city_metrics["φ"].append(φ)
            city_metrics["node_density_per_km2"].append(node_density)
            city_metrics["avg_clustering_coeff"].append(avg_clust)
            
        except Exception as e:
            print(f"Error processing {place}: {e}")

# 5. Convert city_metrics to DataFrame and calculate means by category
df_cities = pd.DataFrame(city_metrics)
numeric_cols = ["H_o", "φ", "node_density_per_km2", "avg_clustering_coeff"]
df_grouped = df_cities.groupby("category")[numeric_cols].mean()

# 6. Plot "average street orientation distribution" rose charts for the three categories
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})

for ax, (category, bearings_list) in zip(axes, category_bearings.items()):
    if not bearings_list: # Skip if no data for this category
        continue
        
    # 6.1 Concatenate all bearings for this category
    combined = np.concatenate(bearings_list)

    # 6.2 Calculate histogram: counts
    counts, _ = np.histogram(combined, bins=bin_edges)

    # 6.3 Normalize to probabilities
    probs = counts / counts.sum()

    # 6.4 Calculate center angle (radians) for each bin
    width = 2 * np.pi / bins
    angles_mid = np.deg2rad(bin_edges[:-1] + (360 / bins) / 2)

    # 6.5 Plot polar bar chart
    bar_color = "#193463"
    bars = ax.bar(
        angles_mid,
        probs,
        width=width,
        bottom=0.0,
        color=bar_color,
        edgecolor="none",
        alpha=0.8
    )

    # 6.6 Set polar grid and labels
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Set y-axis ticks: 4 equidistant concentric circles
    max_r = probs.max() * 1.1
    r_ticks = np.linspace(0, max_r, 5)[1:]  # Remove 0, keep 4 circles
    ax.set_yticks(r_ticks)
    ax.set_ylim(0, r_ticks[-2])  # Set upper limit to the second to last circle position
    
    # Hide radial labels
    ax.set_yticklabels([])
    
    # Add gray diagonal lines at 45 degrees (lines only, no labels)
    for angle in [45, 135, 225, 315]:
        ax.plot([0, np.deg2rad(angle)], [0, max_r], 
                color='gray', linewidth=0.5, alpha=0.3, linestyle='-')
    
    # Set major direction labels (N, E, S, W) - set this last
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=12, fontweight='bold')
    
    # Set grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="gray", alpha=0.3)
    
    # Set title
    ax.set_title(category, fontsize=14, pad=20)

# 7. Save the full figure containing 3 subplots
fig.tight_layout()
fig.subplots_adjust(top=0.88, hspace=0.35)
savefile = os.path.join(OUTPUT_DIR, "Fig 4b. Classified_street-orientations.png")
fig.savefig(savefile, facecolor="white", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved figure to {savefile}")

# 8. Output category averages to screen
print("\n--- Category Averages ---")
print(df_grouped)
print("\nNote: Each plot shows the aggregated street orientation distribution for all cities in its category.")

# 9. Write city-level metrics and category averages to the same Excel file
excel_path = os.path.join(OUTPUT_DIR, "Classified_city_metrics.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_cities.to_excel(writer, sheet_name="City_Metrics", index=False)
    df_grouped.to_excel(writer, sheet_name="Category_Averages")
print(f"Saved metrics to {excel_path}")