import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
import os

# Define output path
output_path = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code'
if not os.path.exists(output_path):
    os.makedirs(output_path)

weight_by_length = False

# define the study sites as label : query
places = {
    'Houston'       : 'Houston, TX, USA',
    'Dallas'        : 'Dallas, TX, USA',
    'Chicago'       : 'Chicago, IL, USA',
    'Los Angeles'   : 'Los Angeles, CA, USA',
    'Phoenix'       : 'Phoenix, AZ, USA',
    'New York'      : 'New York, NY, USA',
    'San Francisco' : 'San Francisco, CA, USA',
    'San Diego'     : 'San Diego, CA, USA',
    'San Antonio'   : 'San Antonio, TX, USA',
    'Philadelphia'  : 'Philadelphia, PA, USA',
    'Oklahoma'      : 'Oklahoma City, OK, USA',
    'Austin'        : 'Austin, TX, USA',
    'San Jose'      : 'San Jose, CA, USA',
    'Jacksonville'  : 'Jacksonville, FL, USA',
    'Indianapolis'  : 'Indianapolis, IN, USA',
    'Denver'        : 'Denver, CO, USA',
    'Columbus'      : 'Columbus, OH, USA',
    'Charlotte'     : 'Charlotte, NC, USA',
    'Nashville'     : 'Nashville, TN, USA',
    'Seattle'       : 'Seattle, WA, USA'
}

# create figure and axes for polar histograms
n = len(places)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))
figsize = (ncols * 5, nrows * 5)
fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": "polar"})

# dictionaries to store metrics
city_entropies = {}
city_order_phi = {}
city_node_density = {}
city_avg_clustering = {}
city_mean_betweenness = {}

# Constants for normalized orientation-order φ
bins = 36
H_max = np.log(bins)
H_g = np.log(2)

for ax, place in zip(axes.flat, sorted(places.keys())):
    print(ox.utils.ts(), place)

    # 1. Download (unprojected) and then project for later metrics
    try:
        G = ox.graph_from_place(places[place], network_type="drive")
        G_proj = ox.project_graph(G)

        # 2. Compute and plot orientation histogram using unprojected G
        Gu_unproj = ox.bearing.add_edge_bearings(G.to_undirected())
        fig, ax = ox.plot.plot_orientation(Gu_unproj, ax=ax, title=place, area=True)

        # 3. Orientation entropy H_o (on unprojected Gu_unproj)
        weight = "length" if weight_by_length else None
        H_o = ox.bearing.orientation_entropy(Gu_unproj, weight=weight)
        city_entropies[place] = H_o

        # 4. Normalized orientation-order φ
        phi = 1 - ((H_o - H_g) / (H_max - H_g))**2
        city_order_phi[place] = phi

        # 5. Node density (nodes per sq km) on projected G_proj
        city_poly = ox.geocode_to_gdf(places[place]).to_crs(G_proj.graph["crs"])
        area_m2 = city_poly.geometry.union_all().convex_hull.area
        area_km2 = area_m2 / 1e6
        node_density = len(G_proj.nodes) / area_km2
        city_node_density[place] = node_density

        # 6. Average clustering coefficient on projected, undirected graph
        G_simple = nx.Graph(G_proj)  # collapse any multiedges
        clustering_dict = nx.clustering(G_simple)
        avg_clust = np.mean(list(clustering_dict.values()))
        city_avg_clustering[place] = avg_clust

        # 7. Mean node betweenness centrality on projected graph
        betweenness_dict = nx.betweenness_centrality(G_simple, weight='length')
        mean_betweenness = np.mean(list(betweenness_dict.values()))
        city_mean_betweenness[place] = mean_betweenness

    except Exception as e:
        print(f"Error processing {place}: {e}")
        city_entropies[place] = None
        city_order_phi[place] = None
        city_node_density[place] = None
        city_avg_clustering[place] = None
        city_mean_betweenness[place] = None


# After loop, print metrics
print("\nOrientation entropy (H_o) for all cities:")
for city, ent in city_entropies.items():
    if ent is not None:
        print(f"↳ {city}: H_o = {ent:.4f}")
    else:
        print(f"↳ {city}: H_o = None")

print("\nNormalized orientation-order (φ) for all cities:")
for city, phi_value in city_order_phi.items():
    if phi_value is not None:
        print(f"↳ {city}: φ   = {phi_value:.4f}")
    else:
        print(f"↳ {city}: φ   = None")

print("\nNode density (nodes/km²) for all cities:")
for city, nd in city_node_density.items():
    if nd is not None:
        print(f"↳ {city}: {nd:.2f}")
    else:
        print(f"↳ {city}: None")

print("\nAverage clustering coefficient for all cities:")
for city, cl in city_avg_clustering.items():
    if cl is not None:
        print(f"↳ {city}: {cl:.4f}")
    else:
        print(f"↳ {city}: None")

print("\nMean node betweenness centrality for all cities:")
for city, bc in city_mean_betweenness.items():
    if bc is not None:
        print(f"↳ {city}: {bc:.6f}")
    else:
        print(f"↳ {city}: None")


# Compile into DataFrame and export to Excel
df = pd.DataFrame({
    "H_o": city_entropies,
    "φ": city_order_phi,
    "node_density_per_km2": city_node_density,
    "avg_clustering_coeff": city_avg_clustering,
    "mean_node_betweenness": city_mean_betweenness
})
df.index.name = "city"
df.to_excel(os.path.join(output_path, "Table 2. city_network_metrics.xlsx"))

# Add figure title and save orientation plot
suptitle_font = {
    "family": "DejaVu Sans",
    "fontsize": 60,
    "fontweight": "normal",
    "y": 1,
}
fig.suptitle("City Street Network Orientation", **suptitle_font)
fig.tight_layout()
fig.subplots_adjust(hspace=0.35)
fig.savefig(os.path.join(output_path, "Fig 4a. street-orientations.png"), facecolor="w", dpi=100, bbox_inches="tight")
plt.close(fig) 