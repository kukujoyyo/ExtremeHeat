import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from tqdm import trange
import matplotlib.patches as mpatches


# ——————————————————————————————————————————————
# 1. Load weekly series data
# ——————————————————————————————————————————————
# NOTE: Please change DATA_DIR to your own valid path
DATA_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\CBG_City_Week_Statistics"
# Define the output directory
OUTPUT_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"

# Create the directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")
else:
    print(f"Output directory exists: {OUTPUT_DIR}")

# Check if the specified data directory exists
if not os.path.isdir(DATA_DIR):
    print(f"Error: The specified directory does not exist: '{DATA_DIR}'")
    print("Please check the path in the DATA_DIR variable.")
    sys.exit()

cities, series = [], []
print(f"Attempting to load data from: {DATA_DIR}")

# Attempt to load real data from the directory
try:
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("CBG_") and fname.endswith("_Week_Statistics.csv"):
            print(f"  Loading file: {fname}")
            city = fname.split("_")[1]
            df = pd.read_csv(os.path.join(DATA_DIR, fname)).sort_values("date_end")
            df['date'] = pd.to_datetime(df['date_end'])
            series.append(df[['date',
                              'Temperature',
                              'distance_from_home_7day_avg',
                              'cbgs_visitor_7day_avg']])
            cities.append(city)
except FileNotFoundError:
    print(f"Error: The directory '{DATA_DIR}' was not found during file loading.")
    sys.exit()

# After attempting to load, check if any data was actually found
if not cities:
    print(f"\nError: No valid CSV data files found in '{DATA_DIR}'.")
    sys.exit()

print(f"\nSuccessfully loaded data for {len(cities)} cities.")
n, T = len(cities), len(series[0])


# ——————————————————————————————————————————————
# 2. Define DTW distance
# ——————————————————————————————————————————————
def dtw_distance(df1, df2):
    D = np.full((T+1, T+1), np.inf)
    D[0,0] = 0
    for i in range(1, T+1):
        for j in range(1, T+1):
            cost = (
                abs(df1['Temperature'].iat[i-1] - df2['Temperature'].iat[j-1])
                + abs(df1['distance_from_home_7day_avg'].iat[i-1] - df2['distance_from_home_7day_avg'].iat[j-1])
                + abs(df1['cbgs_visitor_7day_avg'].iat[i-1] - df2['cbgs_visitor_7day_avg'].iat[j-1])
            )
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return D[T, T]

# ——————————————————————————————————————————————
# 3. Compute DTW distance matrix
# ——————————————————————————————————————————————
dist_mat = np.zeros((n, n))
print("\nComputing DTW distances...")
for i in trange(n):
    for j in range(i+1, n):
        d = dtw_distance(series[i], series[j])
        dist_mat[i,j] = dist_mat[j,i] = d

# ——————————————————————————————————————————————
# 4. MDS embedding + Ward clustering
# ——————————————————————————————————————————————
emb = MDS(n_components=5, dissimilarity="precomputed", random_state=0).fit_transform(dist_mat)
Z = linkage(emb, method="ward")

# ——————————————————————————————————————————————
# 4.1. Method A: Finding Optimal K using Silhouette Score
# ——————————————————————————————————————————————
print("\n=== Method A: Finding Optimal K using Silhouette Score ===")
silhouette_scores = []
k_upper_bound = min(n, 10)
k_range = range(3, k_upper_bound)

for k in k_range:
    labels = fcluster(Z, t=k, criterion="maxclust")
    if len(set(labels)) > 1: # Ensure more than 1 cluster for silhouette score
        score = silhouette_score(emb, labels)
        silhouette_scores.append(score)
        print(f"  k = {k}, Silhouette Score = {score:.4f}")
    else:
        silhouette_scores.append(-1)

# Find optimal k
if silhouette_scores:
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters found: k = {optimal_k} (Highest Silhouette Score)")
else:
    optimal_k = 3 # Default fallback
    print("\nCould not determine optimal k, defaulting to 3")
print("==========================================================")


# ——————————————————————————————————————————————
# 4.2. Perform final clustering using the optimal_k
# ——————————————————————————————————————————————
labels = fcluster(Z, t=optimal_k, criterion="maxclust")

# ——————————————————————————————————————————————
# 4.3. Output classification results
# ——————————————————————————————————————————————
df_clusters = pd.DataFrame({
    'City': cities,
    'Cluster': labels
}).sort_values('Cluster').reset_index(drop=True)
print(f"\n=== City Cluster Assignments (k={optimal_k}) ===")
print(df_clusters.to_string(index=False))

# ——————————————————————————————————————————————
# 5. Build feature matrix (mean & var) for heatmap
# ——————————————————————————————————————————————
features = []
for idx, city in enumerate(cities):
    dfc = series[idx]
    features.append((
        city,
        dfc['cbgs_visitor_7day_avg'].mean(),
        dfc['cbgs_visitor_7day_avg'].var(),
        dfc['Temperature'].mean(),
        dfc['Temperature'].var(),
        dfc['distance_from_home_7day_avg'].mean(),
        dfc['distance_from_home_7day_avg'].var()
    ))

# [修改处] 替换为您要求的列名（按语义一一对应）
feat_df = pd.DataFrame(
    features,
    columns=['City', 'Mean Visit Ratio', 'Var Visit Ratio', 'Mean Temp', 'Var Temp', 'Mean Dist', 'Var Dist']
).set_index('City')

# ——————————————————————————————————————————————
# 6. Normalize for heatmap
# ——————————————————————————————————————————————
scaler = MinMaxScaler()
normed = pd.DataFrame(
    scaler.fit_transform(feat_df),
    index=feat_df.index,
    columns=feat_df.columns
)

# ——————————————————————————————————————————————
# 7. Compute linkage & ordering for visualization
# ——————————————————————————————————————————————
link_rows = linkage(normed.values, method='ward')
row_dendro = dendrogram(link_rows, labels=normed.index.tolist(), no_plot=True)
row_order = row_dendro['leaves']
ordered = normed.iloc[row_order, :]
row_labels = ordered.index.tolist()

col_dendro = dendrogram(linkage(ordered.values.T, method='ward'),
                        labels=normed.columns.tolist(), no_plot=True)
col_order = col_dendro['leaves']
col_labels = ordered.columns[col_order].tolist()
ordered = ordered.iloc[:, col_order]

# ——————————————————————————————————————————————
# 8. OPTIMIZED Enhanced dendrogram + heatmap
# ——————————————————————————————————————————————
mpl.rcParams.update({
    'font.family':'serif','font.size':10,
    'axes.edgecolor':'black',
    'xtick.direction':'out','ytick.direction':'out',
})

city_to_cluster = df_clusters.set_index('City')['Cluster'].to_dict()
cluster_names = {i: f"Cluster {i}" for i in range(1, optimal_k + 1)}
cmap = mpl.colormaps.get('Set1')
# Handle cases where K is larger than Set1 colors (9)
if optimal_k > 9:
    cmap = mpl.colormaps.get('tab20')

cluster_colors_map = {i: cmap(i-1) for i in cluster_names.keys()}


fig = plt.figure(figsize=(16,10), constrained_layout=True)

ax_d = fig.add_axes([0.05, 0.1, 0.5, 0.85])
ax_h = fig.add_axes([0.62, 0.1, 0.2, 0.85])
ax_leg = fig.add_axes([0.83, 0.45, 0.15, 0.15])
cax = fig.add_axes([0.84, 0.1, 0.02, 0.3])

# --- Dendrogram ---
ddata = dendrogram(
    link_rows,
    labels=row_labels,
    orientation='left',
    leaf_font_size=10,
    ax=ax_d
)

ax_d.tick_params(axis='y', labelsize=10, pad=5, length=0)
for label in ax_d.get_yticklabels():
    city_name = label.get_text()
    cluster_id = city_to_cluster.get(city_name)
    if cluster_id:
        try:
            label.set_color(cluster_colors_map[cluster_id])
            label.set_fontweight('bold')
        except KeyError:
            pass

ax_d.set_xticks([])
ax_d.set_ylabel('Cities', fontsize=12)

# --- Heatmap ---
im = ax_h.imshow(ordered.values, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
ax_h.set_xticks(range(len(col_labels)))
ax_h.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
ax_h.set_yticks([])
ax_h.set_xlabel('Features', fontsize=12)

# --- Legend ---
patches = [mpatches.Patch(color=color, label=name) for name, color in zip(cluster_names.values(), cluster_colors_map.values())]
ax_leg.axis('off')
ax_leg.legend(handles=patches, loc='center', frameon=True, fontsize=11, title='Cluster Types', title_fontsize=12)

# --- Colorbar ---
cb = plt.colorbar(im, cax=cax, orientation='vertical')
cb.set_label('Normalized Value', fontsize=10)

save_path_dendro = os.path.join(OUTPUT_DIR, "Fig 2a. DTW Hierarchy.png")
plt.savefig(save_path_dendro)
print(f"\nDendrogram and heatmap plot saved as '{save_path_dendro}'")
plt.show()