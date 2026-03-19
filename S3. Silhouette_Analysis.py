import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from tqdm import trange

# ——————————————————————————————————————————————
# 0. Configuration & Output Setup
# ——————————————————————————————————————————————
# Input path
DATA_PATH = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\CBG_ALLcity_Week_NAICS.CSV"

# Output directory
OUTPUT_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")
else:
    print(f"Output directory exists: {OUTPUT_DIR}")

# ——————————————————————————————————————————————
# 1. Load and process the new NAICS data
# ——————————————————————————————————————————————

# Check if the specified data file exists
if not os.path.isfile(DATA_PATH):
    print(f"Error: The specified file does not exist: '{DATA_PATH}'")
    sys.exit() # Terminate the script

print(f"Attempting to load data from: {DATA_PATH}")
df_all = pd.read_csv(DATA_PATH)


# --- Feature Engineering: Calculate visitor proportion ---
print("Processing data and calculating visitor proportion...")

# Using the correct column name for the date
df_all['date'] = pd.to_datetime(df_all['date_range_start'])

# 1. Calculate total visitors per city-week (denominator)
df_total_visitors = df_all.groupby(['City', 'date'])['cbgs_visitor_7day_avg'].sum().reset_index()
df_total_visitors.rename(columns={'cbgs_visitor_7day_avg': 'total_visitors'}, inplace=True)

# 2. Filter for NAICS of interest and sum their counts (numerator)
naics_of_interest = [8129, 4451, 6221] # Service, Grocery, Healthcare
df_interest = df_all[df_all['naics_code_unique'].isin(naics_of_interest)]
df_interest_visitors = df_interest.groupby(['City', 'date'])['cbgs_visitor_7day_avg'].sum().reset_index()
df_interest_visitors.rename(columns={'cbgs_visitor_7day_avg': 'interest_visitors'}, inplace=True)

# 3. Merge total and interest visitors tables
df_merged = pd.merge(df_total_visitors, df_interest_visitors, on=['City', 'date'], how='left')

# 4. Get the other variables (Temp, distance)
other_vars_df = df_all[['City', 'date', 'Temperature', 'distance_from_home_7day_avg']].drop_duplicates()
df_final = pd.merge(df_merged, other_vars_df, on=['City', 'date'])

# 5. Calculate the proportion
df_final['interest_visitors'].fillna(0, inplace=True)
df_final['visitor_proportion'] = 0.0
df_final.loc[df_final['total_visitors'] > 0, 'visitor_proportion'] = df_final['interest_visitors'] / df_final['total_visitors']


# --- Restructure data into the 'series' list for clustering ---
cities = sorted(df_final['City'].unique().tolist())
series = []
for city in cities:
    df_city = df_final[df_final['City'] == city][[
        'date',
        'Temperature',
        'distance_from_home_7day_avg',
        'visitor_proportion'
    ]].sort_values('date')
    series.append(df_city)

print(f"\nSuccessfully loaded and processed data for {len(cities)} cities.")
n, T = len(series), len(series[0])


# ——————————————————————————————————————————————
# 2. Define DTW distance
# ——————————————————————————————————————————————
def dtw_distance(df1, df2):
    # --- Define weights to emphasize certain features ---
    w_temp = 1.0
    w_dist = 1.0
    w_prop = 1.0  # Increased weight for visitor_proportion

    D = np.full((T+1, T+1), np.inf)
    D[0,0] = 0
    for i in range(1, T+1):
        for j in range(1, T+1):
            cost = (
                w_temp * abs(df1['Temperature'].iat[i-1] - df2['Temperature'].iat[j-1])
                + w_dist * abs(df1['distance_from_home_7day_avg'].iat[i-1] - df2['distance_from_home_7day_avg'].iat[j-1])
                + w_prop * abs(df1['visitor_proportion'].iat[i-1] - df2['visitor_proportion'].iat[j-1])
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
# 4.1. Silhouette Score Analysis for Optimal K
# ——————————————————————————————————————————————
print("\n=== Finding Optimal K using Silhouette Score ===")
silhouette_scores = []
k_upper_bound = min(n, 11)
k_range = range(3, k_upper_bound)

for k in k_range:
    labels = fcluster(Z, t=k, criterion="maxclust")
    if len(set(labels)) > 1:
        # Calculate score normally for all k
        score = silhouette_score(dist_mat, labels, metric='precomputed') 
        print(f"  k = {k}, Silhouette Score = {score:.4f}")
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(-1)
        print(f"  k = {k}, only 1 cluster formed, invalid.")

# Determine optimal k automatically
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters found: k = {optimal_k} (Highest Silhouette Score)")
print("==========================================================")

# --- Generate and save silhouette analysis plot ---
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'o-', color='blue')

# --- Loop through points and add text labels ---
for k, score in zip(k_range, silhouette_scores):
    # Add label only for valid scores
    if score > 0: 
        # Format score to 3 decimal places and offset slightly above the point
        plt.text(k, score + 0.005, f'{score:.3f}', ha='center', fontsize=9)

plt.title('Silhouette Score Analysis for Optimal k', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.xticks(k_range)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save to the specific output directory
sil_output_path = os.path.join(OUTPUT_DIR, "S3. silhouette_analysis_naics.png")
plt.savefig(sil_output_path)
print(f"\nSilhouette analysis plot saved as '{sil_output_path}'")
plt.show()

# ——————————————————————————————————————————————
# 4.2. Perform final clustering using the optimal_k
# ——————————————————————————————————————————————
# Using the optimal_k determined automatically above
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