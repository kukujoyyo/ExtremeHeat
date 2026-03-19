import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

def haversine_distance(lat1, lon1, lat2, lon2):

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius (meters)
    r = 6371000
    
    return c * r

def remove_overlapping_pois(poi_df, centrality_values, lat_column, lon_column, max_overlap=5, overlap_distance=5):
    """   
    Parameters:
    poi_df: POI DataFrame
    centrality_values: Array of centrality values
    lat_column, lon_column: Column names for latitude and longitude
    max_overlap: Maximum number of points to keep per location
    overlap_distance: Distance threshold for overlap (meters)
    """
    print(f"Starting processing overlapping points, keeping max {max_overlap} points per location...")
    
    coordinates = poi_df[[lat_column, lon_column]].values
    n_pois = len(poi_df)
    
    # Build KDTree for fast neighbor search
    from scipy.spatial import cKDTree
    
    # Simplified conversion from lat/lon to meters
    def lat_lon_to_meters_simple(coords, ref_lat, ref_lon):
        lat_m = (coords[:, 0] - ref_lat) * 111000
        lon_m = (coords[:, 1] - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        return np.column_stack([lat_m, lon_m])
    
    ref_lat = coordinates[0, 0]
    ref_lon = coordinates[0, 1]
    coords_meters = lat_lon_to_meters_simple(coordinates, ref_lat, ref_lon)
    tree = cKDTree(coords_meters)
    
    # Track indices to keep
    keep_indices = []
    processed = set()
    
    # Sort by centrality value from high to low
    sorted_indices = np.argsort(centrality_values)[::-1]
    
    for idx in sorted_indices:
        if idx in processed:
            continue
            
        # Find all neighbors around current point
        neighbors = tree.query_ball_point(coords_meters[idx], overlap_distance)
        
        # Sort neighbors by centrality, keep top max_overlap
        neighbor_centralities = [(n, centrality_values[n]) for n in neighbors if n not in processed]
        neighbor_centralities.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the top points
        for i, (neighbor_idx, _) in enumerate(neighbor_centralities):
            if i < max_overlap:
                keep_indices.append(neighbor_idx)
            processed.add(neighbor_idx)
    
    # Create filtered data
    keep_indices = sorted(keep_indices)
    filtered_df = poi_df.iloc[keep_indices].reset_index(drop=True)
    filtered_centralities = centrality_values[keep_indices]
    
    print(f"Before overlap removal: {len(poi_df)} POIs")
    print(f"After overlap removal: {len(filtered_df)} POIs")
    print(f"Removed: {len(poi_df) - len(filtered_df)} overlapping POIs")
    
    return filtered_df, filtered_centralities

def filter_pois_by_road_distance(poi_df, G, lat_column, lon_column, max_distance=5000):
    """
    Filter POI points that are further than the specified distance from the nearest road.
    
    Parameters:
    poi_df: POI DataFrame
    G: OSMnx street network graph
    lat_column, lon_column: Column names for latitude and longitude
    max_distance: Maximum distance threshold (meters)
    """
    print(f"Starting filtering POIs further than {max_distance/1000}km from roads...")
    
    # Get coordinates of all road nodes
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    road_coords = [(row['y'], row['x']) for _, row in nodes_gdf.iterrows()]
    
    # Build KDTree for fast nearest neighbor search
    from scipy.spatial import cKDTree
    
    # Convert lat/lon to approximate meters (simplified calculation)
    def lat_lon_to_meters(lat, lon, ref_lat, ref_lon):
        """Simplified conversion from lat/lon to meters"""
        lat_m = (lat - ref_lat) * 111000  # 1 degree lat is approx 111km
        lon_m = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        return lat_m, lon_m
    
    # Use the first POI as reference point
    ref_lat = poi_df[lat_column].iloc[0]
    ref_lon = poi_df[lon_column].iloc[0]
    
    # Convert road node coordinates
    road_coords_meters = [lat_lon_to_meters(lat, lon, ref_lat, ref_lon) 
                         for lat, lon in road_coords]
    tree = cKDTree(road_coords_meters)
    
    # Check distance to nearest road for each POI
    valid_indices = []
    for idx, row in poi_df.iterrows():
        poi_lat, poi_lon = row[lat_column], row[lon_column]
        poi_meters = lat_lon_to_meters(poi_lat, poi_lon, ref_lat, ref_lon)
        
        # Find nearest road node
        distance, _ = tree.query(poi_meters)
        
        if distance <= max_distance:
            valid_indices.append(idx)
    
    filtered_df = poi_df.loc[valid_indices].reset_index(drop=True)
    print(f"Before filtering: {len(poi_df)} POIs")
    print(f"After filtering: {len(filtered_df)} POIs")
    print(f"Removed: {len(poi_df) - len(filtered_df)} POIs far from roads")
    
    return filtered_df

# Compute and visualize city centrality analysis based on POI change_percentage data.
def compute_POI_centrality_plots(cityString, poi_df_city, 
                                visit_column='change_percentage', lat_column='latitude', lon_column='longitude',
                                poi_name_column='city', poi_category_column=None,
                                distance_threshold=1000, alpha=0.1, beta=0.5,
                                n_color_compartments=6, markersize=50, connection_linewidth=0.5,
                                show_connections=True, min_visits_threshold=1, road_filter_distance=5000,
                                max_overlap_points=5, overlap_distance=5, output_dir=None):
    
    
        
    print(f'Building OSMnx street network of {cityString}...')
    
    # Configure OSMnx
    ox.settings.log_console = True
    ox.settings.use_cache = True
    
    # Build street network
    try:
        G = ox.graph_from_place(f'{cityString}, USA', network_type="drive")
    except Exception as e:
        print(f"Could not find city: {cityString}, Error: {e}")
        return
    
    print('Processing POI data...')
    
    # Make a copy to avoid modifying original data
    poi_df = poi_df_city.copy()
    
    # Check if required columns exist
    required_columns = [visit_column, lat_column, lon_column]
    missing_columns = [col for col in required_columns if col not in poi_df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return
    
    print(f"Initial data: {len(poi_df)} POIs")
    
    # No minimum threshold filter - keep all POIs
    print(f"Processing all {len(poi_df)} POIs (no minimum threshold filter)")
    
    if len(poi_df) == 0:
        print("No POIs remaining after filtering, exiting")
        return
    
    # Reset index
    poi_df = poi_df.reset_index(drop=True)
    
    # Filter POIs too far from roads
    if road_filter_distance is not None and road_filter_distance > 0:
        poi_df = filter_pois_by_road_distance(poi_df, G, lat_column, lon_column, road_filter_distance)
        if len(poi_df) == 0:
            print("No POIs remaining after road distance filtering, exiting")
            return
    
    # Process overlapping POIs
    if max_overlap_points is not None and max_overlap_points > 0:
        visit_values = poi_df[visit_column].values
        poi_df, visit_values = remove_overlapping_pois(
            poi_df, visit_values, lat_column, lon_column,
            max_overlap=max_overlap_points, overlap_distance=overlap_distance
        )
    
    if len(poi_df) == 0:
        print("No POIs remaining after overlap removal, exiting")
        return
    
    print(f"Final data for analysis: {len(poi_df)} POIs")
    
    # Compute visit-based centrality
    print('Computing visit-based centrality...')
    visit_centrality = compute_visit_centrality(
        poi_df, visit_column, lat_column, lon_column,
        distance_threshold=distance_threshold, alpha=alpha, beta=beta
    )
    
    # Compute road betweenness centrality for POIs
    print('Computing road network betweenness centrality for POIs...')
    poi_betweenness = compute_poi_betweenness_from_roads(
        G, poi_df, lat_column, lon_column
    )
    
    # Create combined visualization with street network
    print('Creating combined visualization with street network...')
    fig, ax = ox.plot_graph(G, node_size=0, bgcolor='#FFFFFF', 
                            edge_color="#A1A0A0", edge_linewidth=0.6,
                            node_color="#366FD8", 
                            show=False, figsize=(15, 15))
    
    # First layer: Road betweenness centrality (background)
    plot_road_betweenness_on_network(ax, G, poi_df, lat_column, lon_column)
    
    # Second layer: POI visit centrality (foreground)
    plot_poi_visit_on_network(ax, poi_df, visit_centrality, lat_column, lon_column,
                              distance_threshold, markersize, connection_linewidth,
                              show_connections)
    
    # Set title
    ax.set_title(f'{cityString}: Change Percentage Centrality & Road Betweenness', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_poi_plot(fig, cityString, 'combined', distance_threshold, alpha, beta, output_dir)
    
    plt.close(fig)
    print(f'Analysis complete for {cityString}!')

def compute_visit_centrality(poi_df, visit_column, lat_column, lon_column,
                            distance_threshold=1000, alpha=0.1, beta=0.5):
    """
    Compute POI centrality based on change_percentage and distance.
    """
    coordinates = poi_df[[lat_column, lon_column]].values
    visits = poi_df[visit_column].values
    n_pois = len(poi_df)
    
    # Normalize visits to [0, 1]
    if visits.max() > visits.min():
        normalized_visits = (visits - visits.min()) / (visits.max() - visits.min())
    else:
        normalized_visits = np.ones_like(visits)
    
    # Initialize centrality values
    centrality = np.zeros(n_pois)
    
    # Calculate pairwise distances and centrality
    for i in range(n_pois):
        for j in range(n_pois):
            if i != j:
                # Calculate distance using Haversine formula
                dist = haversine_distance(coordinates[i][0], coordinates[i][1],
                                        coordinates[j][0], coordinates[j][1])
                
                if dist <= distance_threshold:
                    # Weighted contribution
                    weight = np.exp(-alpha * dist) * normalized_visits[j]
                    centrality[i] += weight
    
    # Incorporate own visits
    centrality = beta * centrality + (1 - beta) * normalized_visits
    
    # Normalize to [0, 1]
    if centrality.max() > centrality.min():
        centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min())
    
    return centrality

def compute_poi_betweenness_from_roads(G, poi_df, lat_column, lon_column):
    """
    Compute betweenness centrality for POIs based on nearest road network nodes.
    """
    print("Calculating road network betweenness centrality...")
    road_bc = nx.betweenness_centrality(G, weight='length', normalized=True, 
                                       k=min(200, len(G.nodes)))
    
    # Get road node coordinates
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    road_coords = [(row['y'], row['x']) for _, row in nodes_gdf.iterrows()]
    road_node_ids = list(nodes_gdf.index)
    
    # Build KDTree
    def lat_lon_to_meters_simple(coords, ref_lat, ref_lon):
        lat_m = (coords[:, 0] - ref_lat) * 111000
        lon_m = (coords[:, 1] - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        return np.column_stack([lat_m, lon_m])
    
    ref_lat = road_coords[0][0]
    ref_lon = road_coords[0][1]
    road_coords_meters = lat_lon_to_meters_simple(np.array(road_coords), ref_lat, ref_lon)
    tree = cKDTree(road_coords_meters)
    
    # Find nearest road node for each POI
    poi_betweenness = np.zeros(len(poi_df))
    
    for idx, row in poi_df.iterrows():
        poi_lat, poi_lon = row[lat_column], row[lon_column]
        poi_meters = lat_lon_to_meters_simple(np.array([[poi_lat, poi_lon]]), ref_lat, ref_lon)[0]
        
        # Find nearest road node
        distance, nearest_idx = tree.query(poi_meters)
        nearest_road_node = road_node_ids[nearest_idx]
        
        # Get betweenness centrality value for that road node
        poi_betweenness[idx] = road_bc.get(nearest_road_node, 0)
    
    # Normalize
    if poi_betweenness.max() > 0:
        poi_betweenness = poi_betweenness / poi_betweenness.max()
    
    return poi_betweenness

def plot_poi_visit_on_network(ax, poi_df, centrality_values, lat_column, lon_column,
                              distance_threshold, markersize, connection_linewidth, 
                              show_connections):
    """
    Plot POI visit centrality on existing network map (Blue gradient).
    """
    # Use Blues colormap
    cmap = matplotlib.cm.get_cmap('Blues')
    
    # Normalize centrality values for colormap
    if centrality_values.max() > centrality_values.min():
        color_values = (centrality_values - centrality_values.min()) / (centrality_values.max() - centrality_values.min())
    else:
        color_values = np.zeros_like(centrality_values)
    
    # Plot connections (if enabled)
    if show_connections:
        coordinates = poi_df[[lat_column, lon_column]].values
        n_pois = len(poi_df)
        
        for i in range(n_pois):
            for j in range(i+1, n_pois):
                # Calculate distance using Haversine formula
                dist = haversine_distance(coordinates[i][0], coordinates[i][1],
                                        coordinates[j][0], coordinates[j][1])
                
                if dist <= distance_threshold:
                    # Connection color based on average centrality of the two POIs
                    avg_centrality = (color_values[i] + color_values[j]) / 2
                    ax.plot([coordinates[i][1], coordinates[j][1]], 
                           [coordinates[i][0], coordinates[j][0]], 
                           c=cmap(avg_centrality), linewidth=connection_linewidth, alpha=0.6, zorder=6)
    
    # Sort by centrality to ensure important POIs are shown on top
    sorted_indices = np.argsort(centrality_values)
    
    # Plot POI points - fixed size, color varies by centrality value
    for idx in sorted_indices:
        lat, lon = poi_df.iloc[idx][lat_column], poi_df.iloc[idx][lon_column]
        color_val = color_values[idx]
        
        # Fixed point size, dark blue edge
        ax.scatter(lon, lat, c=[cmap(color_val)], s=markersize, 
                  alpha=0.85, edgecolors='darkblue', linewidth=0.3, zorder=7)
    
    # Legend removed as requested

def plot_road_betweenness_on_network(ax, G, poi_df, lat_column, lon_column):
    """
    Plot road network betweenness centrality on existing network map (Yellow-Orange-Red gradient).
    """
    # Calculate road network betweenness centrality
    print("Calculating road network betweenness centrality...")
    road_bc = nx.betweenness_centrality(G, weight='length', normalized=True, 
                                       k=min(200, len(G.nodes)))
    
    # Get road node coordinates and betweenness values
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    
    # Use YlOrRd colormap
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    
    # Get betweenness values
    bc_values = np.array([road_bc.get(node_id, 0) for node_id in nodes_gdf.index])
    
    # Calculate top 10% threshold
    threshold_10_percent = np.percentile(bc_values, 90)  # 90th percentile
    print(f"Betweenness centrality top 10% threshold: {threshold_10_percent}")
    
    # Keep only top 10% nodes
    top_10_mask = bc_values >= threshold_10_percent
    top_10_indices = np.where(top_10_mask)[0]
    
    # Normalize for colormap (based on top 10% values only)
    top_10_values = bc_values[top_10_mask]
    if top_10_values.max() > top_10_values.min():
        color_values = (top_10_values - top_10_values.min()) / (top_10_values.max() - top_10_values.min())
    else:
        color_values = np.zeros_like(top_10_values)
    
    # Sort top 10% nodes by betweenness
    sorted_indices = np.argsort(top_10_values)
    
    # Plot top 10% nodes
    for i, idx in enumerate(sorted_indices):
        actual_idx = top_10_indices[idx]
        node_data = nodes_gdf.iloc[actual_idx]
        lat, lon = node_data['y'], node_data['x']
        color_val = color_values[idx]
        
        # Adjust node size based on betweenness
        node_size = 15 + 80 * color_val  # Size range 15-95
        ax.scatter(lon, lat, c=[cmap(color_val)], s=node_size, 
                  alpha=0.65, edgecolors='none', linewidth=0, zorder=5)
    
    print(f"Displayed top 10% nodes, total {len(top_10_indices)} (Total nodes: {len(bc_values)})")
    
    # Legend removed as requested

def save_poi_plot(fig, cityString, centrality_type, distance_threshold, alpha, beta, output_dir):
    """
    Save POI centrality plot to specified directory.
    """
    # Use provided output directory or default
    if output_dir is None:
        directory = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\BC"
    else:
        directory = output_dir
    
    # Check if directory exists, create if not
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return
    
    filename = f"{directory}/{cityString.replace(' ', '_')}_{centrality_type}.png"
    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Image saved: {filename}")

# Main execution
if __name__ == "__main__":
    # Read the Excel file
    input_file = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\BC\POI_Heat_NoHeat_Change_Analysis.xlsx"
    output_dir = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\BC"
    
    print(f"Reading data from: {input_file}")
    df_all = pd.read_excel(input_file)
    
    print(f"Total data: {len(df_all)} rows")
    print(f"Columns: {list(df_all.columns)}")
    
    # Get unique cities
    cities = df_all['city'].unique()
    print(f"\nFound {len(cities)} cities: {list(cities)}")
    
    # Process each city
    for i, city in enumerate(cities, 1):
        print(f"\n{'='*80}")
        print(f"Processing city {i}/{len(cities)}: {city}")
        print(f"{'='*80}")
        
        # Filter data for current city
        city_df = df_all[df_all['city'] == city].copy()
        print(f"City {city} has {len(city_df)} POI records")
        
        # Run analysis
        try:
            compute_POI_centrality_plots(
                cityString=city, 
                poi_df_city=city_df,
                visit_column='change_percentage',
                lat_column='latitude',
                lon_column='longitude',
                poi_name_column='city',
                poi_category_column=None,
                distance_threshold=1000,
                alpha=0.1,
                beta=0.5,
                n_color_compartments=8,
                markersize=20,
                show_connections=False,
                min_visits_threshold=10,  # No minimum threshold - process all POIs
                road_filter_distance=1000,
                max_overlap_points=30,
                overlap_distance=0,
                output_dir=output_dir
            )
            print(f"✓ Successfully completed analysis for {city}")
        except Exception as e:
            print(f"✗ Error processing {city}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All cities processed!")
    print(f"{'='*80}")