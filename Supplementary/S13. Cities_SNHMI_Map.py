import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import warnings
import requests
import json
import time
warnings.filterwarnings('ignore')

CENSUS_API_KEY = "731a831d04a9f0c6ac8e52f5972337567422779"


CORRELATION_DATA_VISIT = {
    'betweenness_centrality': {'rho': 0.394, 'p': 0.001, 'significant': True},
    'node_density': {'rho': -0.661, 'p': 0.001, 'significant': True},
    'clustering_coefficient': {'rho': -0.422, 'p': 0.001, 'significant': True},
    'orientation_order': {'rho': -0.154, 'p': 0.150, 'significant': False},
    'orientation_entropy': {'rho': 0.156, 'p': 0.145, 'significant': False}
}

CORRELATION_DATA_DISTANCE = {
    'betweenness_centrality': {'rho': 0.385, 'p': 0.001, 'significant': True},
    'node_density': {'rho': -0.598, 'p': 0.001, 'significant': True},
    'clustering_coefficient': {'rho': -0.401, 'p': 0.001, 'significant': True},
    'orientation_order': {'rho': -0.132, 'p': 0.220, 'significant': False},
    'orientation_entropy': {'rho': 0.145, 'p': 0.175, 'significant': False}
}

def calculate_rho_squared_weights(correlation_data):
    """
    Calculate weights using ρ² normalization method.

    Formula: w_i = ρ_i² / Σ(ρ_j²)
    - Only significant indicators (p < 0.05) are included
    - ρ² represents the proportion of explained variance
    - Normalization ensures weights sum to 1

    Parameters:
    correlation_data: dict, correlation coefficients and significance

    Returns:
    dict: weights for each indicator
    """
    significant_indicators = {
        name: data for name, data in correlation_data.items()
        if data['significant']
    }

    rho_squared_sum = sum(data['rho']**2 for data in significant_indicators.values())

    weights = {}
    for name, data in correlation_data.items():
        if data['significant']:
            weights[name] = (data['rho']**2) / rho_squared_sum
        else:
            weights[name] = 0.0

    return weights

CORRELATION_WEIGHTS_VISIT = calculate_rho_squared_weights(CORRELATION_DATA_VISIT)
CORRELATION_WEIGHTS_DISTANCE = calculate_rho_squared_weights(CORRELATION_DATA_DISTANCE)

print("\n" + "="*60)
print("SNTSI-V (Visit Sensitivity) - Weights:")
print("="*60)
for indicator, weight in CORRELATION_WEIGHTS_VISIT.items():
    if weight > 0:
        rho = CORRELATION_DATA_VISIT[indicator]['rho']
        print(f"{indicator:30s}: ρ={rho:+.3f}, ρ²={rho**2:.6f}, w={weight:.4f}")
    else:
        print(f"{indicator:30s}: Not significant, w=0.0000")
print(f"{'Sum of weights':30s}: {sum(CORRELATION_WEIGHTS_VISIT.values()):.4f}")

print("\n" + "="*60)
print("SNTSI-D (Distance Sensitivity) - Weights:")
print("="*60)
for indicator, weight in CORRELATION_WEIGHTS_DISTANCE.items():
    if weight > 0:
        rho = CORRELATION_DATA_DISTANCE[indicator]['rho']
        print(f"{indicator:30s}: ρ={rho:+.3f}, ρ²={rho**2:.6f}, w={weight:.4f}")
    else:
        print(f"{indicator:30s}: Not significant, w=0.0000")
print(f"{'Sum of weights':30s}: {sum(CORRELATION_WEIGHTS_DISTANCE.values()):.4f}")
print("="*60 + "\n")

CORRELATION_DATA = CORRELATION_DATA_VISIT
CORRELATION_WEIGHTS = CORRELATION_WEIGHTS_VISIT

API_DELAY = 2
MAX_RETRIES = 3
RETRY_WAIT = 60

VISITOR_DATA_DIR = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Heat_NoHeat_OD"

def load_visitor_data(city_name, visitor_data_dir=VISITOR_DATA_DIR):
    """
    Load visitor data for a city from Heat and NoHeat sheets.

    Parameters:
    city_name: str, city name (e.g., 'Houston')
    visitor_data_dir: str, directory containing visitor Excel files

    Returns:
    tuple: (heat_df, noheat_df) or (None, None) if failed
    """
    try:
        import os
        file_path = os.path.join(visitor_data_dir, f"{city_name}.xls")

        if not os.path.exists(file_path):
            print(f"   Warning: Visitor data file not found: {file_path}")
            return None, None

        heat_df = pd.read_excel(file_path, sheet_name='Heat', engine='xlrd')
        noheat_df = pd.read_excel(file_path, sheet_name='NoHeat', engine='xlrd')

        print(f"   Loaded visitor data: {len(heat_df)} heat visitors, {len(noheat_df)} no-heat visitors")

        return heat_df, noheat_df

    except Exception as e:
        print(f"   Warning: Failed to load visitor data for {city_name}: {e}")
        return None, None

def count_cbg_visitors(cbg_gdf, heat_df, noheat_df):
    """
    Count visitors and calculate distances for each CBG.

    Statistics:
    1. heat_visitors: Sum of 'cbgs_visitor' field in Heat sheet where home location is in this CBG
    2. noheat_visitors: Sum of 'cbgs_visitor' field in NoHeat sheet where home location is in this CBG
    3. visitor_change: (heat_visitors - noheat_visitors) / noheat_visitors
    4. heat_distance: Average distance from home to visit location (Heat period)
    5. noheat_distance: Average distance from home to visit location (NoHeat period)
    """
    from shapely.geometry import Point
    from math import radians, sin, cos, sqrt, atan2
    import numpy as np
    import pandas as pd

    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance in km using Haversine formula"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    print(f"   Counting visitors for {len(cbg_gdf)} CBGs...")
    print(f"   Heat: {len(heat_df) if heat_df is not None else 0} records")
    print(f"   NoHeat: {len(noheat_df) if noheat_df is not None else 0} records")

    visitor_counts = {}

    for idx, cbg_row in cbg_gdf.iterrows():
        geoid = cbg_row['GEOID']
        cbg_geom = cbg_row.geometry

        heat_count, noheat_count = 0, 0
        heat_distances, noheat_distances = [], []

        if heat_df is not None and len(heat_df) > 0:
            req_cols = ['home_lat', 'home_lon', 'cbgs_visitor', 'latitude', 'longitude']
            if all(col in heat_df.columns for col in req_cols):
                for _, row in heat_df.dropna(subset=['home_lat', 'home_lon']).iterrows():
                    try:
                        if cbg_geom.contains(Point(row['home_lon'], row['home_lat'])):
                            if not pd.isna(row.get('cbgs_visitor')):
                                heat_count += row['cbgs_visitor']
                            if not pd.isna(row.get('latitude')) and not pd.isna(row.get('longitude')):
                                heat_distances.append(haversine_distance(
                                    row['home_lat'], row['home_lon'],
                                    row['latitude'], row['longitude']))
                    except:
                        continue

        if noheat_df is not None and len(noheat_df) > 0:
            req_cols = ['home_lat', 'home_lon', 'cbgs_visitor', 'latitude', 'longitude']
            if all(col in noheat_df.columns for col in req_cols):
                for _, row in noheat_df.dropna(subset=['home_lat', 'home_lon']).iterrows():
                    try:
                        if cbg_geom.contains(Point(row['home_lon'], row['home_lat'])):
                            if not pd.isna(row.get('cbgs_visitor')):
                                noheat_count += row['cbgs_visitor']
                            if not pd.isna(row.get('latitude')) and not pd.isna(row.get('longitude')):
                                noheat_distances.append(haversine_distance(
                                    row['home_lat'], row['home_lon'],
                                    row['latitude'], row['longitude']))
                    except:
                        continue

        visitor_change = (heat_count - noheat_count) / noheat_count if noheat_count > 0 else (0 if heat_count == 0 else np.inf)

        visitor_counts[geoid] = {
            'heat_visitors': heat_count,
            'noheat_visitors': noheat_count,
            'visitor_change': visitor_change,
            'heat_distance': np.mean(heat_distances) if heat_distances else 0,
            'noheat_distance': np.mean(noheat_distances) if noheat_distances else 0
        }

    total_heat = sum(v['heat_visitors'] for v in visitor_counts.values())
    total_noheat = sum(v['noheat_visitors'] for v in visitor_counts.values())
    print(f"   ✓ Visitors: {total_heat} heat, {total_noheat} noheat")

    return visitor_counts


def calculate_gini_coefficient(values):

    if len(values) == 0:
        return np.nan

    sorted_values = np.sort(values)
    n = len(sorted_values)

    cumsum = np.cumsum(sorted_values)

    Y = cumsum / cumsum[-1] if cumsum[-1] > 0 else np.zeros(n)

    X = np.arange(1, n + 1) / n

    gini = 1 - np.sum((X[1:] - X[:-1]) * (Y[1:] + Y[:-1]))

    return gini

def calculate_node_density(G, area_km2):
    """Calculate node density (nodes per km²)"""
    if area_km2 == 0:
        return 0
    return len(G.nodes) / area_km2

def calculate_clustering_coefficient(G):
    """Calculate average clustering coefficient"""
    try:
        G_undirected = G.to_undirected()

        if G_undirected.is_multigraph():
            G_simple = nx.Graph(G_undirected)
        else:
            G_simple = G_undirected

        clustering_coeffs = nx.clustering(G_simple)
        if len(clustering_coeffs) == 0:
            return 0
        return np.mean(list(clustering_coeffs.values()))
    except Exception as e:
        print(f"      Error calculating clustering coefficient: {e}")
        return np.nan

def calculate_orientation_entropy(G):
    """Calculate orientation entropy of street network"""
    try:
        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

        if 'bearing' not in edges_gdf.columns:
            G = ox.add_edge_bearings(G)
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

        bearings = edges_gdf['bearing'].values

        num_bins = 36
        bins = np.arange(0, 360, 360/num_bins)
        hist, _ = np.histogram(bearings, bins=bins)

        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remove zero probabilities

        entropy = -np.sum(probs * np.log(probs))

        max_entropy = np.log(num_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy
    except:
        return np.nan

def calculate_orientation_order(G):
    """Calculate orientation order (1 - normalized entropy)"""
    entropy = calculate_orientation_entropy(G)
    if np.isnan(entropy):
        return np.nan
    return 1 - entropy

def calculate_betweenness_centrality(G):
    """Calculate average betweenness centrality"""
    try:
        if len(G.nodes) > 500:
            k = 200
        else:
            k = min(100, len(G.nodes))

        bc = nx.betweenness_centrality(G, weight='length', normalized=True, k=k)

        if len(bc) == 0:
            return 0
        return np.mean(list(bc.values()))
    except:
        return np.nan

def get_state_county_fips(place_name):
    """
    Get state and county FIPS codes for a given place based on MSA boundaries.

    Parameters:
    place_name: str, place name (e.g., 'Houston, TX, USA')

    Returns:
    dict with state_fips and county_fips list
    """

    city_fips_mapping = {
        'Houston': {
            'state_fips': '48',
            'county_fips': ['201', '157', '167', '339', '291', '039', '071', '473']  # Harris, Fort Bend, Galveston, Montgomery, Liberty, Brazoria, Chambers, Waller
        },
        'Dallas': {
            'state_fips': '48',
            'county_fips': ['113', '085', '121', '257', '139', '231', '397', '251']  # Dallas, Collin, Denton, Kaufman, Ellis, Hunt, Rockwall, Johnson
        },
        'Chicago': {
            'state_fips': '17',
            'county_fips': ['031', '043', '089', '093', '097', '111', '197']  # Cook, DuPage, Kane, Kendall, Lake, McHenry, Will
        },
        'Los Angeles': {
            'state_fips': '06',
            'county_fips': ['037', '059', '065', '071', '111']  # Los Angeles, Orange, Riverside, San Bernardino, Ventura
        },
        'Phoenix': {
            'state_fips': '04',
            'county_fips': ['013', '021']  # Maricopa, Pinal
        },
        'New York': {
            'state_fips': '36',
            'county_fips': ['005', '047', '061', '081', '085', '027', '059', '071', '079', '087', '103', '119']  # Bronx, Kings, New York, Queens, Richmond, Dutchess, Nassau, Orange, Putnam, Rockland, Suffolk, Westchester
        },
        'San Francisco': {
            'state_fips': '06',
            'county_fips': ['001', '013', '041', '055', '075', '081', '085', '095', '097']  # Alameda, Contra Costa, Marin, Napa, San Francisco, San Mateo, Santa Clara, Solano, Sonoma
        },
        'San Diego': {
            'state_fips': '06',
            'county_fips': ['073']  # San Diego
        },
        'San Antonio': {
            'state_fips': '48',
            'county_fips': ['029', '013', '091', '187', '259', '325', '493']  # Bexar, Atascosa, Comal, Guadalupe, Kendall, Medina, Wilson
        },
        'Philadelphia': {
            'state_fips': '42',
            'county_fips': ['017', '029', '045', '091', '101']  # Bucks, Chester, Delaware, Montgomery, Philadelphia
        },
        'Oklahoma': {
            'state_fips': '40',
            'county_fips': ['027', '051', '081', '083', '087', '109', '125']  # Canadian, Cleveland, Grady, Lincoln, Logan, Oklahoma, McClain
        },
        'Austin': {
            'state_fips': '48',
            'county_fips': ['021', '055', '209', '453', '491']  # Bastrop, Caldwell, Hays, Travis, Williamson
        },
        'San Jose': {
            'state_fips': '06',
            'county_fips': ['085', '069', '081']  # Santa Clara, San Benito, San Mateo
        },
        'Jacksonville': {
            'state_fips': '12',
            'county_fips': ['003', '019', '031', '089', '109']  # Baker, Clay, Duval, Nassau, St. Johns
        },
        'Indianapolis': {
            'state_fips': '18',
            'county_fips': ['011', '013', '023', '057', '059', '063', '081', '095', '097', '109', '145']  # Boone, Brown, Delaware, Hamilton, Hancock, Hendricks, Johnson, Madison, Marion, Morgan, Shelby
        },
        'Denver': {
            'state_fips': '08',
            'county_fips': ['001', '005', '014', '019', '031', '035', '059']  # Adams, Arapahoe, Broomfield, Clear Creek, Denver, Douglas, Jefferson
        },
        'Columbus': {
            'state_fips': '39',
            'county_fips': ['041', '045', '049', '073', '089', '097', '117', '127', '129', '159']  # Delaware, Fairfield, Franklin, Hocking, Licking, Madison, Morrow, Perry, Pickaway, Union
        },
        'Charlotte': {
            'state_fips': '37',
            'county_fips': ['025', '071', '097', '109', '119', '159', '179']  # Cabarrus, Gaston, Iredell, Lincoln, Mecklenburg, Rowan, Union
        },
        'Nashville': {
            'state_fips': '47',
            'county_fips': ['015', '021', '037', '043', '111', '119', '147', '149', '159', '165', '187', '189']  # Cannon, Cheatham, Davidson, Dickson, Macon, Maury, Robertson, Rutherford, Smith, Sumner, Williamson, Wilson
        },
        'Seattle': {
            'state_fips': '53',
            'county_fips': ['033', '053', '061']  # King, Pierce, Snohomish
        }
    }

    city_short = place_name.split(',')[0].strip()

    return city_fips_mapping.get(city_short, None)

def get_census_block_groups_real(place_name, census_api_key=None):
    """
    Get real census block groups from Census Bureau API.

    Parameters:
    place_name: str, place name (e.g., 'Houston, TX, USA')
    census_api_key: str, Census API key (optional but recommended)

    Returns:
    GeoDataFrame of census block groups
    """
    try:
        print(f"   Fetching census block groups from Census Bureau API...")

        fips_info = get_state_county_fips(place_name)

        if fips_info is None:
            print(f"   Warning: FIPS codes not found for {place_name}, using grid approximation")
            return get_census_block_groups_grid(place_name)

        state_fips = fips_info['state_fips']
        county_fips_list = fips_info['county_fips']

        all_cbgs = []

        for county_fips in county_fips_list:
            year = 2021  # Use 2021 data (most recent complete)

            tiger_url = f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2021/MapServer/8/query"

            params = {
                'where': f"STATE='{state_fips}' AND COUNTY='{county_fips}'",
                'outFields': '*',
                'returnGeometry': 'true',
                'f': 'geojson'
            }

            print(f"   Fetching Block Groups for State {state_fips}, County {county_fips}...")

            try:
                response = requests.get(tiger_url, params=params, timeout=60)

                if response.status_code == 200:
                    geojson_data = response.json()

                    if 'features' in geojson_data and len(geojson_data['features']) > 0:
                        cbg_gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                        cbg_gdf.crs = 'EPSG:4326'

                        if 'GEOID' not in cbg_gdf.columns:
                            if 'GEOIDFQ' in cbg_gdf.columns:
                                cbg_gdf['GEOID'] = cbg_gdf['GEOIDFQ']
                            elif 'BLKGRPID' in cbg_gdf.columns:
                                cbg_gdf['GEOID'] = cbg_gdf['BLKGRPID']

                        all_cbgs.append(cbg_gdf)
                        print(f"      Retrieved {len(cbg_gdf)} block groups")
                    else:
                        print(f"      No block groups found for this county")

                else:
                    print(f"      API request failed with status code: {response.status_code}")

            except Exception as e:
                print(f"      Error fetching from TIGER API: {e}")
                print(f"      Trying alternative method...")

                try:
                    import importlib
                    pygris = importlib.import_module('pygris')

                    if census_api_key:
                        try:
                            pygris.set_api_key(census_api_key)
                        except:
                            pass

                    cbg_gdf = pygris.block_groups(
                        state=state_fips,
                        county=county_fips,
                        year=year
                    )

                    all_cbgs.append(cbg_gdf)
                    print(f"      Retrieved {len(cbg_gdf)} block groups using pygris")

                except ImportError:
                    print(f"      pygris not installed. Install with: pip install pygris")
                    print(f"      Falling back to grid approximation for this county")

                except Exception as e2:
                    print(f"      pygris also failed: {e2}")
                    print(f"      Falling back to grid approximation for this county")

        if len(all_cbgs) > 0:
            combined_cbg = pd.concat(all_cbgs, ignore_index=True)

            try:
                print(f"   Filtering CBGs to city boundary...")
                city_boundary = ox.geocode_to_gdf(place_name)
                print(f"   Total CBGs before filtering: {len(combined_cbg)}")

                combined_cbg = combined_cbg[combined_cbg.geometry.intersects(city_boundary.geometry.iloc[0])]

                print(f"   Total CBGs within city boundary: {len(combined_cbg)}")
            except Exception as e:
                print(f"   Warning: Could not filter by city boundary: {e}")
                print(f"   Using all CBGs")

            if 'GEOID' not in combined_cbg.columns:
                combined_cbg['GEOID'] = combined_cbg.index.astype(str)

            cols_to_keep = ['GEOID', 'geometry']
            if 'NAME' in combined_cbg.columns:
                cols_to_keep.append('NAME')
            if 'ALAND' in combined_cbg.columns:
                cols_to_keep.append('ALAND')  # Land area

            combined_cbg = combined_cbg[cols_to_keep]

            print(f"   Total block groups fetched: {len(combined_cbg)}")

            return combined_cbg

        else:
            print(f"   No block groups fetched, using grid approximation")
            return get_census_block_groups_grid(place_name)

    except Exception as e:
        print(f"   Error in get_census_block_groups_real: {e}")
        print(f"   Falling back to grid approximation")
        return get_census_block_groups_grid(place_name)

def get_census_block_groups_grid(place_name):
    """
    Fallback method: Create a grid as proxy for census block groups.
    This is used when Census API is unavailable.

    Parameters:
    place_name: str, place name (e.g., 'Houston, TX, USA')

    Returns:
    GeoDataFrame of grid cells
    """
    try:
        city_gdf = ox.geocode_to_gdf(place_name)

        bounds = city_gdf.total_bounds  # minx, miny, maxx, maxy

        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]

        n_cells = int(np.sqrt(50))

        x_step = x_range / n_cells
        y_step = y_range / n_cells

        geometries = []
        ids = []
        idx = 0

        for i in range(n_cells):
            for j in range(n_cells):
                minx = bounds[0] + i * x_step
                miny = bounds[1] + j * y_step
                maxx = minx + x_step
                maxy = miny + y_step

                from shapely.geometry import box
                poly = box(minx, miny, maxx, maxy)

                if city_gdf.geometry.iloc[0].intersects(poly):
                    geometries.append(poly)
                    ids.append(f"GRID_{idx}")
                    idx += 1

        cbg_gdf = gpd.GeoDataFrame({
            'GEOID': ids,
            'geometry': geometries
        }, crs=city_gdf.crs)

        return cbg_gdf

    except Exception as e:
        print(f"   Error in get_census_block_groups_grid: {e}")
        return None

def get_census_block_groups(place_name, census_api_key=None):

    return get_census_block_groups_real(place_name, census_api_key)

def download_city_street_network(place_name, city_short_name):
    """
    Download street network for entire city once (instead of per CBG).
    This dramatically reduces API calls.

    Parameters:
    place_name: str, place name (e.g., 'Houston, TX, USA')
    city_short_name: str, short city name

    Returns:
    NetworkX graph of city street network
    """
    try:
        print(f"   Downloading street network for entire {city_short_name}...")

        G_city = ox.graph_from_place(place_name, network_type='drive', simplify=True)

        print(f"   Downloaded city network: {len(G_city.nodes)} nodes, {len(G_city.edges)} edges")

        return G_city

    except Exception as e:
        print(f"   Error downloading city network: {e}")
        print(f"   Will fall back to per-CBG downloads")
        return None

def extract_cbg_subgraph(G_city, cbg_geometry):
    """
        Parameters:
    G_city: NetworkX graph of entire city
    cbg_geometry: shapely geometry of census block group

    Returns:
    NetworkX subgraph for this CBG
    """
    try:
        nodes_gdf = ox.graph_to_gdfs(G_city, nodes=True, edges=False)

        nodes_within = nodes_gdf[nodes_gdf.geometry.within(cbg_geometry)]

        if len(nodes_within) < 3:
            nodes_within = nodes_gdf[nodes_gdf.geometry.intersects(cbg_geometry)]

        if len(nodes_within) < 3:
            buffered = cbg_geometry.buffer(0.001)  # ~100m buffer
            nodes_within = nodes_gdf[nodes_gdf.geometry.within(buffered)]

        if len(nodes_within) < 3:
            return None

        node_list = list(nodes_within.index)
        G_cbg = G_city.subgraph(node_list).copy()

        if len(G_cbg.nodes) > 0:
            if not nx.is_connected(G_cbg.to_undirected()):
                components = list(nx.connected_components(G_cbg.to_undirected()))
                if len(components) > 0:
                    largest_cc = max(components, key=len)
                    G_cbg = G_cbg.subgraph(largest_cc).copy()

        return G_cbg if len(G_cbg.nodes) >= 3 else None

    except Exception as e:
        return None

def calculate_road_network_indicators_for_cbg_batch(cbg_geometry, G_city):
    """
    Calculate street network indicators for a CBG using pre-downloaded city network.

    Parameters:
    cbg_geometry: shapely geometry of the census block group
    G_city: NetworkX graph of entire city (pre-downloaded)

    Returns:
    dict with calculated indicators
    """
    try:
        G = extract_cbg_subgraph(G_city, cbg_geometry)

        if G is None or len(G.nodes) < 3:
            return None

        cbg_gdf = gpd.GeoDataFrame({'geometry': [cbg_geometry]}, crs='EPSG:4326')
        cbg_projected = cbg_gdf.to_crs('EPSG:3857')
        area_m2 = cbg_projected.geometry.area.iloc[0]
        area_km2 = area_m2 / 1_000_000

        indicators = {
            'node_density': calculate_node_density(G, area_km2),
            'clustering_coefficient': calculate_clustering_coefficient(G),
            'orientation_order': calculate_orientation_order(G),
            'orientation_entropy': calculate_orientation_entropy(G),
            'betweenness_centrality': calculate_betweenness_centrality(G),
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'area_km2': area_km2
        }

        return indicators

    except Exception as e:
        return None

def calculate_road_network_indicators_for_cbg_fallback(cbg_geometry, place_name, max_retries=MAX_RETRIES):
    """
    Calculate road network indicators for a CBG.

    Parameters:
    cbg_geometry: shapely geometry of the census block group
    place_name: str, place name for context
    max_retries: int, maximum number of retries for network download

    Returns:
    dict with calculated indicators
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = RETRY_WAIT * (attempt + 1)
                print(f"      Waiting {wait_time}s before retry {attempt + 1}...")
                time.sleep(wait_time)
            else:
                time.sleep(API_DELAY)

            G = ox.graph_from_polygon(cbg_geometry, network_type='drive', simplify=True)

            if len(G.nodes) < 3:  # Skip if too few nodes
                return None

            cbg_gdf = gpd.GeoDataFrame({'geometry': [cbg_geometry]}, crs='EPSG:4326')
            cbg_projected = cbg_gdf.to_crs('EPSG:3857')  # Project to meters
            area_m2 = cbg_projected.geometry.area.iloc[0]
            area_km2 = area_m2 / 1_000_000

            indicators = {
                'node_density': calculate_node_density(G, area_km2),
                'clustering_coefficient': calculate_clustering_coefficient(G),
                'orientation_order': calculate_orientation_order(G),
                'orientation_entropy': calculate_orientation_entropy(G),
                'betweenness_centrality': calculate_betweenness_centrality(G),
                'num_nodes': len(G.nodes),
                'num_edges': len(G.edges),
                'area_km2': area_km2
            }

            return indicators

        except Exception as e:
            error_str = str(e)

            if "429" in error_str:
                if attempt < max_retries - 1:
                    print(f"      429 Too Many Requests, waiting {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                    continue
                else:
                    print(f"      Failed after {max_retries} attempts due to rate limit")
                    return None

            elif "504" in error_str or "timeout" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"      Timeout, retrying...")
                    continue
                else:
                    print(f"      Failed after {max_retries} timeouts")
                    return None

            else:
                if attempt == max_retries - 1:
                    print(f"      Error: {e}")
                return None

    return None

def calculate_weighted_road_index(cbg_indicators_df):
    """
    Calculate two-dimensional sensitivity indices for each CBG using Z-score standardization.

    Dimensions:
    1. SNTSI-V (Visit Sensitivity Index): Based on Visit Change Rate correlations
    2. SNTSI-D (Distance Sensitivity Index): Based on Distance Change Rate correlations

    Method: Z-score standardization + ρ² weighted aggregation
    Formula: SNTSI = Σ(w_i × Z_i)

    Parameters:
    cbg_indicators_df: DataFrame with all CBGs' indicator values

    Returns:
    DataFrame with added 'SNTSI_V', 'SNTSI_D' columns
    """
    from scipy import stats

    print(f"\n   {'='*50}")
    print(f"   Calculating Two-Dimensional Sensitivity Indices")
    print(f"   {'='*50}")

    significant_indicators_v = [ind for ind, weight in CORRELATION_WEIGHTS_VISIT.items() if weight > 0]
    significant_indicators_d = [ind for ind, weight in CORRELATION_WEIGHTS_DISTANCE.items() if weight > 0]

    print(f"   Standardizing indicators (Z-score transformation)...")
    z_scores = {}
    for indicator in set(significant_indicators_v + significant_indicators_d):
        if indicator in cbg_indicators_df.columns:
            values = cbg_indicators_df[indicator].dropna().values
            if len(values) > 1 and np.std(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                z_scores[indicator] = (cbg_indicators_df[indicator] - mean_val) / std_val
                print(f"      {indicator}: μ={mean_val:.4f}, σ={std_val:.4f}")
            else:
                z_scores[indicator] = cbg_indicators_df[indicator] * 0

    print(f"\n   Dimension 1: SNTSI-V (Visit Sensitivity)")
    print(f"   Weights:")
    sntsi_v_values = []
    for indicator, weight in CORRELATION_WEIGHTS_VISIT.items():
        if weight > 0:
            rho = CORRELATION_DATA_VISIT[indicator]['rho']
            print(f"      {indicator}: ρ={rho:+.3f}, w={weight:.4f}")

    for idx, row in cbg_indicators_df.iterrows():
        weighted_sum = 0
        for indicator, weight in CORRELATION_WEIGHTS_VISIT.items():
            if weight > 0 and indicator in z_scores:
                z_value = z_scores[indicator].iloc[idx]
                if not pd.isna(z_value):
                    weighted_sum += weight * z_value
        sntsi_v_values.append(weighted_sum)

    cbg_indicators_df['SNTSI_V_raw'] = sntsi_v_values

    print(f"\n   Dimension 2: SNTSI-D (Distance Sensitivity)")
    print(f"   Weights:")
    sntsi_d_values = []
    for indicator, weight in CORRELATION_WEIGHTS_DISTANCE.items():
        if weight > 0:
            rho = CORRELATION_DATA_DISTANCE[indicator]['rho']
            print(f"      {indicator}: ρ={rho:+.3f}, w={weight:.4f}")

    for idx, row in cbg_indicators_df.iterrows():
        weighted_sum = 0
        for indicator, weight in CORRELATION_WEIGHTS_DISTANCE.items():
            if weight > 0 and indicator in z_scores:
                z_value = z_scores[indicator].iloc[idx]
                if not pd.isna(z_value):
                    weighted_sum += weight * z_value
        sntsi_d_values.append(weighted_sum)

    cbg_indicators_df['SNTSI_D_raw'] = sntsi_d_values

    for col_raw, col_norm in [('SNTSI_V_raw', 'SNTSI_V'), ('SNTSI_D_raw', 'SNTSI_D')]:
        min_val = cbg_indicators_df[col_raw].min()
        max_val = cbg_indicators_df[col_raw].max()

        if max_val > min_val:
            cbg_indicators_df[col_norm] = (cbg_indicators_df[col_raw] - min_val) / (max_val - min_val)
            print(f"\n   {col_norm} range: [{min_val:.4f}, {max_val:.4f}] → normalized to [0, 1]")
        else:
            cbg_indicators_df[col_norm] = 0.5

    cbg_indicators_df['Composite_SNHMI_raw'] = 0.5 * cbg_indicators_df['SNTSI_V_raw'] + 0.5 * cbg_indicators_df['SNTSI_D_raw']
    cbg_indicators_df['Composite_SNHMI'] = 0.5 * cbg_indicators_df['SNTSI_V'] + 0.5 * cbg_indicators_df['SNTSI_D']

    print(f"\n   Composite Index: Composite_SNHMI = 0.5 × SNTSI_V + 0.5 × SNTSI_D")
    print(f"   Range: [{cbg_indicators_df['Composite_SNHMI'].min():.4f}, {cbg_indicators_df['Composite_SNHMI'].max():.4f}]")

    cbg_indicators_df['weighted_road_index'] = cbg_indicators_df['Composite_SNHMI']

    print(f"   {'='*50}\n")

    return cbg_indicators_df

def fill_missing_with_neighbors(cbg_gdf, cbg_indicators, column='weighted_road_index', max_iterations=20):
    """
    Parameters:
    cbg_gdf: GeoDataFrame with geometry
    cbg_indicators: DataFrame with indicator values
    column: str, column name to fill
    max_iterations: int, maximum iterations to prevent infinite loop

    Returns:
    DataFrame with filled values
    """
    print(f"   Filling missing {column} values with neighbor averages...")

    for iteration in range(max_iterations):
        merged = cbg_gdf.merge(cbg_indicators[['GEOID', column]], on='GEOID', how='left')

        missing_mask = merged[column].isna() | (merged[column] < 0.001)
        missing_indices = merged[missing_mask].index

        if len(missing_indices) == 0:
            print(f"   All CBGs have values")
            break

        print(f"   Iteration {iteration+1}: {len(missing_indices)} CBGs need values")

        search_distance = 0.0001 * (iteration + 1)  # Gradually expand search

        filled_count = 0
        for idx in missing_indices:
            geom = merged.loc[idx, 'geometry']
            geoid = merged.loc[idx, 'GEOID']

            neighbors = merged[merged.geometry.touches(geom) | merged.geometry.intersects(geom)]
            neighbors = neighbors[neighbors.index != idx]

            if len(neighbors) == 0:
                buffered = geom.buffer(search_distance)
                neighbors = merged[merged.geometry.intersects(buffered)]
                neighbors = neighbors[neighbors.index != idx]

            neighbor_values = neighbors[column].dropna()
            neighbor_values = neighbor_values[neighbor_values >= 0.001]

            if len(neighbor_values) > 0:
                filled_value = neighbor_values.mean()
                cbg_indicators.loc[cbg_indicators['GEOID'] == geoid, column] = filled_value
                filled_count += 1

        if filled_count == 0:
            city_mean = merged[column].replace(0, np.nan).mean()
            if np.isnan(city_mean) or city_mean < 0.001:
                city_mean = 0.15  # Reasonable fallback default

            for idx in missing_indices:
                geoid = merged.loc[idx, 'GEOID']
                current_val = cbg_indicators.loc[cbg_indicators['GEOID'] == geoid, column].values
                if len(current_val) == 0 or np.isnan(current_val[0]) or current_val[0] < 0.001:
                    cbg_indicators.loc[cbg_indicators['GEOID'] == geoid, column] = city_mean

            print(f"   Filled remaining {len(missing_indices)} CBGs with city mean: {city_mean:.4f}")
            break

    final_check = cbg_indicators[column]
    final_missing = final_check.isna().sum()
    final_low = (final_check < 0.001).sum()

    if final_missing > 0 or final_low > 0:
        print(f"   Final verification: {final_missing} NaN, {final_low} near-zero values")
        city_mean = final_check[final_check >= 0.001].mean()
        if np.isnan(city_mean) or city_mean < 0.001:
            city_mean = 0.15

        cbg_indicators.loc[final_check < 0.001, column] = city_mean
        cbg_indicators[column] = cbg_indicators[column].fillna(city_mean)
        print(f"   Force-filled with city mean: {city_mean:.4f}")

    print(f"   ✓ All {len(cbg_indicators)} CBGs now have valid values (min: {cbg_indicators[column].min():.4f})")

    return cbg_indicators

def visualize_cbg_road_quality(cbg_gdf, cbg_indicators, city_short_name, output_dir):
    """Visualize with Jenks classification (5 classes)"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        from jenkspy import jenks_breaks
        use_jenks = True
    except ImportError:
        print("   Warning: jenkspy not available, using quantile")
        use_jenks = False

    for index_type, index_col, full_name in [
        ('Composite_SNHMI', 'Composite_SNHMI', 'Composite Heat Mobility Index')
    ]:
        try:
            print(f"\n   Creating {index_type} map...")
            cbg_indicators = fill_missing_with_neighbors(cbg_gdf, cbg_indicators, index_col)
            merged = cbg_gdf.merge(cbg_indicators[['GEOID', index_col]], on='GEOID', how='inner')

            if len(merged) == 0:
                continue

            merged = merged[merged.geometry.is_valid & ~merged.geometry.is_empty]

            if merged[index_col].isna().sum() > 0:
                city_mean = merged[index_col].mean()
                merged[index_col] = merged[index_col].fillna(city_mean if not np.isnan(city_mean) else 0.5)

            values = merged[index_col].values

            if use_jenks and len(values) >= 10:
                try:
                    breaks = jenks_breaks(values, n_classes=5)
                except:
                    breaks = [values.min()] + [np.percentile(values, q) for q in [20,40,60,80]] + [values.max()]
            else:
                breaks = [values.min()] + [np.percentile(values, q) for q in [20,40,60,80]] + [values.max()]

            print(f"      Breaks: {[f'{b:.3f}' for b in breaks]}")

            categories = np.digitize(values, breaks[1:-1])
            merged['category'] = categories

            colors = ['#2166ac', '#67a9cf', '#fee090', '#ef8a62', '#b2182b']
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

            for i, label in enumerate(labels):
                count = (merged['category'] == i).sum()
                pct = count / len(merged) * 100
                print(f"         {label}: {count} ({pct:.1f}%)")

            fig, ax = plt.subplots(1, 1, figsize=(15, 15))

            for cat_idx in range(5):
                cat_data = merged[merged['category'] == cat_idx]
                if len(cat_data) > 0:
                    cat_data.plot(ax=ax, color=colors[cat_idx], edgecolor=colors[cat_idx], linewidth=0)

            merged.boundary.plot(ax=ax, edgecolor='white', linewidth=0.5, alpha=0.3)

            ax.set_title(f'{city_short_name}: {index_type}\n({full_name})', fontsize=16, fontweight='bold', pad=20)
            ax.set_axis_off()

            legend_elements = [
                Patch(facecolor=colors[i], edgecolor='white',
                      label=f'{labels[i]}\n{breaks[i]:.3f}-{breaks[i+1]:.3f}')
                for i in range(5)
            ]
            ax.legend(handles=legend_elements,
                     loc='upper left',
                     bbox_to_anchor=(0.0, -0.05),
                     fontsize=10,
                     title=index_type,
                     framealpha=0.95,
                     ncol=5,
                     borderaxespad=0)

            plt.tight_layout()
            filename = f"{output_dir}/{city_short_name}_Composite_SNHMI_Map.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"      ✓ Saved: {filename}")

        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    if 'category' in merged.columns:
        cbg_indicators_with_cat = cbg_indicators.copy()
        category_map = merged[['GEOID', 'category']].set_index('GEOID')['category'].to_dict()
        cbg_indicators_with_cat['category'] = cbg_indicators_with_cat['GEOID'].map(category_map)
        return cbg_indicators_with_cat
    return cbg_indicators


def calculate_city_road_quality_index(place_name, city_short_name, test_mode=False, max_cbgs=500):
    """
    Calculate weighted road quality index for each CBG in a city.
    NO Gini coefficient calculation - just weighted index.

    Parameters:
    place_name: str, full place name (e.g., 'Houston, TX, USA')
    city_short_name: str, short city name (e.g., 'Houston')
    test_mode: bool, if True, only process first max_cbgs CBGs
    max_cbgs: int, maximum number of CBGs to process in test mode

    Returns:
    tuple: (results_dict, df_indicators, cbg_gdf)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {city_short_name}")
    if test_mode:
        print(f"*** TEST MODE: Only processing first {max_cbgs} CBGs ***")
    print(f"{'='*60}")

    print(f"1. Getting census block groups...")
    cbg_gdf = get_census_block_groups(place_name, CENSUS_API_KEY)

    if cbg_gdf is None or len(cbg_gdf) == 0:
        print(f"Failed to get census block groups for {city_short_name}")
        return None, None, None

    print(f"   Found {len(cbg_gdf)} census block groups")

    if test_mode and len(cbg_gdf) > max_cbgs:
        print(f"   TEST MODE: Limiting to first {max_cbgs} CBGs")
        cbg_gdf = cbg_gdf.head(max_cbgs).copy()
        print(f"   Processing {len(cbg_gdf)} CBGs")

    print(f"2. Loading visitor data...")
    heat_df, noheat_df = load_visitor_data(city_short_name)

    visitor_counts = {}
    if heat_df is not None and noheat_df is not None:
        print(f"   Counting visitors in each CBG...")
        visitor_counts = count_cbg_visitors(cbg_gdf, heat_df, noheat_df)
        total_heat = sum(v['heat_visitors'] for v in visitor_counts.values())
        total_noheat = sum(v['noheat_visitors'] for v in visitor_counts.values())
        print(f"   Total visitors: {total_heat} heat, {total_noheat} no-heat")

    print(f"3. Downloading city-wide street network (1 API request)...")
    G_city = download_city_street_network(place_name, city_short_name)

    if G_city is None:
        print(f"   Failed to download city network, aborting {city_short_name}")
        return None, None, None

    print(f"4. Filtering CBGs to match road network coverage...")
    nodes_gdf = ox.graph_to_gdfs(G_city, nodes=True, edges=False)
    network_bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]

    original_count = len(cbg_gdf)
    cbg_gdf = cbg_gdf[cbg_gdf.geometry.intersects(nodes_gdf.unary_union.envelope)]

    print(f"   CBGs before filtering: {original_count}")
    print(f"   CBGs after filtering: {len(cbg_gdf)}")

    if len(cbg_gdf) == 0:
        print(f"   No CBGs overlap with road network")
        return None, None, None

    print(f"5. Calculating road network indicators for each CBG...")
    print(f"   Total CBGs to process: {len(cbg_gdf)}")

    all_indicators = []

    for idx, row in cbg_gdf.iterrows():
        cbg_number = idx + 1
        print(f"   Processing CBG {cbg_number}/{len(cbg_gdf)} ({cbg_number/len(cbg_gdf)*100:.1f}%)...", end='\r')

        indicators = calculate_road_network_indicators_for_cbg_batch(
            row.geometry,
            G_city
        )

        if indicators is not None:
            geoid = row['GEOID']
            indicators['GEOID'] = geoid

            if geoid in visitor_counts:
                indicators['heat_visitors'] = visitor_counts[geoid]['heat_visitors']
                indicators['noheat_visitors'] = visitor_counts[geoid]['noheat_visitors']
                indicators['visitor_change'] = visitor_counts[geoid]['visitor_change']
                indicators['heat_distance'] = visitor_counts[geoid]['heat_distance']
                indicators['noheat_distance'] = visitor_counts[geoid]['noheat_distance']
            else:
                indicators['heat_visitors'] = 0
                indicators['noheat_visitors'] = 0
                indicators['visitor_change'] = 0
                indicators['heat_distance'] = 0
                indicators['noheat_distance'] = 0

            all_indicators.append(indicators)

    print(f"\n   Completed {len(all_indicators)}/{len(cbg_gdf)} CBGs successfully          ")

    if len(all_indicators) == 0:
        print(f"No valid indicators calculated for {city_short_name}")
        return None, None, None

    df_indicators = pd.DataFrame(all_indicators)

    print(f"6. Calculating weighted road quality index for each CBG...")
    df_indicators = calculate_weighted_road_index(df_indicators)

    print(f"7. Summary statistics:")

    summary_results = {
        'city': city_short_name,
        'num_cbgs': len(df_indicators),
        'test_mode': test_mode,
        'total_heat_visitors': df_indicators['heat_visitors'].sum() if 'heat_visitors' in df_indicators.columns else 0,
        'total_noheat_visitors': df_indicators['noheat_visitors'].sum() if 'noheat_visitors' in df_indicators.columns else 0,
        'visitor_change': df_indicators['visitor_change'].sum() if 'visitor_change' in df_indicators.columns else 0,
    }

    if 'weighted_road_index' in df_indicators.columns:
        summary_results['mean_road_index'] = df_indicators['weighted_road_index'].mean()
        summary_results['std_road_index'] = df_indicators['weighted_road_index'].std()
        summary_results['min_road_index'] = df_indicators['weighted_road_index'].min()
        summary_results['max_road_index'] = df_indicators['weighted_road_index'].max()
        print(f"   Mean HS-SNHMI: {summary_results['mean_road_index']:.4f}")
        print(f"   Std HS-SNHMI: {summary_results['std_road_index']:.4f}")

    for indicator in ['node_density', 'clustering_coefficient', 'orientation_order',
                     'orientation_entropy', 'betweenness_centrality']:
        values = df_indicators[indicator].dropna().values
        if len(values) > 0:
            summary_results[f'{indicator}_mean'] = np.mean(values)
            summary_results[f'{indicator}_std'] = np.std(values)
            summary_results[f'{indicator}_min'] = np.min(values)
            summary_results[f'{indicator}_max'] = np.max(values)

    return summary_results, df_indicators, cbg_gdf

def main(test_mode=True, max_cbgs_test=20, test_cities=None):
    """

    Parameters:
    test_mode: bool, if True, only process first max_cbgs_test CBGs per city
    max_cbgs_test: int, number of CBGs to process per city in test mode
    test_cities: list, if provided, only process these cities (for quick testing)
    """

    places = {
        # 'Houston'       : 'Houston, TX, USA',
        # 'Dallas'        : 'Dallas, TX, USA',
        # 'Chicago'       : 'Chicago, IL, USA',
        # 'Los Angeles'   : 'Los Angeles, CA, USA',
        # 'Phoenix'       : 'Phoenix, AZ, USA',
        # 'New York'      : 'New York, NY, USA',
        # 'San Francisco' : 'San Francisco, CA, USA',
        # 'San Diego'     : 'San Diego, CA, USA',
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

    if test_cities:
        places = {k: v for k, v in places.items() if k in test_cities}
        print(f"\n*** TEST MODE: Only processing {len(places)} cities: {list(places.keys())} ***\n")

    ox.settings.log_console = True
    ox.settings.use_cache = True
    ox.settings.timeout = 300  # Increase timeout to 300 seconds
    ox.settings.requests_timeout = 300

    print(f"\nAPI Settings:")
    print(f"  - Request delay: {API_DELAY} seconds")
    print(f"  - Timeout: 300 seconds")
    print(f"  - Max retries: {MAX_RETRIES}")
    print(f"  - Using cache: {ox.settings.use_cache}")
    if test_mode:
        print(f"\n*** TEST MODE ENABLED ***")
        print(f"  - Processing only {max_cbgs_test} CBGs per city")
        print(f"  - This is for quick testing only")
        print(f"  - Set test_mode=False for full processing\n")

    output_dir = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\SNHMI"

    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_results = []
    all_cbg_data = {}

    for city_short_name, place_name in places.items():
        try:
            results, cbg_indicators, cbg_gdf = calculate_city_road_quality_index(
                place_name,
                city_short_name,
                test_mode=test_mode,
                max_cbgs=max_cbgs_test
            )

            if results is not None:
                print(f"8. Creating road quality index map...")
                cbg_with_category = visualize_cbg_road_quality(
                    cbg_gdf,
                    cbg_indicators,
                    city_short_name,
                    output_dir
                )

                all_results.append(results)
                all_cbg_data[city_short_name] = cbg_with_category if cbg_with_category is not None else cbg_indicators

        except Exception as e:
            print(f"Error processing {city_short_name}: {e}")
            import traceback
            traceback.print_exc()

    df_results = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("All processing complete!")
    if test_mode:
        print("*** THIS WAS A TEST RUN ***")
        print(f"*** Only {max_cbgs_test} CBGs per city were processed ***")
        print("*** To process all CBGs, set test_mode=False ***")
    print(f"{'='*60}")

    print("\nSummary of HS-SNHMI:")
    display_cols = ['city', 'num_cbgs', 'test_mode', 'visitor_change', 'mean_road_index', 'std_road_index']
    available_cols = [col for col in display_cols if col in df_results.columns]
    print(df_results[available_cols].to_string())

    print("\n" + "="*80)
    print("Classification Distribution by City Type")
    print("="*80)

    CITY_TYPES = {
        'Adaptive/Baseline': ['Houston', 'Dallas', 'Chicago', 'Los_Angeles', 'Phoenix'],
        'Heat-sensitive': ['New_York', 'San_Francisco', 'San_Diego', 'San_Antonio', 'Philadelphia',
                          'Oklahoma', 'Austin', 'San_Jose'],
        'Temperature-Resilient': ['Jacksonville', 'Indianapolis', 'Denver', 'Columbus',
                                 'Charlotte', 'Nashville', 'Seattle']
    }

    if all_cbg_data:
        print("\n--- Composite_SNHMI Classification ---")

        for city_type, cities in CITY_TYPES.items():
            print(f"\n{city_type}:")

            type_cbgs = []
            for city in cities:
                if city in all_cbg_data:
                    city_df = all_cbg_data[city]
                    if 'category' in city_df.columns and 'Composite_SNHMI' in city_df.columns:
                        type_cbgs.append(city_df)

            if type_cbgs:
                combined = pd.concat(type_cbgs, ignore_index=True)
                total = len(combined)

                labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                print(f"  Total CBGs: {total}")
                for i, label in enumerate(labels):
                    count = (combined['category'] == i).sum()
                    pct = count / total * 100 if total > 0 else 0
                    print(f"    {label:15s}: {count:6d} ({pct:5.1f}%)")
            else:
                print("  No data available")

    print("="*80)

    return df_results, all_cbg_data

if __name__ == "__main__":
    TEST_MODE = False
    MAX_CBGS_TEST = 500
    TEST_CITIES = None

    df_results, all_cbg_data = main(
        test_mode=TEST_MODE,
        max_cbgs_test=MAX_CBGS_TEST,
        test_cities=TEST_CITIES
    )
