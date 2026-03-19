import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import atan2, degrees, radians, sqrt
import os
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class TravelDirectionAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cities = [
            'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Phoenix', 'NewYork',
            'SanFrancisco', 'SanDiego', 'SanAntonio', 'Philadelphia',
            'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
            'Indianapolis', 'Denver', 'Columbus', 'Charlotte',
            'Nashville', 'Seattle'
        ]
        self.direction_labels_8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.direction_labels_16 = [
            'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
        ]
        self.direction_labels_40 = [f'{i*9}°' for i in range(40)]
        self.colors_40 = self._generate_color_scheme()
        self.city_clusters = {
            'High Temperature Suppressed': ['Chicago', 'Dallas', 'Houston'],
            'Mid Temperature Stable': ['LosAngeles', 'Phoenix', 'NewYork'],
            'Large Metro Baseline': ['SanFrancisco', 'SanDiego', 'SanAntonio', 'Philadelphia',
                                   'Oklahoma', 'Austin', 'SanJose', 'Jacksonville',
                                   'Indianapolis', 'Denver', 'Columbus', 'Charlotte',
                                   'Nashville', 'Seattle']
        }
    
    def _generate_color_scheme(self):
        base_colors = ['#1f4e79', '#2e5984', '#3d6a91', '#4c7b9e', '#5b8cab', '#6a9db8']
        colors = []
        for i in range(40):
            base_idx = i % len(base_colors)
            colors.append(base_colors[base_idx])
        return colors
    
    def calculate_bearing(self, home_lat, home_lon, poi_lat, poi_lon):
        lat1, lon1 = radians(home_lat), radians(home_lon)
        lat2, lon2 = radians(poi_lat), radians(poi_lon)
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = degrees(atan2(y, x))
        bearing = (bearing + 360) % 360
        return bearing
    
    def calculate_distance(self, home_lat, home_lon, poi_lat, poi_lon):
        R = 6371.0
        lat1, lon1 = radians(home_lat), radians(home_lon)
        lat2, lon2 = radians(poi_lat), radians(poi_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        return distance
    
    def load_city_data(self, city_name):
        """
        Load 'Heat' and 'NoHeat' sheets for a specified city.
        """
        try:
            file_path = os.path.join(self.data_path, f'{city_name}.xls')
            df_dict = pd.read_excel(file_path, sheet_name=['Heat', 'NoHeat'])
            
            df_heat = df_dict.get('Heat')
            df_noheat = df_dict.get('NoHeat')
            
            if df_heat is None or df_noheat is None:
                print(f"Warning: {city_name}.xls file missing 'Heat' or 'NoHeat' sheet.")
                return None, None
                
            return df_heat, df_noheat
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, None
        except ValueError as e:
            print(f"Error loading {city_name} data: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error loading {city_name} data: {e}")
            return None, None
    
    def _process_sheet(self, df, city_name):
        if df is None or len(df) == 0:
            return None
        
        required_columns = ['home_lat', 'home_lon', 'latitude', 'longitude', 'distance_from_home']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: {city_name} data missing columns: {missing_columns}")
            return None
        
        df = df.dropna(subset=required_columns)
        df = df[(df['home_lat'].between(-90, 90)) & 
               (df['home_lon'].between(-180, 180)) &
               (df['latitude'].between(-90, 90)) & 
               (df['longitude'].between(-180, 180)) &
               (df['distance_from_home'] > 0)]
        
        if len(df) == 0:
            print(f"Warning: {city_name} has no remaining data after cleaning.")
            return None
            
        print(f"Successfully loaded and cleaned {city_name} (sheet: {len(df)} rows)")

        df['bearing'] = df.apply(lambda row: self.calculate_bearing(
            row['home_lat'], row['home_lon'], row['latitude'], row['longitude']), axis=1)
        df['distance'] = df.apply(lambda row: self.calculate_distance(
            row['home_lat'], row['home_lon'], row['latitude'], row['longitude']), axis=1)
        
        bearing_adjusted_8 = df['bearing'].copy()
        bearing_adjusted_8 = np.where(bearing_adjusted_8 >= 337.5, bearing_adjusted_8 - 360, bearing_adjusted_8)
        df['direction_8'] = pd.cut(bearing_adjusted_8, 
            bins=[-22.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
            labels=self.direction_labels_8, include_lowest=True)
        
        bearing_adjusted_16 = df['bearing'].copy()
        bearing_adjusted_16 = np.where(bearing_adjusted_16 >= 348.75, bearing_adjusted_16 - 360, bearing_adjusted_16)
        bins_16 = np.arange(-11.25, 348.75 + 22.5, 22.5)
        df['direction_16'] = pd.cut(bearing_adjusted_16, bins=bins_16,
            labels=self.direction_labels_16, include_lowest=True)
        
        bearing_adjusted_40 = df['bearing'].copy()
        bearing_adjusted_40 = np.where(bearing_adjusted_40 >= 355.5, bearing_adjusted_40 - 360, bearing_adjusted_40)
        bins_40 = np.arange(-4.5, 355.5 + 9, 9)
        df['direction_40'] = pd.cut(bearing_adjusted_40, bins=bins_40,
            labels=self.direction_labels_40, include_lowest=True)
        return df
    
    def process_city_data(self, df, city_name):
        return self._process_sheet(df, city_name)
    
    def create_polar_plot_single_city(self, df, city_name, direction_type='40', save_path=None):
        if df is None:
            print(f"Cannot create chart for {city_name}, data is empty")
            return None
            
        if direction_type == '40':
            direction_col = 'direction_40'
            direction_labels = self.direction_labels_40
        elif direction_type == '16':
            direction_col = 'direction_16'
            direction_labels = self.direction_labels_16
        else:
            direction_col = 'direction_8'
            direction_labels = self.direction_labels_8
        
        direction_stats = df.groupby(direction_col).agg({
            'distance_from_home': 'sum', 'distance': 'mean'
        }).reindex(direction_labels, fill_value=0)
        
        if not save_path:
            return direction_stats
            
        return direction_stats
    
    def create_comparison_plot(self, all_city_stats, direction_type='40', save_path=None):
        if not all_city_stats:
            return
        if direction_type == '40':
            direction_labels = self.direction_labels_40
            colors = self.colors_40
        elif direction_type == '16':
            direction_labels = self.direction_labels_16
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(direction_labels)))
        else:
            direction_labels = self.direction_labels_8
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(direction_labels)))
        
        n_cities = len(all_city_stats)
        cols = 4
        rows = (n_cities + cols - 1) // cols
        fig = plt.figure(figsize=(6*cols, 6*rows))
        angles = np.linspace(0, 2*np.pi, len(direction_labels), endpoint=False)
        
        for idx, (city, stats) in enumerate(all_city_stats.items(), 1):
            if stats is None:
                continue
            ax = fig.add_subplot(rows, cols, idx, projection='polar')
            distance_counts = stats['distance_from_home'].values
            total_distance = distance_counts.sum()
            distance_percentages = (distance_counts / total_distance * 100) if total_distance > 0 else distance_counts
            
            bar_colors = colors
            
            bars = ax.bar(angles, distance_percentages, width=2*np.pi/len(direction_labels), 
                         alpha=0.8, color=bar_colors, edgecolor='white', linewidth=0.5)
            
            if direction_type == '40':
                major_angles = angles[::5]
                major_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                ax.set_xticks(major_angles)
                ax.set_xticklabels(major_labels)
            else:
                ax.set_xticks(angles)
                ax.set_xticklabels(direction_labels, fontsize=8)
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title(f'{city}\n{total_distance:,.0f}', fontsize=10, fontweight='bold', pad=15)
            
            max_percentage = max(distance_percentages) if max(distance_percentages) > 0 else 10
            circle_max = np.ceil(max_percentage / 2) * 2
            circle_levels = np.linspace(0, circle_max, 5)[1:]
            ax.set_ylim(0, circle_max)
            ax.set_yticks(circle_levels)
            ax.set_yticklabels([f'{level:.0f}%' for level in circle_levels], fontsize=8)
            
            for level in circle_levels:
                circle = plt.Circle((0, 0), level, fill=False, color='lightgray', 
                                  linewidth=0.8, alpha=0.6, transform=ax.transData._b)
                ax.add_patch(circle)
            ax.grid(True, alpha=0.3, color='lightgray')
            ax.set_facecolor('white')
        
        plt.tight_layout()
        if save_path:
            output_file = os.path.join(save_path, 'S10a. Polar distribution of distance.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparison plot saved to: {output_file}")
        plt.close()
    
    def analyze_all_cities(self, direction_type='40', save_plots=True, output_dir='./polar_analysis_output'):
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        all_city_stats = {}
        all_city_data = {}
        
        print("Starting analysis of city travel directions...")
        print("="*50)
        
        for city in self.cities:
            print(f"Processing {city}...")
            df_heat, df_noheat = self.load_city_data(city)
            
            if df_heat is not None:
                df = df_heat
            else:
                continue
                
            processed_df = self.process_city_data(df, city)
            if processed_df is None:
                continue
                
            direction_col = f'direction_{direction_type}'
            if direction_col not in processed_df.columns:
                print(f"Warning: {direction_col} not found, using default direction_8")
                direction_type_used = '8'
            else:
                direction_type_used = direction_type
                
            stats = self.create_polar_plot_single_city(
                processed_df, city, direction_type=direction_type_used,
                save_path=None)
            
            all_city_stats[city] = stats
            all_city_data[city] = processed_df
        
        print(f"Successfully analyzed {len(all_city_stats)} cities")
        
        if len(all_city_stats) > 1:
            print(f"\nCreating ONLY city comparison plot ({direction_type} directions)...")
            self.create_comparison_plot(all_city_stats, direction_type=direction_type,
                save_path=output_dir if save_plots else None)
            
        return all_city_stats, all_city_data


def main():
    data_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Heat_NoHeat_OD_dist"
    output_path = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code"
    
    analyzer = TravelDirectionAnalyzer(data_path)
    
    all_stats, all_data = analyzer.analyze_all_cities(direction_type='40', save_plots=True, output_dir=output_path)
    print(f"\nAnalysis complete! ONLY 'S10a Polar distribution of distance.png' saved to '{output_path}' directory")
    
    return analyzer, all_stats, all_data


if __name__ == "__main__":
    analyzer, stats, data = main()
    
