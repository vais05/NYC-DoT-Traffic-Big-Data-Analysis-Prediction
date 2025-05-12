# src/visualization.py

import folium
import pandas as pd
from folium.plugins import MarkerCluster
import os

def load_clustered_data(file_path):
    print(f"Loading clustered trip data from {file_path}")
    return pd.read_parquet(file_path)

def create_color_map(zone_label):
    color_map = {
        'red': 'red',
        'yellow': 'orange',
        'green': 'green'
    }
    return color_map.get(zone_label, 'blue')

def plot_clusters_on_map(df, output_html):
    print("Plotting traffic zones on NYC map...")
    
    # Initialize Folium map centered around NYC
    nyc_center = [40.7128, -74.0060]
    folium_map = folium.Map(location=nyc_center, zoom_start=11)

    # Clustered markers
    marker_cluster = MarkerCluster().add_to(folium_map)

    # Plot points
    for _, row in df.iterrows():
        lat = row['pickup_lat']
        lon = row['pickup_lon']
        label = row['traffic_zone']
        color = create_color_map(label)

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Zone: {label} | Speed: {row['speed']:.2f} mph"
        ).add_to(marker_cluster)

    # Save map to HTML
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    folium_map.save(output_html)
    print(f"Map saved to {output_html}")

def generate_map(file_path, output_html, sample_frac=0.01):
    df = load_clustered_data(file_path)

    # Downsample for HTML performance (default 1% of data)
    if sample_frac < 1.0:
        original_count = len(df)
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(df)} records from {original_count} (fraction={sample_frac})")

    plot_clusters_on_map(df, output_html)

if __name__ == "__main__":
    input_file = "data/processed/clustered_trip_data.parquet"
    output_map = "outputs/maps/nyc_traffic_zones_map.html"
    generate_map(input_file, output_map, sample_frac=0.10)
