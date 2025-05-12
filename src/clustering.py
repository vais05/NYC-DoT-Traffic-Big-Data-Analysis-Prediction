# src/clustering.py

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os

def load_feature_data(input_path):
    print(f"Loading PCA features from {input_path}")
    return pd.read_parquet(input_path)

def run_kmeans_clustering(df, n_clusters=10, pca_cols_prefix='pca_component_'):
    print(f"Running KMeans clustering with {n_clusters} clusters...")
    pca_cols = [col for col in df.columns if col.startswith(pca_cols_prefix)]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['zone_cluster'] = kmeans.fit_predict(df[pca_cols])
    return df, kmeans

def assign_zone_labels(df):
    print("Assigning red/yellow/green labels based on average speed...")
    cluster_speed = df.groupby('zone_cluster')['speed'].mean().sort_values()
    cluster_labels = {}
    total_clusters = len(cluster_speed)
    for i, cluster_id in enumerate(cluster_speed.index):
        if i < total_clusters / 3:
            cluster_labels[cluster_id] = 'red'
        elif i < 2 * total_clusters / 3:
            cluster_labels[cluster_id] = 'yellow'
        else:
            cluster_labels[cluster_id] = 'green'
    df['traffic_zone'] = df['zone_cluster'].map(cluster_labels)
    return df

def save_clustered_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Clustered data saved to {output_path}")

def run_clustering_pipeline(input_path, output_path, n_clusters=10):
    df = load_feature_data(input_path)
    df, _ = run_kmeans_clustering(df, n_clusters=n_clusters)
    df = assign_zone_labels(df)
    save_clustered_data(df, output_path)
    return df

if __name__ == "__main__":
    input_file = "data/processed/full_trip_data_with_features.parquet"
    output_file = "data/processed/clustered_trip_data.parquet"
    run_clustering_pipeline(input_file, output_file, n_clusters=10)
