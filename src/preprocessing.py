import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from math import sin, cos, pi

# Load data
def load_data(input_data):
    print('Loading data...')
    if isinstance(input_data, str):
        try:
            if input_data.endswith('.parquet'):
                df = pd.read_parquet(input_data)
            elif input_data.endswith('.csv'):
                df = pd.read_csv(input_data)
            else:
                raise ValueError('Unsupported file format')
            print(f'Data loaded from {input_data}')
        except Exception as e:
            print(f'Error loading data: {e}')
            return None
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
        print('Data loaded from DataFrame')
    else:
        raise ValueError('Input must be a file path or a DataFrame')
    return df

# Data Cleaning
def clean_data(df):
    print('Cleaning data...')
    df = df[df['trip_distance'] > 0]
    df = df[df['fare_amount'] >= 0]
    df = df[df['tpep_pickup_datetime'] != df['tpep_dropoff_datetime']]
    df.dropna(inplace=True)
    return df

# Calculating Derived Fields
def calculate_fields(df):
    print('Calculating derived fields...')
    df['duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 60
    df['speed'] = df['trip_distance'] / (df['duration'] / 60)
    return df

# Extracting Time Features
def extract_time_features(df):
    print('Extracting time features...')
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['pickup_day'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.day
    df['pickup_weekday'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.weekday
    df['is_weekend'] = df['pickup_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour_sin'] = np.sin(2 * pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * pi * df['pickup_hour'] / 24)
    return df

# Geospatial Clustering using MiniBatchKMeans for better memory management
def cluster_zones(df, n_clusters=10, batch_size=1000):
    print('Performing geospatial clustering...')
    # Optimize memory usage by using float32 for coordinates
    df[['pickup_lon', 'pickup_lat']] = df[['pickup_lon', 'pickup_lat']].astype('float32')
    
    # Check for missing values in the coordinates and drop rows with NaNs
    df_clean = df.dropna(subset=['pickup_lon', 'pickup_lat'])
    
    coords = df_clean[['pickup_lon', 'pickup_lat']].values
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    df_clean['zone'] = kmeans.fit_predict(coords)
    
    # Merge the cluster labels back into the original DataFrame
    df = pd.merge(df, df_clean[['zone']], left_index=True, right_index=True, how='left')
    
    return df

# Main processing function
def preprocess(input_data):
    df = load_data(input_data)
    if df is None:
        print('Preprocessing aborted due to data loading error.')
        return None
    
    # Downsample the data (optional, to speed up clustering step)
    df_sampled = df.sample(frac=0.1, random_state=42)  # Sample 10% of the data for clustering
    print('Downsampling completed.')

    # Data cleaning and derived field calculations
    df_sampled = clean_data(df_sampled)
    df_sampled = calculate_fields(df_sampled)
    df_sampled = extract_time_features(df_sampled)

    # Perform clustering on the downsampled data
    df_sampled = cluster_zones(df_sampled)
    
    # After clustering, apply the same clustering model to the full dataset
    df = clean_data(df)  # Clean the full dataset for clustering
    df = calculate_fields(df)
    df = extract_time_features(df)
    
    # Perform clustering on the full dataset
    df = cluster_zones(df)

    return df

# Example usage
if __name__ == '__main__':
    processed_df = preprocess('data/processed/full_trip_data.parquet')
    if processed_df is not None:
        processed_df.to_parquet('data/processed/full_trip_data_cleaned.parquet')
        print('Preprocessing completed successfully.')
    else:
        print('Preprocessing failed.')
