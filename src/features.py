import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os

def load_data(file_path):
    """
    Load data from a Parquet file
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    print(f"Loading data from {file_path}")
    return pd.read_parquet(file_path)

def extract_geo_features(df):
    """
    Extract geographical features from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with geographical data
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added geographical features
    """
    print("Extracting geographical features...")
    
    # Calculate distance between pickup and dropoff (if not already calculated)
    if 'trip_distance' not in df.columns:
        # Calculate haversine distance if coordinates are available
        if all(col in df.columns for col in ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon']):
            df['trip_distance'] = calculate_haversine_distance(
                df['pickup_lat'], df['pickup_lon'], 
                df['dropoff_lat'], df['dropoff_lon']
            )
    
    # Calculate pickup and dropoff density (trips per zone)
    if 'pickup_zone' in df.columns:
        pickup_counts = df['pickup_zone'].value_counts()
        df['pickup_zone_density'] = df['pickup_zone'].map(pickup_counts)
    
    if 'dropoff_zone' in df.columns:
        dropoff_counts = df['dropoff_zone'].value_counts()
        df['dropoff_zone_density'] = df['dropoff_zone'].map(dropoff_counts)
    
    # Calculate zone-to-zone flow
    if 'pickup_zone' in df.columns and 'dropoff_zone' in df.columns:
        flow_counts = df.groupby(['pickup_zone', 'dropoff_zone']).size().reset_index(name='zone_flow')
        df = pd.merge(
            df, 
            flow_counts, 
            on=['pickup_zone', 'dropoff_zone'], 
            how='left'
        )
    
    return df

def extract_time_features(df):
    """
    Extract time-based features from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with datetime data
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added time features
    """
    print("Extracting time features...")
    
    # Convert datetime columns if they're strings
    for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col])
    
    # Trip duration in minutes (if not already calculated)
    if 'duration' not in df.columns and all(col in df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
        df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Calculate speed (miles per hour) if not already calculated
    if 'speed' not in df.columns and all(col in df.columns for col in ['trip_distance', 'duration']):
        # Convert duration from minutes to hours for mph calculation
        df['speed'] = df['trip_distance'] / (df['duration'] / 60)
        # Remove unrealistic speeds (e.g., > 100 mph)
        df.loc[df['speed'] > 100, 'speed'] = np.nan
    
    # Extract time components from pickup datetime
    if 'tpep_pickup_datetime' in df.columns:
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['pickup_year'] = df['tpep_pickup_datetime'].dt.year
        df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
        df['pickup_is_weekend'] = df['pickup_weekday'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Cyclical encoding of time features
        df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['pickup_weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['pickup_weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
        
        # Time of day category
        hours = df['pickup_hour']
        df['time_of_day'] = pd.cut(
            hours, 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Rush hour flag (7-10 AM and 4-7 PM)
        df['is_rush_hour'] = ((hours >= 7) & (hours <= 10)) | ((hours >= 16) & (hours <= 19))
        df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    
    return df

def extract_trip_features(df):
    """
    Extract trip-related features from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with trip data
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added trip features
    """
    print("Extracting trip features...")
    
    # Calculate fare per mile
    if all(col in df.columns for col in ['fare_amount', 'trip_distance']) and 'fare_per_mile' not in df.columns:
        df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']
        # Handle division by zero or very small distances
        df.loc[df['trip_distance'] < 0.1, 'fare_per_mile'] = np.nan
    
    # Calculate tip percentage
    if all(col in df.columns for col in ['tip_amount', 'fare_amount']) and 'tip_percentage' not in df.columns:
        df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']) * 100
        # Handle division by zero or negative fares
        df.loc[df['fare_amount'] <= 0, 'tip_percentage'] = np.nan
    
    # Passenger count features
    if 'passenger_count' in df.columns:
        # Fill missing passenger counts
        df['passenger_count'] = df['passenger_count'].fillna(1)
        # Create passenger count categories
        df['passenger_group'] = pd.cut(
            df['passenger_count'], 
            bins=[0, 1, 2, 4, 10], 
            labels=['single', 'couple', 'small_group', 'large_group'],
            include_lowest=True
        )
        
    # Calculate congestion features
    if 'speed' in df.columns:
        # Speed categories
        df['speed_category'] = pd.cut(
            df['speed'], 
            bins=[0, 5, 15, 30, 100], 
            labels=['very_slow', 'slow', 'normal', 'fast'],
            include_lowest=True
        )
    
    return df

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in miles
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of the first point
    lat2, lon2 : float
        Latitude and longitude of the second point
        
    Returns:
    --------
    float
        Distance in miles
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of earth in miles
    
    return c * r

def engineer_features(df):
    """
    Apply all feature engineering steps to the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with all engineered features
    """
    print("Starting feature engineering...")
    
    # Apply each feature engineering step
    df = extract_geo_features(df)
    df = extract_time_features(df)
    df = extract_trip_features(df)
    
    # Drop unnecessary columns after feature engineering
    cols_to_drop = [
        # Drop original datetime columns as we've extracted features from them
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        # Drop any other columns that are no longer needed
    ]
    
    # Only drop columns that exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    print("Feature engineering complete.")
    return df

def prepare_for_pca(df, numeric_features=None):
    """
    Prepare data for PCA by selecting numeric features and applying scaling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with all features
    numeric_features : list, optional
        List of numeric feature columns to use for PCA. If None, all numeric columns will be used.
        
    Returns:
    --------
    tuple
        (scaled_data, scaler, feature_names)
    """
    print("Preparing data for PCA...")
    
    # If no features specified, use all numeric columns
    if numeric_features is None:
        # Get all numeric columns
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove any target columns or identifiers
        exclude_cols = ['VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type', 'zone', 'pickup_zone', 'dropoff_zone']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
    
    # Handle missing values
    df_numeric = df[numeric_features].copy()
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    print(f"Data prepared for PCA with {len(numeric_features)} features.")
    return scaled_data, scaler, numeric_features

def apply_pca(scaled_data, variance_threshold=0.95, max_components=None):
    """
    Apply PCA to the scaled data, retaining specified variance
    
    Parameters:
    -----------
    scaled_data : numpy.ndarray
        Scaled input data
    variance_threshold : float, optional
        Amount of variance to retain (default: 0.95)
    max_components : int, optional
        Maximum number of components to retain
        
    Returns:
    --------
    tuple
        (pca_transformed_data, pca_model, explained_variance_ratio)
    """
    print(f"Applying PCA with {variance_threshold} variance threshold...")
    
    # Apply PCA with specified variance retention
    pca = PCA(n_components=variance_threshold)
    pca_transformed = pca.fit_transform(scaled_data)
    
    # Get explained variance for each component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Determine how many components were needed to reach the threshold
    n_components = len(cumulative_variance[cumulative_variance <= variance_threshold]) + 1
    
    # Limit to max_components if specified
    if max_components is not None and n_components > max_components:
        n_components = max_components
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
    
    print(f"PCA complete. Retained {n_components} components explaining {np.sum(explained_variance):.2%} of variance.")
    return pca_transformed, pca, explained_variance

def get_feature_importance(pca, feature_names):
    """
    Get feature importance based on PCA loadings
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importance for each component
    """
    # Get the loadings
    loadings = pca.components_.T
    
    # Create a DataFrame with the loadings
    loadings_df = pd.DataFrame(
        loadings, 
        columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
        index=feature_names
    )
    
    # Calculate absolute importance
    abs_loadings = np.abs(loadings_df)
    importance = abs_loadings.sum(axis=1)
    
    # Create and return a DataFrame with feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def add_pca_components_to_df(df, pca_transformed, n_components=5):
    """
    Add PCA components to the original dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    pca_transformed : numpy.ndarray
        PCA transformed data
    n_components : int, optional
        Number of components to add (default: 5)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added PCA components
    """
    n_components = min(n_components, pca_transformed.shape[1])
    for i in range(n_components):
        df[f'pca_component_{i+1}'] = pca_transformed[:, i]
    
    return df

def save_model(model, file_path):
    """
    Save a model to a file
    
    Parameters:
    -----------
    model : object
        Model to save
    file_path : str
        Path to save the model
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a model from a file
    
    Parameters:
    -----------
    file_path : str
        Path to the model file
        
    Returns:
    --------
    object
        Loaded model
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model

def process_data(input_path, output_path=None, pca_model_path=None, scaler_model_path=None):
    """
    Process the data through feature engineering and PCA
    
    Parameters:
    -----------
    input_path : str
        Path to the input Parquet file
    output_path : str, optional
        Path to save the output Parquet file
    pca_model_path : str, optional
        Path to save the PCA model
    scaler_model_path : str, optional
        Path to save the scaler model
        
    Returns:
    --------
    tuple
        (processed_df, pca_model, scaler)
    """
    # Load the data
    df = load_data(input_path)
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Prepare data for PCA
    scaled_data, scaler, feature_names = prepare_for_pca(df)
    
    # Apply PCA
    pca_transformed, pca_model, explained_variance = apply_pca(scaled_data)
    
    # Add top PCA components to the dataframe
    df = add_pca_components_to_df(df, pca_transformed)
    
    # Get feature importance
    importance_df = get_feature_importance(pca_model, feature_names)
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Save processed data if output path is provided
    if output_path:
        df.to_parquet(output_path)
        print(f"Processed data saved to {output_path}")
    
    # Save models if paths are provided
    if pca_model_path:
        save_model(pca_model, pca_model_path)
    
    if scaler_model_path:
        save_model(scaler, scaler_model_path)
    
    return df, pca_model, scaler, feature_names


if __name__ == "__main__":
    # Define paths
    input_path = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\processed\full_trip_data_cleaned.parquet"
    output_path = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\processed\full_trip_data_with_features.parquet"
    pca_model_path = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\models\pca_model.pkl"
    scaler_model_path = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\models\scaler_model.pkl"
    
    # Process the data
    df, pca_model, scaler, feature_names = process_data(
        input_path=input_path,
        output_path=output_path,
        pca_model_path=pca_model_path,
        scaler_model_path=scaler_model_path
    )
