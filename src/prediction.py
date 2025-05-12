import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
import matplotlib.pyplot as plt

def load_clustered_data(file_path):
    print(f"Loading clustered data from {file_path}")
    return pd.read_parquet(file_path)

def prepare_time_series_features(df):
    print("Preparing features for time series modeling...")

    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    elif 'tpep_pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    elif {'pickup_hour', 'pickup_day', 'pickup_month'}.issubset(df.columns):
        df['pickup_datetime'] = pd.to_datetime({
            'year': 2024,
            'month': df['pickup_month'],
            'day': df['pickup_day'],
            'hour': df['pickup_hour']
        }, errors='coerce')
    else:
        raise KeyError("No valid pickup datetime field found in the dataset.")

    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.weekday

    grouped = df.groupby(['zone_cluster', df['pickup_datetime'].dt.date, 'hour']).agg({
        'speed': 'mean'
    }).reset_index().rename(columns={
        'pickup_datetime': 'date',
        'speed': 'avg_speed'
    })

    grouped['date'] = pd.to_datetime(grouped['date'])
    grouped = grouped.sort_values(['zone_cluster', 'date', 'hour'])
    return grouped

def create_lag_features(df, lag_hours=[1, 2, 3]):
    print(f"Creating lag features: {lag_hours}")
    for lag in lag_hours:
        df[f'lag_{lag}'] = df.groupby('zone_cluster')['avg_speed'].shift(lag)
    df.dropna(inplace=True)
    return df

def train_xgboost_model(df):
    print("Training XGBoost model...")
    feature_cols = ['hour', 'lag_1', 'lag_2', 'lag_3']
    target_col = 'avg_speed'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"XGBoost RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # Optional: Visualize
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:100], label='Actual')
    plt.plot(preds[:100], label='Predicted')
    plt.title("Predicted vs Actual Speeds (Sample of 100)")
    plt.xlabel("Sample Index")
    plt.ylabel("Speed (mph)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

def validate_prediction(model, df, hour=15, zone_id=0):
    print(f"\nValidating prediction for zone {zone_id} at hour {hour}...")

    # Ensure datetime is parsed correctly
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    elif 'tpep_pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    elif {'pickup_hour', 'pickup_day', 'pickup_month'}.issubset(df.columns):
        df['pickup_datetime'] = pd.to_datetime({
            'year': 2024,
            'month': df['pickup_month'],
            'day': df['pickup_day'],
            'hour': df['pickup_hour']
        }, errors='coerce')
    else:
        raise KeyError("No valid pickup datetime field found in the dataset.")

    df['hour'] = df['pickup_datetime'].dt.hour
    df['date'] = df['pickup_datetime'].dt.date

    actual_speed = df[
        (df['zone_cluster'] == zone_id) & (df['hour'] == hour)
    ]['speed'].mean()

    sample = pd.DataFrame({
        'hour': [hour],
        'lag_1': [18.5],
        'lag_2': [19.2],
        'lag_3': [17.8]
    })
    predicted_speed = model.predict(sample)[0]

    print(f"Predicted speed: {predicted_speed:.2f} mph")
    print(f"Actual average speed: {actual_speed:.2f} mph")
    print(f"Absolute error: {abs(predicted_speed - actual_speed):.2f} mph")
    return predicted_speed, actual_speed

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def run_prediction_pipeline(input_path, model_output_path, validate=True):
    df = load_clustered_data(input_path)
    ts_df = prepare_time_series_features(df)
    ts_df = create_lag_features(ts_df)
    model = train_xgboost_model(ts_df)
    save_model(model, model_output_path)

    if validate:
        df_raw = pd.read_parquet(input_path)
        validate_prediction(model, df_raw, hour=15, zone_id=0)

    return model

if __name__ == "__main__":
    input_file = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\processed\clustered_trip_data.parquet"
    model_file = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\models\xgboost_model.pkl"
    run_prediction_pipeline(input_file, model_file)
