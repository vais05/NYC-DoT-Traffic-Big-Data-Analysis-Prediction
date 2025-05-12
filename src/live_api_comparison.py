# src/live_api_comparison.py

import requests
import pandas as pd
import time
import os

HEIGIT_API_KEY = "5b3ce3597851110001cf6248a138f11ac7a141e083f6e5034494181f"
HEIGIT_API_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"

headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Authorization': HEIGIT_API_KEY,
    'Content-Type': 'application/json; charset=utf-8'
}

def load_trip_samples(parquet_path, n_samples=10):
    print(f"Loading {n_samples} random trips from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    sample_df = df[[
        'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon',
        'speed', 'pickup_zone', 'dropoff_zone'
    ]].dropna().sample(n=n_samples, random_state=42)
    return sample_df

def get_live_speed(pickup, dropoff):
    body = {
        "locations": [pickup, dropoff],
        "metrics": ["distance", "duration"],
        "units": "m"
    }
    try:
        response = requests.post(HEIGIT_API_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        distance_m = data['distances'][0][1]
        duration_s = data['durations'][0][1]
        if duration_s == 0:
            return None
        speed_mps = distance_m / duration_s
        speed_mph = speed_mps * 2.23694
        return speed_mph
    except Exception as e:
        print(f"API error: {e}")
        return None

def compare_with_live_data(df):
    print("Comparing predicted vs live traffic speed...")
    results = []

    for _, row in df.iterrows():
        pickup = [row['pickup_lon'], row['pickup_lat']]
        dropoff = [row['dropoff_lon'], row['dropoff_lat']]
        predicted_speed = row['speed']

        live_speed = get_live_speed(pickup, dropoff)
        time.sleep(1)

        if live_speed is not None:
            error = abs(predicted_speed - live_speed)
            results.append({
                'predicted_speed': predicted_speed,
                'live_speed': live_speed,
                'error': error,
                'pickup_zone': row.get('pickup_zone'),
                'dropoff_zone': row.get('dropoff_zone'),
                'lat': pickup[1],
                'lon': pickup[0]
            })

    return pd.DataFrame(results)

def run_live_comparison(input_path, output_csv="outputs/predictions/live_vs_predicted.csv", n_samples=10):
    df = load_trip_samples(input_path, n_samples=n_samples)
    df_live = compare_with_live_data(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_live.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    print(df_live.describe())

if __name__ == "__main__":
    input_file = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\processed\clustered_trip_data.parquet"
    run_live_comparison(input_file, n_samples=400)