"""
Configuration settings for NYC Taxi Data Analysis project

This module defines constants, file paths, and configuration variables
used throughout the project.
"""

import os
from pathlib import Path
import h3
import yaml
# Define project root path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Create path constants
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, 'outputs')
LOGS_PATH = os.path.join(OUTPUTS_PATH, 'logs')
EXTERNAL_DATA_PATH = os.path.join(DATA_PATH, 'external')
# Ensure that these directories exist
for path in [DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH, OUTPUTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# Filenames
MAIN_PARQUET_FILE = os.path.join(PROCESSED_DATA_PATH, 'full_trip_data.parquet')
PREPROCESSED_FILE = os.path.join(PROCESSED_DATA_PATH, 'preprocessed_trip_data.parquet')
CLUSTERED_FILE = os.path.join(PROCESSED_DATA_PATH, 'clustered_trip_data.parquet')
FEATURES_FILE = os.path.join(PROCESSED_DATA_PATH, 'features_trip_data.parquet')

# NYC geographical boundaries (approximate)
NYC_MIN_LAT = 40.4774
NYC_MAX_LAT = 40.9176
NYC_MIN_LON = -74.2591
NYC_MAX_LON = -73.7004

# Data cleaning thresholds
MIN_TRIP_DISTANCE = 0.01  # miles
MAX_TRIP_DISTANCE = 100  # miles
MIN_DURATION = 30  # seconds
MAX_DURATION = 10800  # seconds (3 hours)
MIN_FARE = 0  # dollars
MAX_SPEED = 100  # mph

# Time features
RUSH_HOURS_MORNING = (7, 10)  # 7:00 AM - 10:00 AM
RUSH_HOURS_EVENING = (16, 19)  # 4:00 PM - 7:00 PM

# Clustering parameters
H3_RESOLUTION = 8  # Resolution for H3 hexagonal grid
DBSCAN_EPS = 0.01  # Epsilon parameter for DBSCAN (in degrees)
DBSCAN_MIN_SAMPLES = 5  # Minimum samples for DBSCAN

# Speed calculation
MPH_TO_KMH = 1.60934  # Conversion factor from mph to km/h

# Model training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering
TARGET_COLUMN = 'speed_mph'
CATEGORICAL_FEATURES = ['pickup_weekday', 'is_weekend', 'time_of_day']
NUMERICAL_FEATURES = ['trip_distance', 'fare_amount', 'hour_sin', 'hour_cos', 
                      'weekday_sin', 'weekday_cos']

# Visualization settings
MAP_CENTER_LAT = 40.7128
MAP_CENTER_LON = -74.0060
MAP_ZOOM_START = 11

# H3 functions
def get_h3_index(lat, lon, resolution=H3_RESOLUTION):
    """
    Get H3 index for a given latitude and longitude
    
    Args:
        lat: Latitude
        lon: Longitude
        resolution: H3 resolution (default: 8)
        
    Returns:
        H3 index as string
    """
    return h3.geo_to_h3(lat, lon, resolution)

def h3_to_coordinates(h3_index):
    """
    Convert H3 index to coordinates
    
    Args:
        h3_index: H3 index as string
        
    Returns:
        (latitude, longitude) tuple
    """
    geo_boundary = h3.h3_to_geo(h3_index)
    return geo_boundary[0], geo_boundary[1]  # lat, lon

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../configs/mapbox_config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
