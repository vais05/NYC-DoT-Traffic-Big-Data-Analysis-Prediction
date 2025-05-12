"""
Utility functions for NYC Taxi Data Analysis

This module provides common utility functions used across the project.
"""

import os
import logging
from datetime import datetime

def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Set up a logger with console and optional file handlers
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def get_timestamp_str():
    """
    Get current timestamp as a string
    
    Returns:
        String with timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lon1, lat1: Longitude and latitude of point 1
        lon2, lat2: Longitude and latitude of point 2
        
    Returns:
        Distance in miles
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    
    return c * r

def calculate_distance_matrix(coordinates):
    """
    Calculate distance matrix for a set of coordinates
    
    Args:
        coordinates: List of (longitude, latitude) tuples
        
    Returns:
        2D numpy array of distances
    """
    import numpy as np
    
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[j]
            dist = haversine_distance(lon1, lat1, lon2, lat2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Matrix is symmetric
    
    return dist_matrix

def estimate_processing_time(file_size, rows_per_gb=1000000):
    """
    Estimate processing time based on file size
    
    Args:
        file_size: Size of the file in bytes
        rows_per_gb: Estimated number of rows per GB
        
    Returns:
        Tuple of (estimated_rows, estimated_minutes)
    """
    gb_size = file_size / (1024**3)
    estimated_rows = int(gb_size * rows_per_gb)
    
    # Rough estimate: 1 million rows takes about 2 minutes to process
    estimated_minutes = estimated_rows / 500000
    
    return estimated_rows, estimated_minutes