# NYC-DoT-Traffic-Big-Data-Analysis-Prediction
🔧 Step 1: Data Ingestion and Storage
Notebook: 01_data_ingestion.ipynb
Script: src/ingestion.py
Tools: PySpark, Pandas
Tasks:
•	Load 12 monthly Parquet files using PySpark into a unified DataFrame.
•	Load latitude/longitude zone mapping CSV using Pandas.
•	Merge if needed and save combined DataFrame to data/processed/ as full_trip_data.parquet.
This step sets the foundation with scalable, distributed loading.
________________________________________ Step 2: Data Cleaning & Preprocessing
Notebook: 02_preprocessing.ipynb
 Script: src/preprocessing.py
Tools: PySpark
Tasks:
•	Drop nulls and filter invalid entries (e.g., negative or zero lat/lon, impossible durations).
•	Extract time features: hour, day, weekday, month from pickup datetime.
•	Calculate trip distance using Haversine formula.
•	Calculate speed = distance / duration.
•	Remove outliers based on thresholds (e.g., speed > 100 mph).
Resulting cleaned data is ready for advanced feature engineering.
________________________________________
Step 3: Feature Engineering & PCA
Notebook: 03_feature_engineering_pca.ipynb
 Script: src/features.py
Tools: PySpark, scikit-learn
Tasks:
•	Add engineered features: pickup/dropoff lat/lon, duration, speed, time of day, etc.
•	Normalize features.
•	Apply PCA to reduce dimensions while preserving 95% variance.
Reduces noise and computational burden in downstream models.
________________________________________
Step 4: Clustering (Red/Yellow/Green Zones)
Notebook: 04_clustering_zones.ipynb
Script: src/clustering.py
Algorithms: KMeans 
Tasks:
•	Use pickup and dropoff locations to cluster trips.
•	Assign each cluster a zone ID.
•	Label clusters based on average speed:
o	Red: Congested
o	Yellow: Moderate
o	Green: Free-flowing
Creates spatial segmentation to analyze traffic by zones.
________________________________________
Step 5: Time-Series Prediction (Traffic Speed Over Time)
Notebook: 05_time_series_prediction.ipynb
Script: src/prediction.py
Models: XGBoost, LSTM, Prophet
Tasks:
•	Feature engineering for temporal patterns:
o	Lag features (1h, 6h, 24h)
o	Rolling means/std
o	Calendar features (weekend, holiday)
o	Spatial context (zone neighbor speed)
•	Train models to predict average speed per zone for the next 1–24 hours.
•	Use ensemble of XGBoost + LSTM + Prophet with weighted averaging.
Captures both short-term and long-term traffic trends.
________________________________________
Step 6: Visualization on NYC Map
Notebook: 06_map_visualization.ipynb
Script: src/visualization.py
Tools: Folium, Kepler.gl, Plotly
Tasks:
•	Visualize clustered zones on a NYC map (Red/Yellow/Green).
•	Animate traffic changes over time using a time slider.
•	Export as HTML map for interactive use.
 Communicates insights effectively and enables spatial validation.
________________________________________
 Step 7: Live Traffic Comparison
Notebook: 07_live_api_comparison.ipynb
Script: src/live_api_comparison.py
APIs: Google Maps, TomTom, HERE
Tasks:
•	Query live traffic speed for zone pairs via external API.
•	Compare live with model-predicted.
•	Compute accuracy/deviation to monitor drift.
Enables real-time validation and retraining triggers.
________________________________________
Dashboard
File: dashboard/app.py
Tool: Streamlit
Features:
•	Zone-wise visualization
•	Interactive filters (time, date, zone)
•	Real-time vs predicted traffic comparison
Ideal for stakeholders or traffic managers.
