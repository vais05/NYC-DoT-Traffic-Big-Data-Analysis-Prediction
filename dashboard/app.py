# dashboard/app.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# Paths
CLUSTERED_DATA_PATH = "data/processed/clustered_trip_data.parquet"
LIVE_RESULTS_PATH = "outputs/predictions/live_vs_predicted.csv"

@st.cache_data
def load_clustered_data():
    return pd.read_parquet(CLUSTERED_DATA_PATH)

@st.cache_data
def load_comparison_data():
    if os.path.exists(LIVE_RESULTS_PATH):
        return pd.read_csv(LIVE_RESULTS_PATH)
    else:
        return pd.DataFrame()

def create_interactive_map(df, selected_point=None, map_title="Predicted", is_actual=False):
    color_map = {"red": "red", "yellow": "orange", "green": "green"}
    nyc_center = [40.7128, -74.0060]
    folium_map = folium.Map(location=nyc_center, zoom_start=11)

    for _, row in df.iterrows():
        lat = row.get("pickup_lat") if not is_actual else row.get("lat")
        lon = row.get("pickup_lon") if not is_actual else row.get("lon")
        popup_text = (
            f"Zone: {row.get('traffic_zone', '-')}, Speed: {row.get('speed', 0):.2f} mph"
            if not is_actual else
            f"Actual: {row['live_speed']:.2f} mph"
        )
        color = color_map.get(row.get("traffic_zone"), "purple") if not is_actual else "purple"

        radius = 4
        if selected_point:
            try:
                if abs(lat - selected_point['lat']) < 0.0001 and abs(lon - selected_point['lng']) < 0.0001:
                    radius = 6
                    color = "black"
            except KeyError:
                pass

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        )
        marker.add_to(folium_map)

    return folium_map

def main():
    st.set_page_config(page_title="NYC Traffic Zone Dashboard", layout="wide")
    st.title("🚦 NYC Traffic Analysis Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_zone = st.sidebar.selectbox("Select Traffic Zone", options=["All", "red", "yellow", "green"])
    sample_size = st.sidebar.slider("Number of Points", 100, 2000, 500)

    df = load_clustered_data()
    comp_df = load_comparison_data()
    df_sample = df.sample(n=sample_size, random_state=42)

    if selected_zone != "All":
        df_sample = df_sample[df_sample["traffic_zone"] == selected_zone]

    st.subheader("🗺️ Live vs Predicted Traffic Comparison")
    col1, col2 = st.columns(2)

    selected_point = None

    with col2:
        st.markdown("### 🟣 Actual Traffic Map")
        map_return = None
        if not comp_df.empty and {'lat', 'lon', 'predicted_speed', 'live_speed'}.issubset(comp_df.columns):
            map_return = st_folium(
                create_interactive_map(comp_df, map_title="Actual", is_actual=True),
                width=450,
                height=500,
                returned_objects=["last_clicked"]
            )
        else:
            st.warning("Live comparison data missing coordinates or empty.")

        if map_return:
            selected_point = map_return.get("last_clicked")

    with col1:
        st.markdown("### 🔵 Predicted Traffic Map")
        st_folium(
            create_interactive_map(df_sample, selected_point=selected_point),
            width=450,
            height=500
        )

    # User dropdowns for zone speed lookup
    st.subheader("🔍 Select Dropoff and Pickup Zones for Speed Lookup")
    available_pickups = df["pickup_zone"].dropna().unique()
    available_dropoffs = df["dropoff_zone"].dropna().unique()

    selected_pickup = st.selectbox("Select Pickup Zone", sorted(available_pickups))
    selected_dropoff = st.selectbox("Select Dropoff Zone", sorted(available_dropoffs))

    match_rows = df[(df["pickup_zone"] == selected_pickup) & (df["dropoff_zone"] == selected_dropoff)]

    if not match_rows.empty:
        avg_predicted = match_rows["speed"].mean()
        st.success(f"🚗 Predicted Average Speed: {avg_predicted:.2f} mph")
    else:
        st.warning("No predicted data available for selected route.")

    if not comp_df.empty and {'pickup_zone', 'dropoff_zone', 'live_speed'}.issubset(comp_df.columns):
        actual_match = comp_df[(comp_df['pickup_zone'] == selected_pickup) & (comp_df['dropoff_zone'] == selected_dropoff)]
        if not actual_match.empty:
            avg_actual = actual_match['live_speed'].mean()
            st.info(f"📡 Actual Average Speed: {avg_actual:.2f} mph")
        else:
            st.warning("No actual data available for selected route.")

if __name__ == "__main__":
    main()