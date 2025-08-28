import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="📍 Terrorist GPS Tracking", layout="wide")
st.title("📍 Terrorist GPS Tracking & Place Predictions")

# Sample dataset (replace with real one)
@st.cache_data
def load_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "latitude": np.random.uniform(24, 37, 200),
        "longitude": np.random.uniform(60, 77, 200),
        "group": np.random.choice(["Group A", "Group B", "Group C"], 200),
        "date": pd.date_range("2024-01-01", periods=200, freq="D")
    })
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filters")
group_filter = st.sidebar.multiselect("Select Groups", df["group"].unique(), default=df["group"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

# Filter data
filtered_df = df[
    (df["group"].isin(group_filter)) &
    (df["date"].between(date_range[0], date_range[1]))
]

st.subheader("📊 Dataset Preview")
st.dataframe(filtered_df.head(20))

# Map with activity points + heatmap
st.subheader("🌍 Activity Map")
m = folium.Map(location=[30, 70], zoom_start=4, tiles="CartoDB positron")

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        popup=f"Group: {row['group']} | Date: {row['date']}",
        color="red",
        fill=True
    ).add_to(m)

HeatMap(filtered_df[["latitude", "longitude"]].values, radius=15).add_to(m)
st_map = st_folium(m, width=800, height=500)

# Predictive Clustering
st.subheader("📌 Hotspot Predictions (KMeans)")
num_clusters = st.slider("Number of clusters", 2, 8, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
coords = filtered_df[["latitude", "longitude"]]
kmeans.fit(coords)
filtered_df["cluster"] = kmeans.labels_

st.write("Cluster Centers (Predicted Hotspots):")
st.write(pd.DataFrame(kmeans.cluster_centers_, columns=["latitude", "longitude"]))

m2 = folium.Map(location=[30, 70], zoom_start=4, tiles="CartoDB positron")
for cluster_id, row in enumerate(kmeans.cluster_centers_):
    folium.Marker(
        location=row,
        popup=f"Cluster {cluster_id}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m2)

st_folium(m2, width=800, height=500)
