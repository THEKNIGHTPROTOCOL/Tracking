import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pydeck as pdk

# App config
st.set_page_config(page_title="📍 Terrorist GPS Tracking Dashboard", layout="wide")
st.title("📡 Terrorist Activity Tracking & Prediction Dashboard")

# Dataset URL (update if needed)
DATA_URL = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/tracking/refs/heads/main/terrorist_gps.csv"

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, on_bad_lines="skip")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")  # ensure datetime
        df = df.dropna(subset=["latitude", "longitude", "date"])  # clean invalid rows
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ No valid data found. Please check the dataset.")
    st.stop()

# Preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(20))

# Sidebar filters
st.sidebar.header("🔎 Filters")
group_filter = st.sidebar.multiselect("Select Groups", df["group"].unique(), default=df["group"].unique())

# Date range filter (fixed bug here ✅)
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = df["date"].min()
    end_date = df["date"].max()

filtered_df = df[
    (df["group"].isin(group_filter)) &
    (df["date"].between(start_date, end_date))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# ================== MAP VISUALIZATION ==================
st.subheader("🗺️ GPS Tracking Map")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position=["longitude", "latitude"],
    get_color=[255, 0, 0, 160],
    get_radius=2000,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=filtered_df["latitude"].mean(),
    longitude=filtered_df["longitude"].mean(),
    zoom=4,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{group} - {location}"}))

# ================== VISUALIZATIONS ==================
tab1, tab2, tab3 = st.tabs(["Activity Trends", "Group Distribution", "Insights"])

with tab1:
    st.markdown("### 📈 Activity Over Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    filtered_df.groupby("date").size().plot(ax=ax, color="red")
    ax.set_title("Incidents Over Time")
    st.pyplot(fig)

with tab2:
    st.markdown("### 🏴 Group Activity Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_df["group"].value_counts().plot(kind="bar", ax=ax, color="black")
    ax.set_title("Group Activity Counts")
    st.pyplot(fig)

with tab3:
    st.markdown("### 📌 Key Insights")
    st.write("- Activity spikes can be correlated with specific regions & groups.")
    st.write("- Some groups dominate activity in certain areas.")
    st.success("✅ Useful for monitoring hotspots & predicting threats.")
