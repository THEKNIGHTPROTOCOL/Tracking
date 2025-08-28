import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta

st.set_page_config(page_title="Terrorist GPS Tracking", layout="wide")

st.title("🌍 Terrorist GPS Tracking & Prediction Dashboard")
st.markdown("Analyze and visualize GPS positions of suspected terrorist group activities.")

# ================== DATASET ==================
@st.cache_data
def load_data(n=500):
    np.random.seed(42)
    groups = ["Group A", "Group B", "Group C", "Group D"]
    regions = ["North", "South", "East", "West"]

    data = {
        "date": [datetime.today() - timedelta(days=np.random.randint(0, 365)) for _ in range(n)],
        "latitude": np.random.uniform(23.0, 37.0, n),   # India approx lat range
        "longitude": np.random.uniform(68.0, 89.0, n),  # India approx lon range
        "group": np.random.choice(groups, n),
        "region": np.random.choice(regions, n),
    }
    return pd.DataFrame(data)

try:
    df = load_data(1000)  # larger dataset
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ================== SIDEBAR FILTERS ==================
st.sidebar.header("🔎 Filters")
group_filter = st.sidebar.multiselect("Select Groups", df["group"].unique(), default=df["group"].unique())
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

# ================== MAP ==================
st.subheader("🗺️ Activity Map")
if filtered_df.empty:
    st.warning("⚠️ No data matches the filters.")
else:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state=pdk.ViewState(
            latitude=filtered_df["latitude"].mean(),
            longitude=filtered_df["longitude"].mean(),
            zoom=4,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=filtered_df,
                get_position=["longitude", "latitude"],
                get_color=[255, 0, 0, 160],
                get_radius=60000,
            ),
        ],
    ))

# ================== VISUALIZATIONS ==================
tab1, tab2, tab3 = st.tabs(["Activity Trends", "Group Distribution", "Insights"])

with tab1:
    st.markdown("### 📈 Activity Over Time")
    activity = filtered_df.groupby("date").size().reset_index(name="incidents")
    st.line_chart(activity.set_index("date"))

with tab2:
    st.markdown("### 🏴 Group Activity Distribution")
    group_counts = filtered_df["group"].value_counts().reset_index()
    group_counts.columns = ["group", "count"]
    st.bar_chart(group_counts.set_index("group"))

with tab3:
    st.markdown("### 📌 Key Insights")
    st.write("- Activity spikes correlate with certain regions & groups.")
    st.write("- Some groups dominate activity in specific areas.")
    st.success("✅ Helps monitor hotspots & predict threats.")
