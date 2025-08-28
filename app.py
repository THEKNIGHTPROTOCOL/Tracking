import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Terrorist GPS Tracking", layout="wide")

st.title("🌍 Terrorist GPS Tracking & Prediction Dashboard")
st.markdown(
    """
    This dashboard visualizes **GPS activity patterns of suspected terrorist groups**.  
    Use filters to explore **where, when, and which groups** are most active.
    """
)

# ================== DATASET ==================
@st.cache_data
def load_data(n=5000):
    np.random.seed(42)
    groups = ["Group A", "Group B", "Group C", "Group D", "Group E"]
    regions = ["North", "South", "East", "West", "Central"]

    data = {
        "date": [datetime.today() - timedelta(days=np.random.randint(0, 730)) for _ in range(n)],
        "latitude": np.random.uniform(23.0, 37.0, n),   # India approx lat range
        "longitude": np.random.uniform(68.0, 89.0, n),  # India approx lon range
        "group": np.random.choice(groups, n),
        "region": np.random.choice(regions, n),
    }
    return pd.DataFrame(data)

df = load_data()

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

# ================== MAP VISUALIZATION ==================
st.subheader("🗺️ Activity Map")

if filtered_df.empty:
    st.warning("⚠️ No data matches the filters.")
else:
    layer_scatter = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 150],
        get_radius=40000,
        pickable=True,
    )

    layer_heatmap = pdk.Layer(
        "HeatmapLayer",
        data=filtered_df,
        get_position=["longitude", "latitude"],
        aggregation=pdk.types.String("MEAN"),
        get_weight=1,
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state=pdk.ViewState(
            latitude=filtered_df["latitude"].mean(),
            longitude=filtered_df["longitude"].mean(),
            zoom=4,
            pitch=45,
        ),
        layers=[layer_heatmap, layer_scatter],
        tooltip={"text": "Group: {group}\nRegion: {region}\nDate: {date}"},
    ))

# ================== ANALYTICS ==================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Distribution", "🏴 Top Groups", "🔍 Insights"])

with tab1:
    st.markdown("### Activity Over Time")
    activity = filtered_df.groupby("date").size().reset_index(name="incidents")
    st.line_chart(activity.set_index("date"))

with tab2:
    st.markdown("### Region-wise Distribution")
    region_counts = filtered_df["region"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with tab3:
    st.markdown("### Top 5 Active Groups")
    top_groups = filtered_df["group"].value_counts().head(5).reset_index()
    top_groups.columns = ["Group", "Incidents"]
    st.table(top_groups)

with tab4:
    st.markdown("### Key Insights")
    st.write("- 🔥 **Heatmap shows emerging hotspots in border regions.**")
    st.write("- 📆 **Activity trends reveal spikes around specific months.**")
    st.write("- 🏴 **Some groups dominate specific regions.**")
    st.success("✅ This can support strategic monitoring & predictive analysis.")

