import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Terrorist GPS Intelligence Dashboard", layout="wide")

st.title("🛰 Terrorist GPS Intelligence & Prediction System")
st.markdown(
    """
    This system provides *real-time intelligence visualization* of suspected terrorist movements.  
    Use interactive filters, timelines, and hotspots to track *who, where, and when* activity occurs.  
    """
)

# ================== DATASET ==================
@st.cache_data
def load_data(n=10000):
    np.random.seed(42)
    groups = ["Group A", "Group B", "Group C", "Group D", "Group E"]
    regions = ["North", "South", "East", "West", "Central"]

    data = {
        "date": [datetime.today() - timedelta(days=np.random.randint(0, 730)) for _ in range(n)],
        "latitude": np.random.uniform(23.0, 37.0, n),
        "longitude": np.random.uniform(68.0, 89.0, n),
        "group": np.random.choice(groups, n),
        "region": np.random.choice(regions, n),
    }
    return pd.DataFrame(data)

df = load_data()

# ================== SIDEBAR FILTERS ==================
st.sidebar.header("🔎 Filters")
group_filter = st.sidebar.multiselect("Select Groups", df["group"].unique(), default=df["group"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])
animation_day = st.sidebar.slider("📅 Timeline Replay", 0, 730, 100)

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

# Timeline replay filter
timeline_df = filtered_df[filtered_df["date"] > (datetime.today() - timedelta(days=animation_day))]

# ================== KPIs ==================
st.subheader("📊 Key Intelligence Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Incidents", f"{len(filtered_df):,}")
col2.metric("Active Groups", filtered_df['group'].nunique())
col3.metric("Regions Involved", filtered_df['region'].nunique())

# ================== MAP ==================
st.subheader("🗺 Geo-Intelligence Map")

if filtered_df.empty:
    st.warning("⚠ No data matches the filters.")
else:
    # Heatmap
    heatmap = pdk.Layer(
        "HeatmapLayer",
        data=timeline_df,
        get_position=["longitude", "latitude"],
        get_weight=1,
        opacity=0.6
    )
    # Scatter
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=timeline_df,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 180],
        get_radius=40000,
        pickable=True,
    )
    # Clustered view
    cluster = pdk.Layer(
        "GridLayer",
        data=timeline_df,
        get_position=["longitude", "latitude"],
        extruded=True,
        cell_size=50000,
        elevation_scale=10,
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state=pdk.ViewState(
            latitude=filtered_df["latitude"].mean(),
            longitude=filtered_df["longitude"].mean(),
            zoom=4,
            pitch=40,
        ),
        layers=[heatmap, cluster, scatter],
        tooltip={"text": "📍 Group: {group}\nRegion: {region}\nDate: {date}"},
    ))

# ================== ANALYTICS TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "📊 Region Split", "🏴 Top Groups", "🧠 AI Insights", "📅 Replay"
])

with tab1:
    st.markdown("### Daily Activity Trend")
    activity = filtered_df.groupby("date").size().reset_index(name="incidents")
    st.area_chart(activity.set_index("date"))

with tab2:
    st.markdown("### Region-wise Incident Share")
    region_counts = filtered_df["region"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with tab3:
    st.markdown("### Top Active Groups")
    top_groups = filtered_df["group"].value_counts().head(5).reset_index()
    top_groups.columns = ["Group", "Incidents"]
    st.bar_chart(top_groups.set_index("Group"))

with tab4:
    st.markdown("### Intelligence Summary (AI-Style)")
    if not filtered_df.empty:
        peak_month = filtered_df['date'].dt.month.value_counts().idxmax()
        dom_group = filtered_df['group'].value_counts().idxmax()
        dom_region = filtered_df['region'].value_counts().idxmax()

        st.success(f"🔥 *Peak Activity Month:* {peak_month}")
        st.info(f"🏴 *Dominant Group:* {dom_group}")
        st.warning(f"🌍 *Most Affected Region:* {dom_region}")
        st.write("🔎 Suggests possible *seasonal spikes* and regional dominance patterns.")

with tab5:
    st.markdown("### Timeline Replay")
    st.write(f"Showing events from the last *{animation_day} days*.")
    st.map(timeline_df[['latitude', 'longitude']])

st.info("✅ Tip: Tune DBSCAN eps/min_samples and KMeans cluster counts to change detection sensitivity.")

