# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN, KMeans
from io import StringIO
import base64

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="🛰️ GPS Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🛰️ GPS Intelligence & Hotspot Prediction Dashboard")
st.markdown(
    """
    **Full-featured intelligence dashboard** for visualizing GPS events, detecting hotspots,
    replaying timelines, and exporting analytic outputs.
    """
)

# ------------------------------
# Synthetic dataset generator (large)
# ------------------------------
@st.cache_data
def generate_data(n=15000, seed=42):
    np.random.seed(seed)
    # You can change these groups/regions to reflect your scenario
    groups = ["Group A", "Group B", "Group C", "Group D", "Group E"]
    regions = ["North", "South", "East", "West", "Central"]

    # Two-year window
    days_back = 730
    base_date = datetime.today()
    dates = [base_date - timedelta(days=int(x)) for x in np.random.randint(0, days_back, size=n)]

    # India-ish lat/lon for realism; adjust to your theater
    latitudes = np.random.uniform(23.0, 37.0, n)
    longitudes = np.random.uniform(68.0, 89.0, n)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "latitude": latitudes,
        "longitude": longitudes,
        "group": np.random.choice(groups, n),
        "region": np.random.choice(regions, n),
        # an optional short note/label field
        "note": np.random.choice(["checkpoint", "movement", "meeting", "incident", "unknown"], n)
    })
    return df

# Try to load real CSV first if present (optional)
use_sample = st.sidebar.checkbox("Use synthetic dataset (recommended)", value=True)
if use_sample:
    df = generate_data(n=15000)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (date,latitude,longitude,group,region,...)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["date"], on_bad_lines="skip")
        # Basic cleanup
        df = df.dropna(subset=["latitude", "longitude", "date"])
    else:
        st.sidebar.info("No file uploaded — falling back to synthetic dataset.")
        df = generate_data(n=15000)

# ------------------------------
# Sidebar filters & timeline
# ------------------------------
st.sidebar.header("🔎 Filters & Playback")
group_choices = sorted(df["group"].unique().tolist())
region_choices = sorted(df["region"].unique().tolist())

selected_groups = st.sidebar.multiselect("Groups", options=group_choices, default=group_choices)
selected_regions = st.sidebar.multiselect("Regions", options=region_choices, default=region_choices)

# date range filter
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()], help="Select start and end dates")

# timeline replay slider (days from today)
max_days = (datetime.today() - min_date).days
replay_days = st.sidebar.slider("Timeline replay window (last N days)", 1, max(1, max_days), 90)
animate = st.sidebar.checkbox("Animate replay (progressive update)", value=False)

# clustering parameters
st.sidebar.markdown("---")
st.sidebar.header("🧭 Hotspot Detection")
db_eps = st.sidebar.slider("DBSCAN eps (approx. degrees)", 0.01, 1.0, 0.08, 0.01)
db_min_samples = st.sidebar.slider("DBSCAN min samples", 3, 30, 6, 1)
k_clusters = st.sidebar.slider("KMeans clusters (for predicted centers)", 2, 12, 4)

# ------------------------------
# Filter & prep data
# ------------------------------
# Ensure date_range are Timestamps
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date = min_date
    end_date = max_date

filtered = df[
    (df["group"].isin(selected_groups)) &
    (df["region"].isin(selected_regions)) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
].copy()

# timeline subset for replay
replay_cutoff = datetime.today() - timedelta(days=replay_days)
timeline_df = filtered[filtered["date"] >= replay_cutoff].copy()

# show warnings if empty
if filtered.empty:
    st.warning("No data after applying filters. Adjust filters or date range.")
    st.stop()

# ------------------------------
# KPIs
# ------------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents (filtered)", f"{len(filtered):,}")
col2.metric("Unique Groups", filtered["group"].nunique())
col3.metric("Regions", filtered["region"].nunique())
col4.metric("Time Span (days)", (filtered["date"].max() - filtered["date"].min()).days)

# ------------------------------
# Hotspot detection (DBSCAN)
# ------------------------------
coords = filtered[["latitude", "longitude"]].to_numpy()

# DBSCAN uses eps in coordinate units; for small areas this is fine.
db = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric="haversine" if False else "euclidean")
# If haversine needed, convert degrees to radians first; here we use euclidean approx
db_labels = db.fit_predict(coords)
filtered["db_cluster"] = db_labels

# compute cluster summaries for DBSCAN (skip label -1 noise)
cluster_centers = []
good_labels = [lab for lab in set(db_labels) if lab != -1]
for lab in good_labels:
    sub = filtered[filtered["db_cluster"] == lab]
    cluster_centers.append({
        "cluster": int(lab),
        "count": int(len(sub)),
        "latitude": float(sub["latitude"].mean()),
        "longitude": float(sub["longitude"].mean()),
        "top_group": sub["group"].value_counts().idxmax()
    })
cluster_centers_df = pd.DataFrame(cluster_centers).sort_values("count", ascending=False)

# KMeans "predicted" centers (coarse)
kmeans_centers = []
if len(filtered) >= k_clusters:
    km = KMeans(n_clusters=k_clusters, random_state=42)
    km.fit(coords)
    kmeans_centers = [{"cluster": int(i), "latitude": float(c[0]), "longitude": float(c[1])} for i, c in enumerate(km.cluster_centers_)]

# ------------------------------
# Map visualization (pydeck)
# ------------------------------
st.subheader("🗺️ Geo-Intelligence Map")

def make_deck(data_for_map, center_lat, center_lon, zoom=5):
    # Heat layer
    heat = pdk.Layer(
        "HeatmapLayer",
        data=data_for_map,
        get_position=["longitude", "latitude"],
        aggregation=pdk.types.String("SUM"),
        get_weight=1,
        radius_pixels=60
    )
    # scatter
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=data_for_map,
        get_position=["longitude", "latitude"],
        get_color=[255, 90, 90],
        get_radius=30000,
        pickable=True,
    )
    # DBSCAN centers layer
    centers_layer = pdk.Layer(
        "TextLayer",
        data=cluster_centers_df if not cluster_centers_df.empty else pd.DataFrame(),
        get_position=["longitude", "latitude"],
        get_text="cluster",
        get_size=24,
        get_angle=0,
        get_color=[255, 255, 255],
        get_text_anchor=pdk.types.String("middle")
    )
    # KMeans markers
    k_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(kmeans_centers) if kmeans_centers else pd.DataFrame(),
        get_position=["longitude", "latitude"],
        get_color=[50, 150, 250],
        get_radius=60000,
    )
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=40)
    return pdk.Deck(map_style="mapbox://styles/mapbox/dark-v9", initial_view_state=view_state, layers=[heat, scatter, centers_layer, k_layer], tooltip={"text": "Group: {group}\nRegion: {region}\nDate: {date}"})

# Determine center
center_lat = float(filtered["latitude"].mean())
center_lon = float(filtered["longitude"].mean())

# Animation handling: if animate, show timeline_df progressively binned by day slices (client-side approximate)
if animate:
    st.info("Animation mode: showing events progressively by day windows (client-side approximate).")
    steps = st.slider("Animation steps (frames)", 5, 50, 12)
    # compute daily bins and show small multiples (or step through)
    # For simplicity we show a slider to pick last X days shown in map
    frame = st.slider("Frame (most recent frames show more events)", 1, steps, steps)
    days_shown = int(replay_days * frame / steps)
    anim_cutoff = datetime.today() - timedelta(days=days_shown)
    anim_df = filtered[filtered["date"] >= anim_cutoff]
    deck = make_deck(anim_df, center_lat, center_lon, zoom=5)
    st.pydeck_chart(deck)
else:
    deck = make_deck(timeline_df, center_lat, center_lon, zoom=5)
    st.pydeck_chart(deck)

# ------------------------------
# Analytics & charts
# ------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("### 📈 Incidents Over Time")
    timeseries = filtered.groupby(filtered["date"].dt.date).size().reset_index(name="incidents")
    timeseries["date"] = pd.to_datetime(timeseries["date"])
    fig = px.area(timeseries, x="date", y="incidents", title="Incidents Over Time", labels={"date":"Date","incidents":"Incidents"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Group Breakdown")
    group_counts = filtered["group"].value_counts().reset_index()
    group_counts.columns = ["group", "count"]
    fig2 = px.bar(group_counts, x="group", y="count", title="Incidents by Group", labels={"count":"Incidents","group":"Group"})
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.markdown("### 🏴 Top 6 Groups")
    top6 = group_counts.head(6)
    st.table(top6.set_index("group"))

    st.markdown("### 🗺️ DBSCAN Hotspots (clusters)")
    if not cluster_centers_df.empty:
        st.dataframe(cluster_centers_df.head(10))
    else:
        st.info("No dense DBSCAN clusters detected with current parameters.")

    st.markdown("### 🔮 KMeans Predicted Centers")
    if kmeans_centers:
        st.dataframe(pd.DataFrame(kmeans_centers))
    else:
        st.info("KMeans centers unavailable (too few points).")

# ------------------------------
# AI-style summary: auto insights
# ------------------------------
st.subheader("🧠 Auto Insights")
ins_cols = st.columns(3)
most_active_group = filtered["group"].value_counts().idxmax()
most_active_region = filtered["region"].value_counts().idxmax()
peak_month = filtered["date"].dt.month.value_counts().idxmax()

ins_cols[0].metric("Dominant Group", most_active_group)
ins_cols[1].metric("Dominant Region", most_active_region)
ins_cols[2].metric("Peak Month (1-12)", int(peak_month))

st.write("**Quick take:** Patterns above indicate dominant actors and regions; DBSCAN clusters highlight dense local hotspots; KMeans centers provide coarse predicted hotspots.")

# ------------------------------
# Export & download
# ------------------------------
st.subheader("💾 Export / Download")
csv = filtered.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown(f"[⬇️ Download filtered data as CSV](data:text/csv;base64,{b64})", unsafe_allow_html=True)

if not cluster_centers_df.empty:
    csv2 = cluster_centers_df.to_csv(index=False)
    b64c = base64.b64encode(csv2.encode()).decode()
    st.markdown(f"[⬇️ Download DBSCAN cluster centers CSV](data:text/csv;base64,{b64c})", unsafe_allow_html=True)
if kmeans_centers:
    csv3 = pd.DataFrame(kmeans_centers).to_csv(index=False)
    b64k = base64.b64encode(csv3.encode()).decode()
    st.markdown(f"[⬇️ Download KMeans centers CSV](data:text/csv;base64,{b64k})", unsafe_allow_html=True)

st.info("✅ Tip: Tune DBSCAN eps/min_samples and KMeans cluster counts to change detection sensitivity.")

