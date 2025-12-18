import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Global Disaster Monitor (GDACS)", page_icon="🌐", layout="wide")

GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
CACHE_FILE = "GDACS_cache.csv"

# Only the namespaces we actually use in XPath paths below
NS = {
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "gdacs": "http://www.gdacs.org",
}


# ---------- helpers ----------
def _parse_rfc822(s: str):
    # Example: "Wed, 17 Dec 2025 15:15:04 GMT"
    if not s:
        return pd.NaT
    try:
        dt = datetime.strptime(s.strip(), "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return pd.NaT


def _find_text(el, path: str, default=None):
    if el is None:
        return default
    node = el.find(path, NS)
    if node is None or node.text is None:
        return default
    return node.text.strip()


@st.cache_data(ttl=600)
def fetch_gdacs_rss_xml() -> str:
    """
    Fetch GDACS RSS.
    Some networks return HTML (blocked/redirect pages) instead of XML.
    Detect that early to avoid ElementTree's vague errors.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }
    r = requests.get(GDACS_RSS_URL, timeout=30, headers=headers, allow_redirects=True)
    r.raise_for_status()

    text = r.content.decode("utf-8", errors="replace").lstrip("\ufeff").strip()
    head = text[:200].lower()

    if head.startswith("<!doctype html") or head.startswith("<html") or "<html" in head:
        raise ValueError(
            "GDACS returned HTML instead of XML (blocked/redirect/proxy page).\n"
            f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}\n"
            f"First chars: {text[:200]}"
        )

    if not text.startswith("<"):
        raise ValueError(
            "Response does not look like XML.\n"
            f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}\n"
            f"First chars: {text[:200]}"
        )

    return text


def rss_to_df(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    items = root.findall("./channel/item")

    # Keep ONLY columns that are used by the website (filters/charts/table/map)
    FIELDS = {
        # text fields (used in UI)
        "title": ("text", "title", ""),
        "link": ("text", "link", ""),
        "event_type": ("text", "gdacs:eventtype", "Unknown"),
        "alert_level": ("text", "gdacs:alertlevel", "Unknown"),
        "country": ("text", "gdacs:country", "Unknown"),
        "severity_text": ("text", "gdacs:severity", ""),
        "population_text": ("text", "gdacs:population", ""),
        # numeric fields (used in charts/map)
        "alert_score": ("num", "gdacs:alertscore", None),
        "latitude": ("num", "geo:Point/geo:lat", None),
        "longitude": ("num", "geo:Point/geo:long", None),
        # datetime fields (used to build main_time/date_utc)
        "pub_date": ("date", "pubDate", None),
        "from_date": ("date", "gdacs:fromdate", None),
    }

    rows = []
    for it in items:
        row = {}
        for col, (kind, path, default) in FIELDS.items():
            if kind == "text":
                row[col] = _find_text(it, path, default)
            elif kind == "num":
                row[col] = pd.to_numeric(_find_text(it, path, default), errors="coerce")
            elif kind == "date":
                row[col] = _parse_rfc822(_find_text(it, path, default))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Main time for daily aggregation (prefer from_date, fallback to pub_date)
    df["main_time"] = pd.to_datetime(df["from_date"].fillna(df["pub_date"]), utc=True, errors="coerce")
    df["date_utc"] = df["main_time"].dt.date
    df = df.sort_values("main_time", ascending=True)

    for col in ["event_type", "alert_level", "country"]:
        df[col] = df[col].fillna("Unknown")

    return df


def load_data_with_cache():
    try:
        xml_text = fetch_gdacs_rss_xml()
        df = rss_to_df(xml_text)
        df.to_csv(CACHE_FILE, index=False, encoding="utf-8")
        return df, None

    except Exception as e:
        if os.path.exists(CACHE_FILE):
            cached = pd.read_csv(CACHE_FILE, encoding="utf-8")

            # Restore only datetime columns we actually rely on
            dt_cols = ["pub_date", "from_date", "main_time"]
            for c in dt_cols:
                if c in cached.columns:
                    cached[c] = pd.to_datetime(cached[c], utc=True, errors="coerce")

            if "date_utc" in cached.columns:
                cached["date_utc"] = pd.to_datetime(cached["date_utc"], errors="coerce").dt.date

            for col in ["event_type", "alert_level", "country"]:
                if col in cached.columns:
                    cached[col] = cached[col].fillna("Unknown")

            return cached, "Error: GDACS fetch failed — using cached file."

        return pd.DataFrame(), "Error: GDACS fetch failed and no cache found."


# ---------- UI ----------
st.title("Global Disaster Monitor — Live Alerts Dashboard")
st.caption("Source: GDACS RSS feed (near real-time natural disaster alerts)")

df, warn = load_data_with_cache()
if warn:
    st.warning(warn)

if df is None or df.empty:
    st.error("No data available.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters (UTC)")

all_types = sorted([x for x in df["event_type"].dropna().unique().tolist() if x])
all_levels = sorted([x for x in df["alert_level"].dropna().unique().tolist() if x])
all_countries = sorted([x for x in df["country"].dropna().unique().tolist() if x])

selected_types = st.sidebar.multiselect("Event type", all_types)
selected_levels = st.sidebar.multiselect("Alert level", all_levels)
selected_countries = st.sidebar.multiselect("Country", all_countries)

today_utc = datetime.now(timezone.utc).date()
default_start = today_utc - timedelta(days=14)
date_range = st.sidebar.date_input("Date range", value=(default_start, today_utc))

max_points = st.sidebar.slider("Map points (highest alert score)", 20, 300, 120, 20)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = default_start, today_utc

# Selection-aware message
no_filter_selected = (not selected_types) or (not selected_levels) or (not selected_countries)
if no_filter_selected:
    st.info("Please select at least one option from each filter to continue.")
    st.stop()

# Filter
mask = (
    df["event_type"].isin(selected_types)
    & df["alert_level"].isin(selected_levels)
    & df["country"].isin(selected_countries)
    & df["date_utc"].between(start_d, end_d)
)
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No records match your filters.")
    st.stop()

# Sidebar KPIs
st.sidebar.subheader("Summary")
st.sidebar.write("Alerts:", int(len(filtered)))

avg_score = pd.to_numeric(filtered["alert_score"], errors="coerce").mean()
max_score = pd.to_numeric(filtered["alert_score"], errors="coerce").max()
st.sidebar.write("Avg alert score:", float(avg_score) if pd.notna(avg_score) else "N/A")
st.sidebar.write("Max alert score:", float(max_score) if pd.notna(max_score) else "N/A")

# Daily aggregates
daily_count = filtered.groupby("date_utc").size()
daily_avg_score = filtered.groupby("date_utc")["alert_score"].mean()
cumulative_alerts = daily_count.cumsum()

# Daily Summary
st.sidebar.subheader("Daily summary")

# Toggle: default is last 10 days
show_all_days = st.sidebar.checkbox("Show all days", value=True)

summary_tbl = pd.DataFrame(
    {
        "date_utc": daily_count.index,
        "alerts": daily_count.values,
        "avg_alert_score": daily_avg_score.reindex(daily_count.index).values,
    }
).reset_index(drop=True)

# Default behavior: last 10 days
if not show_all_days:
    summary_tbl = summary_tbl.tail(10)

st.sidebar.dataframe(
    summary_tbl,
    use_container_width=True,
    height=300,  # makes it scrollable when showing all days
)

# Main charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily alert count")
    st.line_chart(daily_count)

with col2:
    st.subheader("Daily average alert score")
    st.line_chart(daily_avg_score)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Cumulative alerts (selected range)")
    st.line_chart(cumulative_alerts)

with col4:
    st.subheader("Alert level distribution")
    level_counts = filtered["alert_level"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(level_counts.index.astype(str), level_counts.values)
    ax.set_ylabel("Alerts")
    ax.set_xticklabels(level_counts.index.astype(str), rotation=30, ha="right")
    st.pyplot(fig)

st.subheader("Event type proportions")
type_counts = filtered["event_type"].value_counts()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(type_counts.values, labels=type_counts.index.astype(str), autopct="%1.1f%%", startangle=90)
st.pyplot(fig)

# Top countries + slider
st.subheader("Top countries (by alert count)")

country_counts = filtered["country"].value_counts().sort_values(ascending=False)

default_n = min(10, len(country_counts))
max_n = min(60, len(country_counts))
top_n = st.session_state.get("top_n_countries", default_n)

st.bar_chart(country_counts.head(top_n))

st.markdown(
    """
    <style>
    /* Slider track */
    div[data-baseweb="slider"] > div > div > div {
        background-color: #1f77b4 !important;
    }
    /* Slider knob */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

top_n = st.slider(
    "Number of countries shown",
    min_value=5,
    max_value=max_n if max_n >= 5 else 5,
    value=top_n,
    key="top_n_countries",
)

st.subheader("Map (highest alert score)")
map_df = (
    filtered.dropna(subset=["latitude", "longitude", "alert_score"])
    .sort_values("alert_score", ascending=False)
    .head(max_points)
)
st.map(map_df[["latitude", "longitude"]])

st.subheader("Most recent alerts")
show_cols = [
    "main_time",
    "event_type",
    "alert_level",
    "alert_score",
    "country",
    "severity_text",
    "population_text",
    "link",
]
st.dataframe(
    filtered.sort_values("main_time", ascending=False)[show_cols].head(50),
    use_container_width=True,
)

with st.expander("Debug: show raw feed preview (first 400 chars)"):
    try:
        st.code(fetch_gdacs_rss_xml()[:400])
    except Exception as e:
        st.error(str(e))



