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

    except Exception:
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


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)


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
show_all_days = st.sidebar.checkbox("Show all days", value=True)

summary_tbl = pd.DataFrame(
    {
        "date_utc": daily_count.index,
        "alerts": daily_count.values,
        "avg_alert_score": daily_avg_score.reindex(daily_count.index).values,
    }
).reset_index(drop=True)

if not show_all_days:
    summary_tbl = summary_tbl.tail(10)

# Make row numbering start at 1 instead of 0
summary_tbl.index = range(1, len(summary_tbl) + 1)

st.sidebar.dataframe(
    summary_tbl,
    use_container_width=True,
    height=387,
)

# ---------- Main charts (Streamlit built-ins where possible) ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily alert count")
    st.line_chart(daily_count)

with col2:
    st.subheader("Daily average alert score")
    st.line_chart(daily_avg_score)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Cumulative alerts")
    st.line_chart(cumulative_alerts)

with col4:
    st.subheader("Alert level distribution")
    level_counts = filtered["alert_level"].value_counts().sort_values(ascending=False)
    st.bar_chart(level_counts, height=369)

st.subheader("Top countries (by alert count)")
country_counts = filtered["country"].value_counts().sort_values(ascending=False)
st.bar_chart(country_counts, height=500)

# ---------- Newly added charts go HERE (under Top countries, before Map) ----------
st.divider()
st.header("Additional insights")

# Row 1: Pie (event types) + Histogram (alert scores)
cA, cB = st.columns(2)

with cA:
    st.subheader("Event type distribution (pie)")
    event_counts = filtered["event_type"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        event_counts.values,
        labels=event_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Event type distribution")
    ax.axis("equal")
    show_fig(fig)

with cB:
    st.subheader("Alert score distribution (histogram)")
    scores = pd.to_numeric(filtered["alert_score"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20)
    ax.set_title("Distribution of alert scores")
    ax.set_xlabel("Alert score")
    ax.set_ylabel("Frequency")
    show_fig(fig)

# Row 2: Boxplot (score by event type) + Stacked bar (levels by type)
cC, cD = st.columns(2)

with cC:
    st.subheader("Alert score by event type (boxplot)")
    box_df = filtered[["event_type", "alert_score"]].copy()
    box_df["alert_score"] = pd.to_numeric(box_df["alert_score"], errors="coerce")
    box_df = box_df.dropna(subset=["event_type", "alert_score"])

    if not box_df.empty:
        order = (
            box_df.groupby("event_type")["alert_score"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        data = [box_df.loc[box_df["event_type"] == t, "alert_score"].values for t in order]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title("Alert score by event type")
        ax.set_xlabel("Event type")
        ax.set_ylabel("Alert score")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        show_fig(fig)
    else:
        st.info("Not enough numeric alert_score values to draw the boxplot.")

with cD:
    st.subheader("Alert levels by event type (stacked bar)")
    stacked = (
        filtered.groupby(["event_type", "alert_level"])
        .size()
        .unstack(fill_value=0)
    )

    if not stacked.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        stacked.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Alert levels by event type")
        ax.set_xlabel("Event type")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(title="Alert level", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        show_fig(fig)
    else:
        st.info("No data available for stacked bar chart.")

# Row 3: Trend by type + Rolling average
cE, cF = st.columns(2)

with cE:
    st.subheader("Alerts over time by event type (trend)")
    pivot = (
        filtered.groupby(["date_utc", "event_type"])
        .size()
        .unstack(fill_value=0)
    )

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(ax=ax)
        ax.set_title("Alerts over time by event type")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Alerts")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(title="Event type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        show_fig(fig)
    else:
        st.info("Not enough data to plot event-type trends over time.")

with cF:
    st.subheader("Daily alerts (7-day rolling average)")
    rolling = daily_count.rolling(7).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_count.index, daily_count.values, label="Daily", alpha=0.4)
    ax.plot(rolling.index, rolling.values, label="7-day rolling avg")
    ax.set_title("Daily alerts (smoothed)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Alerts")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    show_fig(fig)

# ---------- Map ----------
st.subheader("Map (highest alert score)")
map_df = (
    filtered.dropna(subset=["latitude", "longitude", "alert_score"])
    .sort_values("alert_score", ascending=False)
    .head(max_points)
)
st.map(map_df[["latitude", "longitude"]])
