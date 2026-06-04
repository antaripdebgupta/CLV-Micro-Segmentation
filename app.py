"""
app.py
Main Streamlit entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(
    page_title="CLV Micro-Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/processed/customers_clean.db"


@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        st.error("Database not found. Please run `python train_pipeline.py` first.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


df = load_data()

# Sidebar
with st.sidebar:
    st.title("CLV Segmentation")
    st.caption("E-Commerce Customer Behaviour Analysis")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("pages/1_segment_explorer.py",  label="Segment Explorer",  icon="🗂️")
    st.page_link("pages/2_whatif_simulator.py",  label="What-If Simulator", icon="🔮")
    st.page_link("pages/3_batch_upload.py",      label="Batch Prediction",  icon="📂")
    st.markdown("---")
    st.caption("Final Year DS Project · 2026")

# Home page
st.title("E-Commerce CLV Micro-Segmentation")
st.markdown(
    "A complete end-to-end Data Science pipeline: "
    "**customer segmentation** (K-Means + DBSCAN) combined with "
    "**CLV band prediction** (Gradient Boosting) to drive data-driven retention decisions."
)

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",  f"{len(df):,}")
col2.metric("Avg Total Spend",  f"₹{df['Total Spend'].mean():,.0f}" if "Total Spend" in df.columns else "—")
col3.metric("Avg Items Bought", f"{df['Items Purchased'].mean():.1f}" if "Items Purchased" in df.columns else "—")
col4.metric("Avg Rating",       f"{df['Average Rating'].mean():.2f}" if "Average Rating" in df.columns else "—")

st.markdown("---")

# Segment distribution
if "Segment" in df.columns:
    st.subheader("Customer Segment Distribution")
    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    import plotly.express as px
    fig = px.pie(
        seg_counts, values="Count", names="Segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

if "CLV_Band" in df.columns:
    st.subheader("CLV Band Distribution")
    clv_counts = df["CLV_Band"].value_counts().reset_index()
    clv_counts.columns = ["CLV Band", "Count"]
    fig2 = px.bar(
        clv_counts, x="CLV Band", y="Count",
        color="CLV Band",
        color_discrete_map={"Low": "#F09595", "Medium": "#FAC775", "High": "#97C459"},
    )
    fig2.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Data source: Kaggle — E-Commerce Customer Behavior Dataset")
