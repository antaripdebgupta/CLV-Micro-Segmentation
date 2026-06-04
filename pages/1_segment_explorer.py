"""
pages/1_segment_explorer.py
Interactive segment explorer with filters, profiles, and EDA charts.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Segment Explorer", layout="wide")

DB_PATH = "data/processed/customers_clean.db"


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


df = load_data()

st.title("Segment Explorer")
st.markdown("Explore customer micro-segments, CLV bands, and behavioural profiles.")

# Sidebar Filters
with st.sidebar:
    st.header("Filters")

    segments = ["All"] + sorted(df["Segment"].dropna().unique().tolist()) if "Segment" in df.columns else ["All"]
    selected_segment = st.selectbox("Customer Segment", segments)

    clv_bands = ["All", "High", "Medium", "Low"]
    selected_clv = st.selectbox("CLV Band", clv_bands)

    if "Membership Type" in df.columns:
        memberships = ["All"] + sorted(df["Membership Type"].dropna().unique().tolist())
        selected_membership = st.selectbox("Membership Type", memberships)
    else:
        selected_membership = "All"

    if "Gender" in df.columns:
        genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
        selected_gender = st.selectbox("Gender", genders)
    else:
        selected_gender = "All"

# Apply filters
filtered = df.copy()
if selected_segment != "All" and "Segment" in df.columns:
    filtered = filtered[filtered["Segment"] == selected_segment]
if selected_clv != "All" and "CLV_Band" in df.columns:
    filtered = filtered[filtered["CLV_Band"] == selected_clv]
if selected_membership != "All" and "Membership Type" in df.columns:
    filtered = filtered[filtered["Membership Type"] == selected_membership]
if selected_gender != "All" and "Gender" in df.columns:
    filtered = filtered[filtered["Gender"] == selected_gender]

st.markdown(f"**Showing {len(filtered):,} of {len(df):,} customers**")

#  KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers",      f"{len(filtered):,}")
c2.metric("Avg Spend",      f"₹{filtered['Total Spend'].mean():,.0f}" if "Total Spend" in filtered.columns else "—")
c3.metric("Avg CLV Score",  f"{filtered['CLV_Score'].mean():.2f}"     if "CLV_Score" in filtered.columns else "—")
c4.metric("Avg Days Since", f"{filtered['Days Since Last Purchase'].mean():.0f} days" if "Days Since Last Purchase" in filtered.columns else "—")

st.markdown("---")

# Charts Row 1 
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Spend Distribution by Segment")
    if "Segment" in filtered.columns and "Total Spend" in filtered.columns:
        fig = px.box(filtered, x="Segment", y="Total Spend",
                     color="Segment", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("CLV Score vs Total Spend")
    if "CLV_Score" in filtered.columns and "Total Spend" in filtered.columns:
        fig2 = px.scatter(
            filtered.sample(min(500, len(filtered))),
            x="Total Spend", y="CLV_Score",
            color="Segment" if "Segment" in filtered.columns else None,
            size="Items Purchased" if "Items Purchased" in filtered.columns else None,
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.7,
        )
        fig2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# Charts Row 2 
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Purchase Frequency Histogram")
    if "Items Purchased" in filtered.columns:
        fig3 = px.histogram(filtered, x="Items Purchased", nbins=30,
                            color_discrete_sequence=["#534AB7"])
        fig3.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Membership Type Breakdown")
    if "Membership Type" in filtered.columns:
        mem_counts = filtered["Membership Type"].value_counts().reset_index()
        mem_counts.columns = ["Type", "Count"]
        fig4 = px.bar(mem_counts, x="Type", y="Count",
                      color="Type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

# Segment Profile Heatmap 
st.subheader("Segment Profile Heatmap")
if "Segment" in df.columns:
    profile_cols = [c for c in ["Total Spend", "Items Purchased", "Days Since Last Purchase",
                                 "Average Rating", "CLV_Score", "Purchase_Prob", "Engagement_Score"]
                    if c in df.columns]
    profile = df.groupby("Segment")[profile_cols].median()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig5 = px.imshow(
        profile_norm.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels={"color": "Normalised Median"},
        text_auto=".2f",
    )
    fig5.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig5, use_container_width=True)

# Raw data table 
with st.expander("View raw filtered data"):
    st.dataframe(filtered.head(200), use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, "filtered_customers.csv", "text/csv")
