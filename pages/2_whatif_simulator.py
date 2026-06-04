"""
pages/2_whatif_simulator.py
What-If CLV Simulator — adjust customer features with sliders
and watch the predicted CLV band update in real time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="What-If Simulator", layout="wide")

MODEL_PATH   = "models/clv_classifier.pkl"
ENCODER_PATH = "models/clv_label_encoder.pkl"
SCALER_PATH  = "models/scaler.pkl"


@st.cache_resource
def load_artifacts():
    import joblib
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, le, scaler


def check_models_ready():
    return all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, SCALER_PATH])


st.title("What-If CLV Simulator")
st.markdown(
    "Adjust customer attributes using the sliders below. "
    "The predicted **CLV band** and probability scores update instantly."
)

if not check_models_ready():
    st.error("Model files not found. Run `python train_pipeline.py` first.")
    st.stop()

clf, le, scaler = load_artifacts()

# Input Sliders 
st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 35)
    total_spend = st.slider("Total Spend (₹)", 0, 5000, 1200, step=50)
    items = st.slider("Items Purchased", 1, 50, 10)

with col2:
    days_since = st.slider("Days Since Last Purchase", 0, 365, 45)
    avg_rating = st.slider("Average Rating", 1.0, 5.0, 3.8, step=0.1)
    discount   = st.slider("Discount Applied (%)", 0, 50, 10)

with col3:
    satisfaction = st.slider("Satisfaction Level", 1, 5, 3)
    gender_enc   = st.selectbox("Gender", options=["Female (0)", "Male (1)"])
    membership   = st.selectbox("Membership Type", options=["Bronze (0)", "Silver (1)", "Gold (2)"])

gender_val     = int(gender_enc.split("(")[1].replace(")", ""))
membership_val = int(membership.split("(")[1].replace(")", ""))

# ── Feature Engineering (mirrors features.py) ─────────────────────────────────
freq_norm    = (items - 1) / 49
recency_norm = days_since / 365
p_alive      = freq_norm * (1 - recency_norm)
engagement   = 0.4 * freq_norm + 0.3 * (avg_rating / 5) + 0.3 * p_alive
spend_per_item = total_spend / max(items, 1)

# Scale numerics (using saved scaler — Satisfaction Level is categorical, not scaled)
numeric_vals = pd.DataFrame([[age, days_since, total_spend, items, avg_rating, discount]],
                            columns=["Age", "Days Since Last Purchase", "Total Spend",
                                     "Items Purchased", "Average Rating", "Discount Applied"])
scaled = scaler.transform(numeric_vals)[0]

feature_input = {
    "Age_Scaled": scaled[0],
    "Days Since Last Purchase_Scaled": scaled[1],
    "Total Spend_Scaled": scaled[2],
    "Items Purchased_Scaled": scaled[3],
    "Average Rating_Scaled": scaled[4],
    "Discount Applied_Scaled": scaled[5],
    "Satisfaction Level_Enc": satisfaction - 1,  # slider 1-5 → encoded 0-4
    "Purchase_Prob": p_alive,
    "Engagement_Score": engagement,
    "Spend_Per_Item": spend_per_item,
    "Gender_Enc": gender_val,
    "Membership Type_Enc": membership_val,
}

from src.features import get_ml_feature_cols
feature_cols = get_ml_feature_cols()
X = pd.DataFrame([{c: feature_input.get(c, 0) for c in feature_cols}])

proba  = clf.predict_proba(X)[0]
label  = le.inverse_transform(clf.predict(X))[0]
proba_dict = dict(zip(le.classes_, proba))

# Result Display 
st.markdown("---")
st.subheader("Prediction Result")

band_colors = {"Low": "#E24B4A", "Medium": "#EF9F27", "High": "#639922"}
band_color  = band_colors.get(label, "#7F77DD")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown(
        f"""
        <div style="
            background: {band_color}22;
            border: 2px solid {band_color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        ">
            <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Predicted CLV Band</div>
            <div style="font-size: 42px; font-weight: 700; color: {band_color};">{label}</div>
            <div style="font-size: 13px; color: #888; margin-top: 8px;">
                Confidence: {max(proba)*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("")
    # Retention recommendation
    recs = {
        "High":   "VIP treatment — early access, loyalty rewards",
        "Medium": "Targeted email campaigns + personalised offers",
        "Low":    "Re-engagement campaign + discount incentive",
    }
    st.info(f"**Recommended Action:** {recs.get(label, '')}")

with res_col2:
    # Probability gauge chart
    fig = go.Figure()
    for cls in ["Low", "Medium", "High"]:
        p = proba_dict.get(cls, 0)
        fig.add_trace(go.Bar(
            x=[p], y=[cls],
            orientation="h",
            marker_color=band_colors.get(cls, "#999"),
            text=f"{p*100:.1f}%",
            textposition="inside",
            name=cls,
        ))
    fig.update_layout(
        title="CLV Band Probabilities",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        barmode="group",
        showlegend=False,
        height=220,
        margin=dict(t=40, b=20, l=80, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# Derived metrics display 
st.markdown("---")
st.subheader("Computed Feature Values")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Purchase Probability (P-alive)", f"{p_alive:.3f}")
m2.metric("Engagement Score",               f"{engagement:.3f}")
m3.metric("Spend per Item",                 f"₹{spend_per_item:.2f}")
m4.metric("Discount Sensitivity",           f"{discount}%")
