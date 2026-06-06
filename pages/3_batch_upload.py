"""
Batch CSV upload page — upload any customer CSV,
get CLV band predictions + downloadable results.
Also shows full model evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Batch Prediction", layout="wide")

MODEL_PATH   = "models/clv_classifier.pkl"
ENCODER_PATH = "models/clv_label_encoder.pkl"
SCALER_PATH  = "models/scaler.pkl"


@st.cache_resource
def load_artifacts():
    import joblib
    clf    = joblib.load(MODEL_PATH)
    le     = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, le, scaler


def check_models_ready():
    return all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, SCALER_PATH])


from src import dataset_store

active_record = dataset_store.get_active_dataset_record()
active_name = active_record["name"] if active_record else "Default Dataset"
db_path = active_record["db_path"] if active_record else "data/processed/customers_clean.db"

# Sidebar
dataset_store.render_sidebar()

st.title("Batch Prediction & Model Evaluation")
st.info(f"Showing predictions and evaluation based on: **{active_name}**")

tab1, tab2 = st.tabs(["Batch CSV Upload", "Model Evaluation Report"])

# TAB 1 — Batch Upload
with tab1:
    st.markdown(
        "Upload a CSV with customer data and download CLV band predictions for every row."
    )

    if not check_models_ready():
        st.error("Models not found. Run `python train_pipeline.py` first.")
        st.stop()

    clf, le, scaler = load_artifacts()

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        raw.columns = [c.strip().title().replace("  ", " ") for c in raw.columns]
        st.markdown(f"**Uploaded:** {len(raw):,} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(5), use_container_width=True)

        # Attempt to engineer features and predict
        try:
            import sys; sys.path.insert(0, ".")
            from src.features import engineer_features, get_ml_feature_cols
            from src.preprocess import NUMERIC_COLS, CATEGORICAL_COLS
            from sklearn.preprocessing import LabelEncoder as LE

            df = raw.copy()

            # Fill missing numerics
            for col in NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if not df[col].isnull().all() else 0)

            # Encode categoricals
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    enc = LE()
                    df[col + "_Enc"] = enc.fit_transform(df[col].astype(str))

            # Scale numerics
            avail_num = [c for c in NUMERIC_COLS if c in df.columns]
            scaled = scaler.transform(df[avail_num].fillna(0))
            for i, col in enumerate(avail_num):
                df[col + "_Scaled"] = scaled[:, i]

            # Feature engineering
            df = engineer_features(df)

            # Predict
            feature_cols = get_ml_feature_cols()
            X = df[[c for c in feature_cols if c in df.columns]].fillna(0)
            preds  = clf.predict(X)
            probas = clf.predict_proba(X)

            df["Predicted_CLV_Band"] = le.inverse_transform(preds)
            for i, cls in enumerate(le.classes_):
                df[f"Prob_{cls}"] = probas[:, i].round(3)

            st.success(f"Predictions complete for {len(df):,} customers.")

            # Summary pie
            band_counts = df["Predicted_CLV_Band"].value_counts().reset_index()
            band_counts.columns = ["CLV Band", "Count"]
            fig = px.pie(
                band_counts, values="Count", names="CLV Band",
                color="CLV Band",
                color_discrete_map={"Low": "#F09595", "Medium": "#FAC775", "High": "#97C459"},
                hole=0.4, title="Predicted CLV Band Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download
            out_cols = list(raw.columns) + ["Predicted_CLV_Band"] + [f"Prob_{c}" for c in le.classes_]
            out_df   = df[[c for c in out_cols if c in df.columns]]
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                csv, "clv_predictions.csv", "text/csv",
                use_container_width=True
            )
            st.dataframe(out_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Make sure the uploaded CSV has columns similar to the training data: "
                "Age, Total Spend, Items Purchased, Days Since Last Purchase, etc."
            )

    else:
        st.info("Upload a CSV file to begin batch prediction.")
        st.markdown("**Expected columns:** Age, Gender, Total Spend, Items Purchased, "
                    "Days Since Last Purchase, Average Rating, Membership Type, "
                    "Discount Applied, Satisfaction Level")


# TAB 2 — Evaluation Report
with tab2:
    st.subheader("Model Evaluation Report")

    if not check_models_ready():
        st.warning("Run train_pipeline.py to generate evaluation metrics.")
        st.stop()

    clf, le, scaler = load_artifacts()

    try:
        import sys; sys.path.insert(0, ".")
        import sqlite3
        from src.preprocess import load_from_db
        from src.features import engineer_features, get_ml_feature_cols
        from src.cluster import run_clustering, CLUSTER_FEATURES
        from src.evaluate import (
            plot_confusion_matrix, plot_feature_importance,
            plot_shap_summary, cluster_metrics_table, plot_elbow
        )
        from sklearn.model_selection import train_test_split

        if not os.path.exists(db_path):
            st.warning("Database not found. Run train_pipeline.py first.")
            st.stop()

        @st.cache_data(ttl=0)
        def get_eval_data(path):
            df = load_from_db(db_path=path)
            df = engineer_features(df)
            df = run_clustering(df)
            return df

        df = get_eval_data(db_path)
        feature_cols = [c for c in get_ml_feature_cols() if c in df.columns]
        X = df[feature_cols].fillna(0)
        y = le.transform(df["CLV_Band"].astype(str))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred = clf.predict(X_test)

        # ── Clustering metrics ──
        st.markdown("### Clustering Evaluation")
        X_cluster = df[[c for c in CLUSTER_FEATURES if c in df.columns]].fillna(0).values
        km_labels = df["KMeans_Cluster"].values if "KMeans_Cluster" in df.columns else np.zeros(len(df))

        metrics = cluster_metrics_table(X_cluster, km_labels)
        if metrics:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Silhouette Score",    str(metrics.get("Silhouette Score", "—")))
            mc2.metric("Davies-Bouldin Index",str(metrics.get("Davies-Bouldin Index", "—")))
            mc3.metric("Clusters Found",      str(metrics.get("Number of Clusters", "—")))

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**Elbow + Silhouette curves**")
            elbow_fig = plot_elbow(X_cluster)
            st.pyplot(elbow_fig, use_container_width=True)

        # ── Classification metrics ──
        st.markdown("### CLV Classification Evaluation")

        from sklearn.metrics import classification_report, roc_auc_score
        report_str = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df  = pd.DataFrame(report_str).T.round(3)
        st.dataframe(report_df, use_container_width=True)

        try:
            y_proba = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            st.metric("ROC-AUC (weighted OvR)", f"{auc:.4f}")
        except Exception:
            pass

        cc1, cc2 = st.columns(2)
        with cc1:
            cm_fig = plot_confusion_matrix(y_test, y_pred, list(le.classes_))
            st.pyplot(cm_fig, use_container_width=True)
        with cc2:
            fi_fig = plot_feature_importance(clf, feature_cols)
            st.pyplot(fi_fig, use_container_width=True)

        # SHAP (optional — may be slow on large datasets)
        if st.checkbox("Generate SHAP summary (may take ~30s)"):
            shap_fig = plot_shap_summary(clf, X_test.sample(min(200, len(X_test))))
            if shap_fig:
                st.pyplot(shap_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate evaluation report: {e}")
