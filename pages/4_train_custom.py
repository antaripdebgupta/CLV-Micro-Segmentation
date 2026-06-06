"""
Upload any e-commerce CSV → column mapping → hyperparameter panel →
live training → PDF download.

State is stored in st.session_state so the rest of the app can use it.
"""

import os
import sys
import io
import traceback

import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, ".")

st.set_page_config(page_title="Train on Your Data", layout="wide")

from src import dataset_store

# Session state initialisation 
_defaults = {
    "custom_df":        None,   # trained + enriched DataFrame
    "custom_metrics":   None,   # dict from src.model.train()
    "custom_report_path": None, # path to generated PDF
    "train_history":    [],     # list of summary dicts
    "column_map":       {},     # user-chosen column mapping
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state["train_history"]:
    datasets = dataset_store.list_datasets()
    for idx, d in enumerate(reversed(datasets)):
        if not d.get("is_default"):
            st.session_state["train_history"].append({
                "dataset_id": d["dataset_id"],
                "Run":        len(st.session_state["train_history"]) + 1,
                "File":       d["name"],
                "Rows":       d["n_rows"],
                "Segments":   d["n_segments"],
                "ROC-AUC":    f"{d['auc']:.4f}" if d.get("auc") else "n/a",
                "Silhouette": f"{d.get('silhouette', 0):.4f}",
                "Time":       d["trained_at"][11:19], # HH:MM:SS
            })


# Helpers

def _safe_import():
    """Import pipeline modules and surface a clear error if src/ is missing."""
    try:
        from src.preprocess import NUMERIC_COLS, CATEGORICAL_COLS
        from src.features   import engineer_features, get_ml_feature_cols
        from src.cluster    import run_clustering
        from src.model      import train as train_model
        from src.data_quality import check_quality, render_quality_report
        from src.report     import generate_report
        from src.logger     import get_logger
        return (NUMERIC_COLS, CATEGORICAL_COLS, engineer_features,
                get_ml_feature_cols, run_clustering, train_model,
                check_quality, render_quality_report, generate_report, get_logger)
    except ImportError as e:
        st.error(
            f"Cannot import pipeline modules: {e}\n\n"
            "Make sure you are running from the project root: `streamlit run app.py`"
        )
        st.stop()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and title-case column names to match pipeline expectations."""
    df = df.copy()
    df.columns = [c.strip().title().replace("  ", " ") for c in df.columns]
    return df


def _apply_column_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename user CSV columns to the names the pipeline expects."""
    df = df.copy()
    rename = {v: k for k, v in col_map.items() if v and v != "(not available)"}
    df = df.rename(columns=rename)
    return df


def _run_pipeline(df_raw: pd.DataFrame, hp: dict, col_map: dict, dataset_name: str):
    """
    Execute the full training pipeline on a raw DataFrame.
    Returns (enriched_df, metrics, pdf_path) or raises on failure.
    """
    (NUMERIC_COLS, CATEGORICAL_COLS, engineer_features,
     get_ml_feature_cols, run_clustering, train_model,
     check_quality, render_quality_report, generate_report, get_logger) = _safe_import()

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib

    log = get_logger("custom_train")
    log.info("Custom training started — dataset: %s, rows: %d", dataset_name, len(df_raw))

    # 1. Normalise + map columns
    df = _normalise_columns(df_raw)
    if col_map:
        df = _apply_column_map(df, col_map)

    # 2. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info("Dropped %d duplicates", before - len(df))

    # 3. Fill missing numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
            # clip negatives for non-negative cols
            if col in ["Total Spend", "Items Purchased", "Days Since Last Purchase", "Age"]:
                df[col] = df[col].clip(lower=0)

    # 4. Fill / add missing optional columns with sensible defaults
    defaults = {
        "Gender":             "Unknown",
        "City":               "Unknown",
        "Membership Type":    "Bronze",
        "Satisfaction Level": "Neutral",
        "Discount Applied":   0,
        "Average Rating":     3.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
            log.warning("Column '%s' not found — filled with default: %s", col, default)
        else:
            df[col] = df[col].fillna(default)

    # 5. Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_Enc"] = le.fit_transform(df[col].astype(str))

    # 6. Scale numerics
    avail_num = [c for c in NUMERIC_COLS if c in df.columns]
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(df[avail_num].values)
    for i, col in enumerate(avail_num):
        df[col + "_Scaled"] = scaled_arr[:, i]

    # Save scaler so What-If page still works (overrides the default one)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # 7. Feature engineering
    df = engineer_features(df)

    # Fallback: if CLV_Band has fewer than 2 unique values (can happen on tiny CSVs),
    # use quantile cut instead of equal-width bins
    if df["CLV_Band"].nunique() < 2 or df["CLV_Band"].isnull().all():
        log.warning("CLV band binning produced < 2 classes — switching to quantile cut")
        df["CLV_Band"] = pd.qcut(
            df["CLV_Score"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
        )

    # 8. Clustering (with user-specified K and DBSCAN eps)
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from src.cluster import map_segment_names, CLUSTER_FEATURES

    avail_cluster = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_c = df[avail_cluster].fillna(0).values

    k = hp.get("n_clusters", 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_c)
    joblib.dump(km, "models/kmeans_model.pkl")

    eps = hp.get("dbscan_eps", 0.5)
    min_s = hp.get("dbscan_min_samples", 5)
    db_model = DBSCAN(eps=eps, min_samples=min_s)
    db_labels = db_model.fit_predict(X_c)
    joblib.dump(db_model, "models/dbscan_model.pkl")

    # Silhouette comparison
    km_sil = 0.0
    db_sil = 0.0
    try:
        km_sil = silhouette_score(X_c, km_labels)
    except Exception:
        pass
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_db_clusters >= 2:
        try:
            mask = db_labels != -1
            db_sil = silhouette_score(X_c[mask], db_labels[mask])
        except Exception:
            pass

    df["KMeans_Cluster"] = km_labels
    df["KMeans_Segment"] = map_segment_names(km_labels)
    df["DBSCAN_Cluster"]  = db_labels
    df["DBSCAN_Segment"]  = map_segment_names(db_labels)
    df["Segment"] = df["KMeans_Segment"] if km_sil >= db_sil else df["DBSCAN_Segment"]
    primary = "K-Means" if km_sil >= db_sil else "DBSCAN"
    log.info("Primary segmentation: %s (sil=%.4f)", primary, max(km_sil, db_sil))

    # 9. Train classifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder as LE2
    from sklearn.metrics import classification_report, roc_auc_score

    feature_cols = [c for c in get_ml_feature_cols() if c in df.columns]
    X = df[feature_cols].fillna(0)
    y_raw = df["CLV_Band"].astype(str)
    le2 = LE2()
    y = le2.fit_transform(y_raw)

    # Need at least 2 classes
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Only one CLV band class found in this dataset. "
            "The data may be too uniform to train a classifier."
        )

    # Stratify only if each class has >= 2 samples
    class_counts = np.bincount(y)
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    n_est = hp.get("n_estimators", 200)
    lr    = hp.get("learning_rate", 0.05)

    clf = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=4,
        learning_rate=lr, subsample=0.8, random_state=42
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf,  "models/clv_classifier.pkl")
    joblib.dump(le2,  "models/clv_label_encoder.pkl")

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    report_str = classification_report(y_test, y_pred, target_names=le2.classes_)
    log.info("Classification report:\n%s", report_str)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = None
    log.info("ROC-AUC: %s", f"{auc:.4f}" if auc else "n/a")

    df["CLV_Band"] = le2.inverse_transform(clf.predict(X))

    metrics = {
        "report": report_str,
        "auc":    auc,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": le2.classes_,
        "silhouette": km_sil,
        "n_clusters": k,
        "primary_algo": primary,
        "n_rows": len(df),
        "feature_cols": feature_cols,
        "clf": clf,
    }

    # 10. Save enriched DB
    import sqlite3
    os.makedirs("data/processed", exist_ok=True)
    conn = sqlite3.connect("data/processed/customers_clean.db")
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    log.info("Enriched data saved to SQLite DB")

    # 11. PDF report
    pdf_path = None
    try:
        from src.report import generate_report as gen_pdf
        pdf_path = gen_pdf(df, metrics=metrics, dataset_name=dataset_name)
        log.info("PDF report: %s", pdf_path)
    except Exception as exc:
        log.warning("PDF generation skipped: %s", exc)

    return df, metrics, pdf_path


# Sidebar
dataset_store.render_sidebar()

st.title("Train on Your Data")
st.markdown(
    "Upload any e-commerce customer CSV, map columns to the expected schema, "
    "adjust hyperparameters, and run the full pipeline live. "
    "Results are stored session-wide so all other pages update automatically."
)

# Upload
uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="custom_upload")

if uploaded is None:
    st.info(
        "**Expected columns (required):** Age, Total Spend, Items Purchased, "
        "Days Since Last Purchase, Average Rating\n\n"
        "**Optional:** Gender, Membership Type, Satisfaction Level, "
        "Discount Applied, City"
    )
    st.stop()

# Reset state on new upload
if "last_upload_name" not in st.session_state or st.session_state["last_upload_name"] != uploaded.name:
    st.session_state["custom_df"]          = None
    st.session_state["custom_metrics"]     = None
    st.session_state["custom_report_path"] = None
    st.session_state["column_map"]         = {}
    st.session_state["last_upload_name"]   = uploaded.name

# Read CSV
try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

raw_df = _normalise_columns(raw_df)
st.markdown(f"**Uploaded:** `{uploaded.name}` — {len(raw_df):,} rows × {raw_df.shape[1]} columns")

# Data Quality Check
st.subheader("Data Quality Check")
(NUMERIC_COLS, CATEGORICAL_COLS, engineer_features,
 get_ml_feature_cols, run_clustering, train_model,
 check_quality, render_quality_report, generate_report, get_logger) = _safe_import()

quality = check_quality(raw_df)
render_quality_report(quality)

if not quality["can_train"]:
    st.error("Fix the errors above before training.")
    st.stop()

# Column Mapping
REQUIRED_PIPELINE_COLS = [
    "Age", "Total Spend", "Items Purchased",
    "Days Since Last Purchase", "Average Rating",
]
OPTIONAL_PIPELINE_COLS = [
    "Gender", "Membership Type", "Satisfaction Level",
    "Discount Applied", "City",
]

csv_cols = ["(not available)"] + list(raw_df.columns)
already_present = [c for c in REQUIRED_PIPELINE_COLS if c in raw_df.columns]
needs_mapping   = [c for c in REQUIRED_PIPELINE_COLS if c not in raw_df.columns]

col_map: dict[str, str] = {}
if needs_mapping:
    st.subheader("Column Mapping")
    st.markdown(
        "Some required columns were not detected automatically. "
        "Please map them to your CSV's columns below."
    )
    n = len(needs_mapping)
    cols = st.columns(min(n, 3))
    for i, req_col in enumerate(needs_mapping):
        with cols[i % 3]:
            chosen = st.selectbox(
                f"Which column is **{req_col}**?",
                options=csv_cols,
                key=f"map_{req_col}",
            )
            if chosen != "(not available)":
                col_map[req_col] = chosen
    st.session_state["column_map"] = col_map

if needs_mapping:
    still_missing = [c for c in needs_mapping if c not in col_map]
    if still_missing:
        st.warning(
            f"Still unmapped required columns: {still_missing}. "
            "The pipeline will attempt to run with defaults but accuracy may suffer."
        )

# Hyperparameter Panel
with st.expander("Advanced Settings (Hyperparameters)", expanded=False):
    hp_col1, hp_col2, hp_col3 = st.columns(3)
    with hp_col1:
        n_clusters   = st.slider("K-Means clusters (K)", 2, 8, 4)
        dbscan_eps   = st.slider("DBSCAN epsilon", 0.1, 2.0, 0.5, step=0.05)
    with hp_col2:
        dbscan_min_s = st.slider("DBSCAN min_samples", 2, 20, 5)
        n_estimators = st.slider("GB n_estimators", 50, 500, 200, step=50)
    with hp_col3:
        learning_rate = st.slider("GB learning_rate", 0.01, 0.3, 0.05, step=0.01)

hp = {
    "n_clusters":      n_clusters,
    "dbscan_eps":      dbscan_eps,
    "dbscan_min_samples": dbscan_min_s,
    "n_estimators":    n_estimators,
    "learning_rate":   learning_rate,
}

# Train Button
st.markdown("---")
if st.button("Run Full Pipeline", type="primary", use_container_width=True):
    with st.spinner("Running pipeline — preprocessing → clustering → training…"):
        try:
            df_out, metrics, pdf_path = _run_pipeline(
                raw_df,
                hp=hp,
                col_map=col_map,
                dataset_name=uploaded.name,
            )
            st.session_state["custom_df"]          = df_out
            st.session_state["custom_metrics"]     = metrics
            st.session_state["custom_report_path"] = pdf_path

            # Register dataset in store
            n_rows = len(df_out)
            n_segments = df_out["Segment"].nunique() if "Segment" in df_out.columns else 0
            clv_dist = df_out["CLV_Band"].value_counts().to_dict() if "CLV_Band" in df_out.columns else {}
            
            dataset_id = dataset_store.register_dataset(
                name=uploaded.name,
                db_path="data/processed/customers_clean.db",
                model_dir="models",
                n_rows=n_rows,
                n_segments=n_segments,
                clv_dist=clv_dist,
                auc=metrics.get("auc"),
                silhouette=metrics.get("silhouette"),
                is_default=False
            )
            st.session_state["newly_trained_id"] = dataset_id

            # Append to training history
            import datetime
            st.session_state["train_history"].append({
                "dataset_id": dataset_id,
                "Run":       len(st.session_state["train_history"]) + 1,
                "File":      uploaded.name,
                "Rows":      len(df_out),
                "Segments":  n_segments,
                "ROC-AUC":   f"{metrics['auc']:.4f}" if metrics.get("auc") else "n/a",
                "Silhouette": f"{metrics.get('silhouette', 0):.4f}",
                "Time":      datetime.datetime.now().strftime("%H:%M:%S"),
            })
            st.success("Pipeline complete!")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            st.stop()

# Results (shown if training has been run)
if st.session_state["custom_df"] is not None:
    df_out  = st.session_state["custom_df"]
    metrics = st.session_state["custom_metrics"]

    st.markdown("---")
    st.subheader("Results")

    new_id = st.session_state.get("newly_trained_id")
    if new_id:
        rec = dataset_store.get_dataset(new_id)
        if rec:
            st.success(
                f"✅ Training complete — {rec['name']}\n\n"
                f"{rec['n_rows']:,} customers · {rec['n_segments']} segments · "
                f"ROC-AUC {rec['auc'] if rec['auc'] else 'n/a'} · Silhouette {rec['silhouette'] if rec['silhouette'] else 'n/a'}\n\n"
                f"PDF report ready for download below."
            )
            if st.button("Set as active dataset", key="set_active_new_ds", type="primary", use_container_width=True):
                if dataset_store.activate_dataset(new_id):
                    st.session_state["active_dataset_id"] = new_id
                    del st.session_state["newly_trained_id"]
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success(f"Activated dataset: {rec['name']}")
                    st.rerun()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Customers",    f"{len(df_out):,}")
    r2.metric("Segments",     df_out["Segment"].nunique() if "Segment" in df_out.columns else "—")
    r3.metric("ROC-AUC",      f"{metrics['auc']:.4f}" if metrics.get("auc") else "n/a")
    r4.metric("Silhouette",   f"{metrics.get('silhouette', 0):.4f}")

    # CLV band distribution
    if "CLV_Band" in df_out.columns:
        import plotly.express as px
        clv_c = df_out["CLV_Band"].value_counts().reset_index()
        clv_c.columns = ["CLV Band", "Count"]
        fig = px.pie(
            clv_c, values="Count", names="CLV Band",
            color="CLV Band",
            color_discrete_map={"Low": "#F09595", "Medium": "#FAC775", "High": "#97C459"},
            hole=0.4, title="CLV Band Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Segment distribution
    if "Segment" in df_out.columns:
        import plotly.express as px
        seg_c = df_out["Segment"].value_counts().reset_index()
        seg_c.columns = ["Segment", "Count"]
        fig2 = px.bar(
            seg_c, x="Segment", y="Count",
            color="Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Customer Segments"
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Classification report
    if metrics.get("report"):
        with st.expander("Full Classification Report"):
            st.code(metrics["report"])

    # PDF download
    if st.session_state["custom_report_path"] and os.path.exists(st.session_state["custom_report_path"]):
        with open(st.session_state["custom_report_path"], "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(st.session_state["custom_report_path"]),
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
    else:
        st.info("PDF report was not generated (fpdf2 may not be installed).")

    # Raw data preview
    with st.expander("View enriched data (first 100 rows)"):
        st.dataframe(df_out.head(100), use_container_width=True)
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download enriched CSV",
            csv_bytes, "custom_enriched.csv", "text/csv"
        )

# Training History
if st.session_state["train_history"]:
    st.markdown("---")
    st.subheader("Training History (this session)")
    
    # Headers
    h_cols = st.columns([1, 3, 2, 2, 2, 2, 2, 2])
    h_cols[0].markdown("**Run**")
    h_cols[1].markdown("**File**")
    h_cols[2].markdown("**Rows**")
    h_cols[3].markdown("**Segments**")
    h_cols[4].markdown("**ROC-AUC**")
    h_cols[5].markdown("**Silhouette**")
    h_cols[6].markdown("**Time**")
    h_cols[7].markdown("**Restore**")
    
    for row in st.session_state["train_history"]:
        dataset_id = row.get("dataset_id")
        r_cols = st.columns([1, 3, 2, 2, 2, 2, 2, 2])
        r_cols[0].write(str(row["Run"]))
        r_cols[1].write(str(row["File"]))
        r_cols[2].write(f"{row['Rows']:,}" if isinstance(row['Rows'], int) else str(row['Rows']))
        r_cols[3].write(str(row["Segments"]))
        r_cols[4].write(str(row["ROC-AUC"]))
        r_cols[5].write(str(row["Silhouette"]))
        r_cols[6].write(str(row["Time"]))
        
        # Restore button
        if dataset_id:
            if r_cols[7].button("↩ Restore", key=f"restore_{dataset_id}"):
                if dataset_store.activate_dataset(dataset_id):
                    st.session_state["active_dataset_id"] = dataset_id
                    # Clear newly_trained_id in case it is restored
                    if "newly_trained_id" in st.session_state:
                        del st.session_state["newly_trained_id"]
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success(f"Restored dataset run: {row['File']}")
                    st.rerun()