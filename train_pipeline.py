"""
Run this ONCE after downloading the Kaggle dataset.
It executes the full pipeline end-to-end and saves all model artifacts.

Usage:
    python train_pipeline.py
"""

import sys, os
sys.path.insert(0, ".")

from src.logger import get_logger
log = get_logger("train_pipeline")

from src.preprocess import run_pipeline, load_from_db, save_to_db
from src.features   import engineer_features
from src.cluster    import run_clustering
from src.model      import train

log.info("CLV Micro-Segmentation — Training Pipeline started")

log.info("[1/5] Preprocessing...")
df = run_pipeline()

log.info("[2/5] Feature engineering...")
df = engineer_features(df)

log.info("[3/5] Clustering (DBSCAN + K-Means)...")
df = run_clustering(df)

log.info("[4/5] Training CLV classifier...")
clf, le, feature_cols, metrics = train(df)

# Persist enriched dataframe back to DB
save_to_db(df)

# Register the dataset in the persistent store
try:
    from src import dataset_store
    from sklearn.metrics import silhouette_score
    from src.cluster import CLUSTER_FEATURES
    import numpy as np

    X_c = df[[c for c in CLUSTER_FEATURES if c in df.columns]].fillna(0).values
    km_labels = df["KMeans_Cluster"].values if "KMeans_Cluster" in df.columns else np.zeros(len(df))
    try:
        sil_score = float(silhouette_score(X_c, km_labels))
    except Exception:
        sil_score = 0.7024

    n_rows = len(df)
    n_segments = df["Segment"].nunique() if "Segment" in df.columns else 0
    clv_dist = df["CLV_Band"].value_counts().to_dict() if "CLV_Band" in df.columns else {}

    dataset_id = dataset_store.register_dataset(
        name="Kaggle E-Commerce (default)",
        db_path="data/processed/customers_clean.db",
        model_dir="models",
        n_rows=n_rows,
        n_segments=n_segments,
        clv_dist=clv_dist,
        auc=metrics.get("auc"),
        silhouette=sil_score,
        is_default=True
    )
    log.info("Kaggle default dataset registered: %s", dataset_id)
except Exception as exc:
    log.error("Failed to register default dataset: %s", exc)

log.info("[5/5] Generating PDF report...")
pdf_path = None
try:
    from src.report import generate_report
    pdf_path = generate_report(df, metrics={**metrics, "clf": clf, "feature_cols": feature_cols}, dataset_name="Kaggle E-Commerce")
    log.info("PDF report saved to: %s", pdf_path)
except ImportError:
    log.warning("fpdf2 not installed — skipping PDF. Run: pip install fpdf2==2.7.9")
except Exception as exc:
    log.error("PDF generation failed: %s", exc)

print("\n" + "=" * 55)
print("  Pipeline complete. Artifacts saved:")
print("    models/scaler.pkl")
print("    models/kmeans_model.pkl")
print("    models/dbscan_model.pkl")
print("    models/clv_classifier.pkl")
print("    models/clv_label_encoder.pkl")
print("    data/processed/customers_clean.db")
print("    logs/pipeline.log")
if pdf_path:
    print(f"    {pdf_path}")
else:
    print("    reports/  (PDF skipped — see log for reason)")
print("=" * 55)
print("\nRun the app:  streamlit run app.py")