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

log.info("[5/5] Generating PDF report...")
pdf_path = None
try:
    from src.report import generate_report
    pdf_path = generate_report(df, metrics=metrics, dataset_name="Kaggle E-Commerce")
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