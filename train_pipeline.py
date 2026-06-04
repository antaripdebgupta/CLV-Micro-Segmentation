"""
train_pipeline.py
Run this ONCE after downloading the Kaggle dataset.
It executes the full pipeline end-to-end and saves all model artifacts.

Usage:
    python train_pipeline.py
"""

import sys, os
sys.path.insert(0, ".")

import sqlite3
import pandas as pd

from src.preprocess import run_pipeline, load_from_db, save_to_db
from src.features   import engineer_features
from src.cluster    import run_clustering
from src.model      import train

print("  CLV Micro-Segmentation — Training Pipeline")

print("\n[1/4] Preprocessing...")
df = run_pipeline()

print("\n[2/4] Feature engineering...")
df = engineer_features(df)

print("\n[3/4] Clustering (DBSCAN + K-Means)...")
df = run_clustering(df)

print("\n[4/4] Training CLV classifier...")
clf, le, feature_cols, metrics = train(df)

# Persist enriched dataframe back to DB
save_to_db(df)

print("\n" + "=" * 55)
print("  Pipeline complete. Artifacts saved:")
print("    models/scaler.pkl")
print("    models/kmeans_model.pkl")
print("    models/dbscan_model.pkl")
print("    models/clv_classifier.pkl")
print("    models/clv_label_encoder.pkl")
print("    data/processed/customers_clean.db")
print("=" * 55)
print("\nRun the app:  streamlit run app.py")
