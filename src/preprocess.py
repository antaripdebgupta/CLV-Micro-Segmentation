"""
preprocess.py
Loads raw Kaggle CSV, cleans it, encodes categoricals,
scales numerics, and saves to SQLite.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_PATH = "data/raw/ecommerce_customer_data.csv"
DB_PATH  = "data/processed/customers_clean.db"
SCALER_PATH = "models/scaler.pkl"

CATEGORICAL_COLS = ["Gender", "City", "Membership Type", "Satisfaction Level"]
NUMERIC_COLS = [
    "Age", "Days Since Last Purchase", "Total Spend",
    "Items Purchased", "Average Rating", "Discount Applied"
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().title().replace("  ", " ") for c in df.columns]

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Dropped {before - len(df)} duplicates")

    # Fill numeric nulls with median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip extreme outliers (IQR × 3)
    for col in ["Total Spend", "Items Purchased", "Days Since Last Purchase"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)

    print(f"[preprocess] Clean shape: {df.shape}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
    return df


def scale(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    df = df.copy()
    scale_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[[c + "_Scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Scaler saved to {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[[c + "_Scaled" for c in scale_cols]] = scaler.transform(df[scale_cols])

    return df


def save_to_db(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[preprocess] Saved {len(df):,} rows to {db_path}")


def load_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


def run_pipeline() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = encode(df)
    df = scale(df, fit=True)
    save_to_db(df)
    return df


if __name__ == "__main__":
    run_pipeline()
