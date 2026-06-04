"""
features.py
Engineers CLV-specific features including a manual BG/NBD-style
purchase probability score and CLV band labelling.
"""

import pandas as pd
import numpy as np


def compute_bgnbd_score(df: pd.DataFrame) -> pd.Series:
    """
    Simplified BG/NBD-style purchase probability proxy.
    Uses recency (days since last purchase) and frequency (items purchased)
    to estimate probability the customer is still active.

    Formula inspired by Pareto/NBD:
        P(alive) ≈ frequency / (frequency + recency_penalty)
    where recency_penalty scales with days since last purchase.
    """
    freq = df["Items Purchased"].clip(lower=1)
    recency = df["Days Since Last Purchase"].clip(lower=1)

    # Normalise both to 0-1 range
    freq_norm    = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
    recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-9)

    # Higher frequency + lower recency = higher P(alive)
    p_alive = freq_norm * (1 - recency_norm)
    return p_alive.rename("Purchase_Prob")


def compute_clv_score(df: pd.DataFrame) -> pd.Series:
    """
    Proxy CLV score = avg_spend_per_item × frequency × purchase_probability × satisfaction_weight
    """
    avg_spend  = df["Total Spend"] / df["Items Purchased"].clip(lower=1)
    freq       = df["Items Purchased"]
    p_alive    = compute_bgnbd_score(df)
    satisfaction = df.get("Average Rating", pd.Series(3.0, index=df.index))
    sat_weight = satisfaction / 5.0  # normalise rating to 0-1

    clv = avg_spend * freq * p_alive * sat_weight
    return clv.rename("CLV_Score")


def assign_clv_band(clv_series: pd.Series) -> pd.Series:
    """
    Bin CLV scores into Low / Medium / High using tertile cut.
    """
    labels = ["Low", "Medium", "High"]
    return pd.cut(clv_series, bins=3, labels=labels).rename("CLV_Band")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Purchase_Prob"] = compute_bgnbd_score(df)
    df["CLV_Score"]     = compute_clv_score(df)
    df["CLV_Band"]      = assign_clv_band(df["CLV_Score"])

    # Engagement score (composite)
    df["Engagement_Score"] = (
        0.4 * (df["Items Purchased"] / df["Items Purchased"].max()) +
        0.3 * (df.get("Average Rating", 3) / 5) +
        0.3 * df["Purchase_Prob"]
    )

    # Spend per visit proxy
    df["Spend_Per_Item"] = df["Total Spend"] / df["Items Purchased"].clip(lower=1)

    print(f"[features] CLV band distribution:\n{df['CLV_Band'].value_counts()}")
    return df


def get_ml_feature_cols() -> list:
    """Returns the feature columns used for ML model training."""
    return [
        "Age_Scaled",
        "Days Since Last Purchase_Scaled",
        "Total Spend_Scaled",
        "Items Purchased_Scaled",
        "Average Rating_Scaled",
        "Discount Applied_Scaled",
        "Satisfaction Level_Enc",
        "Purchase_Prob",
        "Engagement_Score",
        "Spend_Per_Item",
        "Gender_Enc",
        "Membership Type_Enc",
    ]


if __name__ == "__main__":
    from preprocess import load_from_db
    df = load_from_db()
    df = engineer_features(df)
    print(df[["CLV_Score", "CLV_Band", "Purchase_Prob"]].head())
