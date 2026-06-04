"""
cluster.py
Runs DBSCAN and K-Means on engineered features.
Compares via silhouette score, saves best model.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os

KMEANS_PATH = "models/kmeans_model.pkl"
DBSCAN_PATH = "models/dbscan_model.pkl"

CLUSTER_FEATURES = [
    "Days Since Last Purchase_Scaled",
    "Total Spend_Scaled",
    "Items Purchased_Scaled",
    "Purchase_Prob",
    "Engagement_Score",
]

SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyalists",
    2: "At-Risk",
    3: "Lost",
}


def get_optimal_k(X: np.ndarray, k_range: range = range(2, 9)) -> int:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    optimal_k = max(scores, key=scores.get)
    print(f"[cluster] Silhouette scores: {scores}")
    print(f"[cluster] Optimal K = {optimal_k}")
    return optimal_k


def run_kmeans(X: np.ndarray, k: int = None) -> tuple:
    if k is None:
        k = get_optimal_k(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil   = silhouette_score(X, labels)
    db    = davies_bouldin_score(X, labels)
    print(f"[cluster] K-Means  | k={k} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(km, KMEANS_PATH)
    return labels, km, sil


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[cluster] DBSCAN   | clusters={n_clusters} | noise points={n_noise}")

    # Only score if we have >= 2 clusters
    sil = 0.0
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X[mask], labels[mask])
            db_score = davies_bouldin_score(X[mask], labels[mask])
            print(f"[cluster] DBSCAN   | Silhouette={sil:.4f} | Davies-Bouldin={db_score:.4f}")

    joblib.dump(db, DBSCAN_PATH)
    return labels, db, sil


def map_segment_names(labels: np.ndarray) -> pd.Series:
    """Map integer cluster labels to human-readable segment names."""
    unique = [l for l in np.unique(labels) if l != -1]
    mapping = {label: SEGMENT_NAMES.get(i, f"Segment {i}") for i, label in enumerate(unique)}
    mapping[-1] = "Outlier"
    return pd.Series(labels).map(mapping)


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[available].fillna(0).values

    km_labels, km_model, km_sil = run_kmeans(X)
    db_labels, db_model, db_sil = run_dbscan(X)

    df["KMeans_Cluster"]    = km_labels
    df["KMeans_Segment"]    = map_segment_names(km_labels)
    df["DBSCAN_Cluster"]    = db_labels
    df["DBSCAN_Segment"]    = map_segment_names(db_labels)

    # Use best model as primary segment column
    if km_sil >= db_sil:
        df["Segment"] = df["KMeans_Segment"]
        print("[cluster] Using K-Means as primary segmentation.")
    else:
        df["Segment"] = df["DBSCAN_Segment"]
        print("[cluster] Using DBSCAN as primary segmentation.")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocess import load_from_db
    from src.features import engineer_features

    df = load_from_db()
    df = engineer_features(df)
    df = run_clustering(df)
    print(df["Segment"].value_counts())
