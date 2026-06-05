"""
evaluate.py
Generates all evaluation artifacts:
  - Silhouette + Davies-Bouldin plots for clustering
  - Confusion matrix for CLV classification
  - Feature importance bar chart
  - SHAP summary plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
import joblib
import os

MODEL_PATH   = "models/clv_classifier.pkl"
ENCODER_PATH = "models/clv_label_encoder.pkl"


# Clustering Evaluation 

def plot_elbow(X: np.ndarray, k_range: range = range(2, 9)) -> plt.Figure:
    inertias   = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(list(k_range), inertias, marker="o", color="#534AB7")
    ax1.set_title("Elbow Curve — Inertia")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, marker="o", color="#1D9E75")
    ax2.set_title("Silhouette Score by K")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")

    fig.tight_layout()
    return fig


def cluster_metrics_table(X: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    if mask.sum() < 2:
        return {}
    sil = silhouette_score(X[mask], labels[mask])
    db  = davies_bouldin_score(X[mask], labels[mask])
    n_clusters = len(set(labels[mask]))
    return {
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin Index": round(db, 4),
        "Number of Clusters": n_clusters,
        "Noise Points (DBSCAN)": int((labels == -1).sum()),
    }


# Classification Evaluation 

def plot_confusion_matrix(y_true, y_pred, class_names: list) -> plt.Figure:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("CLV Band — Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(clf, feature_names: list, top_n: int = 12) -> plt.Figure:
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax, color="#7F77DD")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_shap_summary(clf, X_sample: pd.DataFrame) -> plt.Figure:
    try:
        import shap
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_sample)
        fig, ax = plt.subplots(figsize=(8, 5))
        # For multi-class use mean abs shap across classes
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals)
        imp = pd.Series(mean_abs.mean(axis=0), index=X_sample.columns).nlargest(12).sort_values()
        imp.plot(kind="barh", ax=ax, color="#D85A30")
        ax.set_title("SHAP Mean |value| — Feature Impact")
        ax.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"[evaluate] SHAP skipped: {e}")
        return None


# Segment Profile Chart 

def plot_segment_profiles(df: pd.DataFrame, segment_col: str = "Segment") -> plt.Figure:
    cols = ["Total Spend", "Items Purchased", "Days Since Last Purchase", "CLV_Score"]
    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    palette = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517"]

    for ax, col in zip(axes, cols):
        df.groupby(segment_col)[col].median().plot(
            kind="bar", ax=ax, color=palette[:df[segment_col].nunique()], edgecolor="none"
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Segment Median Profiles", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocess import load_from_db
    from src.features import engineer_features
    from src.cluster import run_clustering, CLUSTER_FEATURES
    from src.model import train

    df = load_from_db()
    df = engineer_features(df)
    df = run_clustering(df)
    clf, le, feature_cols, metrics = train(df)

    X = df[[c for c in CLUSTER_FEATURES if c in df.columns]].fillna(0).values
    print(cluster_metrics_table(X, df["KMeans_Cluster"].values))
