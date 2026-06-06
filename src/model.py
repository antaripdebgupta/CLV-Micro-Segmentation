"""
Trains a Gradient Boosting classifier to predict CLV band (Low/Medium/High).
Saves and loads trained model artifact.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

MODEL_PATH    = "models/clv_classifier.pkl"
ENCODER_PATH  = "models/clv_label_encoder.pkl"

from src.features import get_ml_feature_cols


def prepare_xy(df: pd.DataFrame) -> tuple:
    feature_cols = [c for c in get_ml_feature_cols() if c in df.columns]
    X = df[feature_cols].fillna(0)
    y_raw = df["CLV_Band"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    return X, y, le, feature_cols


def train(df: pd.DataFrame) -> tuple:
    X, y, le, feature_cols = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    print("[model] Classification Report:\n", report)
    if auc:
        print(f"[model] ROC-AUC (weighted OvR): {auc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"[model] Model saved to {MODEL_PATH}")

    return clf, le, feature_cols, {
        "report": report,
        "auc": auc,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": le.classes_,
        "clf": clf,
        "feature_cols": feature_cols,
    }


def load_model() -> tuple:
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    return clf, le


def predict_single(input_dict: dict) -> dict:
    """
    Predict CLV band for a single customer.
    input_dict: {feature_name: value, ...}
    Returns: {band: str, probabilities: {Low: float, Medium: float, High: float}}
    """
    clf, le = load_model()
    feature_cols = get_ml_feature_cols()
    X = pd.DataFrame([{c: input_dict.get(c, 0) for c in feature_cols}])
    proba  = clf.predict_proba(X)[0]
    label  = le.inverse_transform(clf.predict(X))[0]
    return {
        "band": label,
        "probabilities": dict(zip(le.classes_, proba.round(3)))
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    clf, le = load_model()
    feature_cols = [c for c in get_ml_feature_cols() if c in df.columns]
    X = df[feature_cols].fillna(0)
    preds  = clf.predict(X)
    probas = clf.predict_proba(X)
    df = df.copy()
    df["Predicted_CLV_Band"] = le.inverse_transform(preds)
    for i, cls in enumerate(le.classes_):
        df[f"Prob_{cls}"] = probas[:, i].round(3)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocess import load_from_db
    from src.features import engineer_features
    from src.cluster import run_clustering

    df = load_from_db()
    df = engineer_features(df)
    df = run_clustering(df)
    train(df)
