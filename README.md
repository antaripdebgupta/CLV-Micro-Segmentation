# CLV Micro-Segmentation — E-Commerce Customer Analysis

> Final Year Data Science Project · 2026  
> **Stack:** Python · Pandas · Scikit-learn · Streamlit · Plotly · SQLite · SHAP

---

## What This Project Does

An end-to-end Data Science pipeline that:
1. **Segments** e-commerce customers using K-Means and DBSCAN
2. **Predicts** each customer's CLV band (Low / Medium / High) using Gradient Boosting
3. **Deploys** as an interactive Streamlit web app with three pages

---

## Project Structure

```
clv-segmentation/
├── app.py                        # Streamlit home page
├── train_pipeline.py             # One-shot training script
├── requirements.txt
├── data/
│   ├── raw/                      # Put Kaggle CSV here
│   └── processed/                # SQLite DB (auto-created)
├── src/
│   ├── preprocess.py             # Clean, encode, scale
│   ├── features.py               # BG/NBD score + CLV features
│   ├── cluster.py                # DBSCAN + K-Means
│   ├── model.py                  # Gradient Boosting classifier
│   └── evaluate.py               # All evaluation metrics
├── models/                       # Saved .pkl artifacts (auto-created)
└── pages/
    ├── 1_segment_explorer.py
    ├── 2_whatif_simulator.py
    └── 3_batch_upload.py
```

---

## Setup Guide

### Step 1 — Clone / Download the project

```bash
git clone https://github.com/YOUR_USERNAME/clv-segmentation.git
cd clv-segmentation
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Kaggle dataset

1. Go to: https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset
2. Download `E Commerce Customer Behavior - Sheet1.csv`
3. Rename it to `ecommerce_customer_data.csv`
4. Place it in `data/raw/`

### Step 5 — Run the training pipeline

```bash
python train_pipeline.py
```

This will:
- Clean and preprocess the data
- Engineer CLV features (BG/NBD score, engagement, spend per item)
- Run K-Means + DBSCAN clustering
- Train the Gradient Boosting CLV classifier
- Save all model artifacts to `models/`
- Save enriched data to `data/processed/customers_clean.db`

### Step 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploying to Streamlit Community Cloud (Free)

1. Push this project to a GitHub repository
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy**

Your app will be live at `https://YOUR_APP_NAME.streamlit.app` in ~2 minutes.

> **Note:** The training pipeline must be run locally first, and the `models/` and `data/processed/` folders must be committed to GitHub. Add them to git with `git add models/ data/processed/`.

---

## App Pages

| Page | Description |
|------|-------------|
| Home | KPI overview, segment + CLV distribution charts |
| Segment Explorer | Filter by segment / CLV band / membership type; EDA charts; heatmap |
| What-If Simulator | Adjust sliders → live CLV band prediction + recommended action |
| Batch Prediction | Upload CSV → download predictions; full evaluation report |

---

## ML Pipeline Summary

| Component | Detail |
|-----------|--------|
| Dataset | Kaggle E-Commerce Customer Behavior (~350 features) |
| Storage | SQLite via `sqlite3` (standard library) |
| Clustering | K-Means (elbow + silhouette), DBSCAN; best model selected automatically |
| CLV Feature | Manual BG/NBD-style purchase probability + engagement score |
| Classifier | `GradientBoostingClassifier` (200 estimators, lr=0.05) |
| Target | CLV Band: Low / Medium / High (tertile binning) |
| Evaluation | Silhouette, Davies-Bouldin, F1, ROC-AUC, Confusion Matrix, SHAP |
| Deployment | Streamlit Community Cloud (free) |

---

## Dataset Citation

E-Commerce Customer Behavior Dataset  
https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset
