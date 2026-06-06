# Deployment Guide — CLV Micro-Segmentation

> All options below are **100% free** — no credit card required.

---

## Option 1 — Streamlit Community Cloud (Recommended)

Streamlit's own hosting platform. Zero configuration, automatic HTTPS, free subdomain.

### Prerequisites

- A GitHub account (free)
- This repository pushed to a **public** GitHub repo

### Steps

1. **Prepare the repo for deployment**

   The pipeline generates models, databases, and reports locally.
   For Streamlit Cloud to work, commit the essential artifacts:

   ```bash
   # Remove artifacts from .gitignore temporarily (or use git add -f)
   git add -f models/*.pkl
   git add -f data/processed/customers_clean.db
   git add -f data/processed/dataset_registry.json
   git add -f data/processed/datasets/
   git commit -m "Add trained artifacts for cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**

   - Go to [https://share.streamlit.io](https://share.streamlit.io)
   - Click **"New app"**
   - Select your GitHub repo and branch `main`
   - **Main file path:** leave as the default `streamlit_app.py`
     _(this thin wrapper re-exports `app.py`, so both files work identically)_
   - Click **Deploy**

3. **Your app is live** at `https://YOUR-APP-NAME.streamlit.app` within ~2 minutes.

### System Dependencies

If fonts or native libraries are needed (e.g., `fpdf2` requires font files), create a
file called `packages.txt` in the project root:

```
fonts-dejavu-core
```

Streamlit Cloud installs these apt packages automatically before starting the app.

### Secrets Management

If you ever need API keys or environment variables:

1. Go to your app's dashboard on Streamlit Cloud
2. Click **Settings → Secrets**
3. Add key-value pairs in TOML format:
   ```toml
   [general]
   API_KEY = "your_key_here"
   ```
4. Access in code: `st.secrets["general"]["API_KEY"]`

---

## Option 2 — Render (Free Tier)

Render offers free web services with automatic deploys from GitHub.

### Steps

1. **Create `render.yaml`** in the project root:

   ```yaml
   services:
     - type: web
       name: clv-segmentation
       runtime: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
       envVars:
         - key: PYTHON_VERSION
           value: "3.12"
   ```

2. **Push to GitHub** and connect at [https://render.com](https://render.com)
3. Select **"New Web Service"** → connect your repo → deploy

> **Note:** Render free tier spins down after 15 min of inactivity. First request
> after idle takes ~30s to cold-start.

---

## Option 3 — Hugging Face Spaces (Free)

Hugging Face Spaces supports Streamlit apps natively.

### Steps

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces) and create a **New Space**
2. Select **Streamlit** as the SDK
3. Clone the created Space repo locally:

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/clv-segmentation
   ```

4. Copy your project files into the Space repo:

   ```bash
   cp -r app.py pages/ src/ models/ data/ requirements.txt ./clv-segmentation/
   ```

5. Commit and push:

   ```bash
   cd clv-segmentation
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/clv-segmentation`

### Requirements

Create a `packages.txt` file if system-level fonts are needed:
```
fonts-dejavu-core
```

---

## Option 4 — Railway (Free Starter Plan)

Railway provides $5/month free credit — more than enough for a Streamlit app.

### Steps

1. Go to [https://railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select your repository
4. Add a `Procfile` to the project root:

   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```

5. Set the environment variable `PORT` to `8501` in the Railway dashboard
6. Deploy — your app gets a `.up.railway.app` URL

---

## Pre-Deployment Checklist

Before deploying to any platform, verify these:

| Check | Command / Action |
|-------|------------------|
| Pipeline runs cleanly | `python train_pipeline.py` |
| App starts locally | `streamlit run app.py` |
| All pages load | Visit each page in sidebar |
| Models are committed | `git status models/` shows tracked files |
| DB is committed | `git status data/processed/customers_clean.db` |
| requirements.txt is complete | `pip freeze > requirements.txt` (or verify manually) |
| No hardcoded absolute paths | Search for `/home/` in code — should find nothing |

### Resource Limits (Free Tiers)

| Platform | RAM | Storage | Sleep Policy |
|----------|-----|---------|--------------|
| Streamlit Cloud | 1 GB | 1 GB | Never sleeps |
| Render | 512 MB | 512 MB | Sleeps after 15 min idle |
| Hugging Face Spaces | 2 GB | 50 GB | Sleeps after 48 hr idle |
| Railway | 512 MB | 1 GB | No auto-sleep (credit-based) |

---

## Troubleshooting

### "ModuleNotFoundError" on deployment

Ensure `requirements.txt` lists all dependencies. Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Fonts missing in PDF (fpdf2)

Create `packages.txt` with `fonts-dejavu-core`. Streamlit Cloud and HF Spaces
install apt packages from this file automatically.

### App runs but shows "Database not found"

You forgot to commit model/data artifacts. Run:
```bash
git add -f models/ data/processed/
git commit -m "Add pipeline artifacts"
git push
```

### Memory errors on free tier

The dataset is small (~350 rows), so this shouldn't happen. If using a custom
CSV with >50K rows, consider downsampling or upgrading to a paid tier.
