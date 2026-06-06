"""
Persistent registry of all trained datasets.
Stored as JSON at data/processed/dataset_registry.json.
Allows the app to switch between any previously trained dataset.
"""

import json
import os
import datetime
import shutil
from pathlib import Path

REGISTRY_PATH = "data/processed/dataset_registry.json"
DATASETS_DIR  = "data/processed/datasets"   # one sub-folder per dataset


def _load_registry() -> list[dict]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_registry(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def register_dataset(
    name: str,
    db_path: str,
    model_dir: str,
    n_rows: int,
    n_segments: int,
    clv_dist: dict,
    auc: float | None,
    silhouette: float | None,
    is_default: bool = False,
) -> str:
    """
    Register a newly trained dataset. Returns its unique dataset_id.
    Also copies the trained models into a dataset-specific folder so
    switching datasets restores the correct models.
    """
    records = _load_registry()
    dataset_id = f"ds_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Copy models into a dataset-specific folder
    ds_dir = os.path.join(DATASETS_DIR, dataset_id)
    models_dst = os.path.join(ds_dir, "models")
    os.makedirs(models_dst, exist_ok=True)
    for fname in ["clv_classifier.pkl", "clv_label_encoder.pkl",
                  "scaler.pkl", "kmeans_model.pkl", "dbscan_model.pkl"]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(models_dst, fname))

    # Copy DB into dataset-specific folder
    db_dst = os.path.join(ds_dir, "customers_clean.db")
    if os.path.exists(db_path):
        shutil.copy2(db_path, db_dst)

    record = {
        "dataset_id":  dataset_id,
        "name":        name,
        "db_path":     db_dst,
        "model_dir":   models_dst,
        "n_rows":      n_rows,
        "n_segments":  n_segments,
        "clv_dist":    clv_dist,
        "auc":         round(auc, 4) if auc else None,
        "silhouette":  round(silhouette, 4) if silhouette else None,
        "is_default":  is_default,
        "trained_at":  datetime.datetime.now().isoformat(),
    }
    records.append(record)
    _save_registry(records)
    return dataset_id


def list_datasets() -> list[dict]:
    """Return all registered datasets, newest first."""
    return list(reversed(_load_registry()))


def get_dataset(dataset_id: str) -> dict | None:
    for r in _load_registry():
        if r["dataset_id"] == dataset_id:
            return r
    return None


def get_default_dataset() -> dict | None:
    for r in reversed(_load_registry()):
        if r.get("is_default"):
            return r
    return None


def activate_dataset(dataset_id: str) -> bool:
    """
    Copy a registered dataset's DB and models back into the live paths
    so all pages pick them up. Returns True on success.
    """
    record = get_dataset(dataset_id)
    if not record:
        return False

    # Restore DB
    os.makedirs("data/processed", exist_ok=True)
    if os.path.exists(record["db_path"]):
        shutil.copy2(record["db_path"], "data/processed/customers_clean.db")

    # Restore models
    os.makedirs("models", exist_ok=True)
    for fname in ["clv_classifier.pkl", "clv_label_encoder.pkl",
                  "scaler.pkl", "kmeans_model.pkl", "dbscan_model.pkl"]:
        src = os.path.join(record["model_dir"], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("models", fname))
    return True


def get_active_dataset_record() -> dict | None:
    import streamlit as st
    active_id = st.session_state.get("active_dataset_id")
    if active_id:
        record = get_dataset(active_id)
        if record:
            return record
    # Fallback to default
    record = get_default_dataset()
    if record:
        st.session_state["active_dataset_id"] = record["dataset_id"]
        return record
    # Fallback to first available
    datasets = list_datasets()
    if datasets:
        st.session_state["active_dataset_id"] = datasets[0]["dataset_id"]
        return datasets[0]
    return None


def render_sidebar():
    import streamlit as st
    st.sidebar.title("CLV Segmentation")
    st.sidebar.caption("E-Commerce Customer Behaviour Analysis")
    st.sidebar.markdown("---")
    
    datasets = list_datasets()
    if datasets:
        with st.sidebar.expander("Active Dataset", expanded=True):
            options = {d["dataset_id"]: d["name"] for d in datasets}
            active_id = st.session_state.get("active_dataset_id")
            if not active_id or active_id not in options:
                active_id = datasets[0]["dataset_id"]
                st.session_state["active_dataset_id"] = active_id
                
            ds_ids = list(options.keys())
            default_idx = ds_ids.index(active_id) if active_id in ds_ids else 0
            
            selected_id = st.selectbox(
                "Select Dataset",
                options=ds_ids,
                format_func=lambda x: options[x],
                index=default_idx,
                key="sidebar_dataset_select"
            )
            
            if st.button("Switch Dataset", key="sidebar_switch_btn", use_container_width=True):
                if selected_id != st.session_state.get("active_dataset_id"):
                    if activate_dataset(selected_id):
                        st.session_state["active_dataset_id"] = selected_id
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success(f"Switched to {options[selected_id]}")
                        st.rerun()
    else:
        st.sidebar.info("No datasets registered yet.")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    
    # Detect which entry-point script Streamlit is running (app.py locally,
    # streamlit_app.py on Streamlit Cloud) so page_link always resolves.
    # We check which file exists; streamlit_app.py takes priority when present
    # because Streamlit Cloud uses it as the default main file.
    _entry = "streamlit_app.py" if os.path.exists("streamlit_app.py") else "app.py"
    st.sidebar.page_link(_entry,                      label="Home / Dashboard",    icon="🏠")
    st.sidebar.page_link("pages/1_segment_explorer.py",  label="Segment Explorer",    icon="🗂️")
    st.sidebar.page_link("pages/2_whatif_simulator.py",  label="What-If Simulator",   icon="🔮")
    st.sidebar.page_link("pages/3_batch_upload.py",       label="Batch Prediction",    icon="📂")
    st.sidebar.page_link("pages/4_train_custom.py",       label="Train on Your Data",  icon="🧪")
    st.sidebar.page_link("pages/5_log_viewer.py",         label="Log Viewer",          icon="📋")
    st.sidebar.markdown("---")
    st.sidebar.caption("Final Year DS Project · 2026")
