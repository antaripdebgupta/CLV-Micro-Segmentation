"""
Generates a professional multi-page PDF report after training.

Sections:
  1. Cover page  — dataset name, date, row count, model summary
  2. EDA         — up to 3 notebook PNGs (univariate, correlation, CLV band)
  3. Clustering  — elbow + segment profile PNGs + metrics table
  4. Model       — confusion matrix + feature importance + classification report
  5. Business    — segment action recommendations table
"""

import os
import io
import datetime
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR  = "reports"
NOTEBOOKS_DIR = "notebooks"

# Segment → (interpretation, recommended action)
SEGMENT_ACTIONS = {
    "Champions":  ("High-value, frequent, recent buyers.",   "VIP loyalty rewards + early product access."),
    "Loyalists":  ("Consistent buyers with moderate spend.",  "Personalised email campaigns + cross-sell."),
    "At-Risk":    ("Previously active, now drifting away.",   "Win-back discounts + re-engagement emails."),
    "Lost":       ("Haven't purchased in a long time.",       "Deep-discount reactivation or suppress."),
    "Outlier":    ("Atypical behaviour — DBSCAN noise.",      "Manual review before targeting."),
}

# Fallback for generic cluster names like "Segment 0"
FALLBACK_ACTION = ("Behaviour not yet characterised.", "Monitor and classify with more data.")


# Helpers

def _notebook_png(filename: str) -> str | None:
    """Return full path if the PNG exists inside reports/tmp_plots/ or notebooks/, else None."""
    tmp_path = os.path.join(REPORTS_DIR, "tmp_plots", filename)
    if os.path.exists(tmp_path):
        return tmp_path
    path = os.path.join(NOTEBOOKS_DIR, filename)
    return path if os.path.exists(path) else None


def _df_to_png(df: pd.DataFrame, title: str = "") -> str:
    """
    Render a small DataFrame as a matplotlib table PNG and return its path.
    Unicode characters in cell text (arrows, em-dashes) are preserved because
    matplotlib renders via FreeType, not a PDF font subset.
    """
    tmp = os.path.join(REPORTS_DIR, "_tmp_table.png")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Replace any special Unicode arrows with ASCII equivalents so the PNG
    # looks clean regardless of the system's font coverage.
    def _clean(val):
        return (
            str(val)
            .replace("↑", "(+)")
            .replace("↓", "(-)")
            .replace("—", "-")
        )

    clean_values = [[_clean(v) for v in row] for row in df.values.tolist()]
    clean_cols   = [_clean(c) for c in df.columns]

    n_rows = len(df)
    fig_h  = max(1.5, 0.5 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=clean_values,
        colLabels=clean_cols,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Style header row
    for j in range(len(clean_cols)):
        tbl[0, j].set_facecolor("#534AB7")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row shading
    for i in range(1, n_rows + 1):
        for j in range(len(clean_cols)):
            tbl[i, j].set_facecolor("#F5F5FA" if i % 2 == 0 else "white")

    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    fig.tight_layout(pad=0.5)
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tmp


def _clamp_image(pdf, img_path: str, max_w: float = 170, max_h: float = 100) -> None:
    """
    Add an image to the PDF, scaled to fit within max_w × max_h mm.
    Automatically adds a new page if the image would overflow the current one.
    """
    from PIL import Image as PILImage  # fpdf2 bundles Pillow

    try:
        with PILImage.open(img_path) as im:
            w_px, h_px = im.size
    except Exception:
        return  # corrupt image — skip silently

    if h_px == 0:
        return

    aspect = w_px / h_px
    w_mm = min(max_w, aspect * max_h)
    h_mm = w_mm / aspect
    if h_mm > max_h:
        h_mm = max_h
        w_mm = h_mm * aspect

    # Page geometry: usable height = page height - top margin - bottom margin
    usable_h = pdf.h - pdf.t_margin - pdf.b_margin
    # Clamp image height to fit on a single page
    if h_mm > usable_h - 10:
        h_mm = usable_h - 10
        w_mm = h_mm * aspect

    # Remaining space on current page (distance from cursor to bottom margin)
    remaining = pdf.h - pdf.b_margin - pdf.get_y()

    # If the image (plus a small 6 mm buffer for the caption) won't fit, new page
    if h_mm + 6 > remaining:
        pdf.add_page()

    x = (pdf.w - w_mm) / 2  # centre horizontally
    pdf.image(img_path, x=x, y=pdf.get_y(), w=w_mm, h=h_mm)
    pdf.ln(h_mm + 4)


# Main generator

def _generate_all_plots(df: pd.DataFrame, metrics: dict | None, tmp_dir: str) -> None:
    """Generate all the evaluation and EDA plots dynamically and save them in tmp_dir."""
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. eda_univariate.png
    cols_to_plot = [c for c in ["Age", "Total Spend", "Items Purchased", "Days Since Last Purchase", "Average Rating"] if c in df.columns]
    if cols_to_plot:
        n_cols = len(cols_to_plot)
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 2.5))
        if n_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            sns.histplot(df[col], kde=True, ax=ax, color="#534AB7")
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "eda_univariate.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        
    # 2. eda_correlation.png
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if not (c.endswith("_Scaled") or c.endswith("_Enc") or c.endswith("Cluster") or "ID" in c or c == "Customer ID")]
    if num_cols:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True)
        ax.set_title("Pearson Correlation Heatmap", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "eda_correlation.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        
    # 3. eda_clv_band.png
    if "CLV_Band" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["CLV_Band"].value_counts()
        colors = {"Low": "#F09595", "Medium": "#FAC775", "High": "#97C459"}
        color_list = [colors.get(idx, "#534AB7") for idx in counts.index]
        counts.plot(kind="bar", color=color_list, edgecolor="none", ax=ax)
        ax.set_title("CLV Band Distribution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "eda_clv_band.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 4. clustering_elbow.png
    from src.cluster import CLUSTER_FEATURES
    avail = [c for c in CLUSTER_FEATURES if c in df.columns]
    if avail:
        X_c = df[avail].fillna(0).values
        k_range = range(2, min(7, len(df)))
        k_list = list(k_range)
        if len(k_list) >= 2:
            inertias = []
            silhouettes = []
            for k in k_list:
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = km.fit_predict(X_c)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_c, labels))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
            ax1.plot(k_list, inertias, marker="o", color="#534AB7")
            ax1.set_title("Elbow Curve — Inertia")
            ax1.set_xlabel("K")
            ax1.set_ylabel("Inertia")
            ax2.plot(k_list, silhouettes, marker="o", color="#1D9E75")
            ax2.set_title("Silhouette Score by K")
            ax2.set_xlabel("K")
            ax2.set_ylabel("Silhouette Score")
            fig.tight_layout()
            fig.savefig(os.path.join(tmp_dir, "clustering_elbow.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

    # 5. clustering_kmeans_pca.png
    cluster_col = "Segment" if "Segment" in df.columns else ("KMeans_Cluster" if "KMeans_Cluster" in df.columns else None)
    if cluster_col and avail:
        X_c = df[avail].fillna(0).values
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_c)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        categories = pd.Categorical(df[cluster_col]).categories
        codes = pd.Categorical(df[cluster_col]).codes
        
        try:
            cmap = plt.colormaps["Set2"]
        except (AttributeError, KeyError):
            cmap = plt.cm.get_cmap("Set2")
            
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=codes, cmap=cmap, alpha=0.8, edgecolors="none")
        ax.set_title(f"Segments Projected on PCA Space", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / max(1, len(categories) - 1)), markersize=8) for i in range(len(categories))]
        ax.legend(handles, categories, title="Segment")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "clustering_kmeans_pca.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 6. clustering_segment_profile.png
    if "Segment" in df.columns:
        cols = ["Total Spend", "Items Purchased", "Days Since Last Purchase", "CLV_Score"]
        cols = [c for c in cols if c in df.columns]
        if cols:
            fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3))
            if len(cols) == 1:
                axes = [axes]
            palette = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#97C459"]
            unique_segs = df["Segment"].dropna().unique()
            for ax, col in zip(axes, cols):
                df.groupby("Segment")[col].median().plot(
                    kind="bar", ax=ax, color=palette[:len(unique_segs)], edgecolor="none"
                )
                ax.set_title(col, fontsize=10)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle("Segment Median Profiles", y=1.05, fontsize=11, fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(tmp_dir, "clustering_segment_profile.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

    # 7. model_confusion_matrix.png
    if metrics and "y_test" in metrics and "y_pred" in metrics and "classes" in metrics:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=metrics["classes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("CLV Band — Confusion Matrix", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "model_confusion_matrix.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 8. model_feature_importance.png
    clf = None
    if os.path.exists("models/clv_classifier.pkl"):
        try:
            import joblib
            clf = joblib.load("models/clv_classifier.pkl")
        except Exception as exc:
            log.warning("Could not load classifier: %s", exc)
            
    if clf and hasattr(clf, "feature_importances_") and metrics and "feature_cols" in metrics:
        importances = pd.Series(clf.feature_importances_, index=metrics["feature_cols"])
        top = importances.nlargest(12).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        top.plot(kind="barh", ax=ax, color="#7F77DD")
        ax.set_title("Top Feature Importances", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "model_feature_importance.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 9. model_roc_curves.png
    if metrics and "y_test" in metrics and "y_proba" in metrics and "classes" in metrics:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        classes = list(metrics["classes"])
        y_test_bin = label_binarize(metrics["y_test"], classes=range(len(classes)))
        y_proba = metrics["y_proba"]
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for i, class_name in enumerate(classes):
            if y_test_bin.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
            else:
                fpr, tpr, _ = roc_curve(metrics["y_test"], y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
                break
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "model_roc_curves.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 10. model_shap.png
    try:
        import shap
        if metrics and "X_test" in metrics and clf:
            X_sample = metrics["X_test"].sample(min(100, len(metrics["X_test"])), random_state=42)
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_sample)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            if isinstance(shap_vals, list):
                mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            else:
                mean_abs = np.abs(shap_vals)
            imp = pd.Series(mean_abs.mean(axis=0), index=X_sample.columns).nlargest(12).sort_values()
            imp.plot(kind="barh", ax=ax, color="#D85A30")
            ax.set_title("SHAP Mean |value| — Feature Impact", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean |SHAP value|")
            fig.tight_layout()
            fig.savefig(os.path.join(tmp_dir, "model_shap.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        log.warning("SHAP generation failed, skipping SHAP plot: %s", e)


def generate_report(
    df: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
    dataset_name: str = "E-Commerce Customer Data",
    output_dir: str = REPORTS_DIR,
) -> str:
    """
    Generate a PDF report and save it to output_dir.

    Args:
        df:           Enriched DataFrame (after all pipeline steps).
        metrics:      Dict returned by src.model.train() — may be None.
        dataset_name: Label shown on the cover page.
        output_dir:   Destination folder (created if absent).

    Returns:
        Absolute path of the saved PDF.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        log.error("fpdf2 not installed. Run: pip install fpdf2")
        raise

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path  = os.path.join(output_dir, f"clv_report_{timestamp}.pdf")

    # Setup temporary plots directory
    tmp_plots_dir = os.path.join(output_dir, "tmp_plots")
    os.makedirs(tmp_plots_dir, exist_ok=True)

    try:
        # Generate the figures dynamically based on the current df and metrics
        try:
            _generate_all_plots(df, metrics, tmp_plots_dir)
        except Exception as exc:
            log.warning("Dynamic figure generation failed, falling back to defaults: %s", exc)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=15, top=15, right=15)

        # Register DejaVuSans TTF if available
        font_family = "Helvetica"
        try:
            possible_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/local/share/fonts/DejaVuSans.ttf",
                str(Path.home() / ".fonts/DejaVuSans.ttf"),
            ]
            ttf_path = next(p for p in possible_paths if os.path.exists(p))
            pdf.add_font("DejaVu", "", ttf_path, uni=True)
            # attempt common variants
            if os.path.exists(ttf_path.replace('.ttf', '-Bold.ttf')):
                pdf.add_font("DejaVu", "B", ttf_path.replace('.ttf', '-Bold.ttf'), uni=True)
            elif os.path.exists(ttf_path.replace('.ttf', 'Bold.ttf')):
                pdf.add_font("DejaVu", "B", ttf_path.replace('.ttf', 'Bold.ttf'), uni=True)
            if os.path.exists(ttf_path.replace('.ttf', '-Oblique.ttf')):
                pdf.add_font("DejaVu", "I", ttf_path.replace('.ttf', '-Oblique.ttf'), uni=True)
            elif os.path.exists(ttf_path.replace('.ttf', 'Oblique.ttf')):
                pdf.add_font("DejaVu", "I", ttf_path.replace('.ttf', 'Oblique.ttf'), uni=True)

            font_family = "DejaVu"
            log.info("Registered PDF font: %s", ttf_path)
        except StopIteration:
            log.warning("DejaVuSans TTF not found; PDF may fail on some Unicode characters.")
        except Exception as exc:
            log.warning("Failed to register TTF font for PDF: %s", exc)

        def _set_font(style: str = "", size: int = 10):
            pdf.set_font(font_family, style, size)

        def _s(text: str) -> str:
            """Sanitise text for PDF output: replace Unicode dashes and symbols with ASCII."""
            t = str(text)
            t = t.replace("—", "-").replace("–", "-").replace("·", "-")
            t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2022", "*")
            try:
                t.encode("latin-1")
            except UnicodeEncodeError:
                t = "".join(c if ord(c) < 256 else "?" for c in t)
            return t

        # Page helpers 
        def add_cover():
            pdf.add_page()
            pdf.set_fill_color(83, 74, 183)
            pdf.rect(0, 0, pdf.w, 60, "F")

            pdf.set_text_color(255, 255, 255)
            _set_font("B", 22)
            pdf.ln(14)
            pdf.cell(0, 10, "CLV Micro-Segmentation", align="C", new_x="LMARGIN", new_y="NEXT")
            _set_font("", 13)
            pdf.cell(0, 8, "E-Commerce Customer Behaviour Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(22)

            pdf.set_text_color(30, 30, 30)
            _set_font("B", 13)
            pdf.cell(0, 8, "Report Summary", new_x="LMARGIN", new_y="NEXT")
            _set_font("", 11)

            generated = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
            n_customers = len(df)
            n_segments  = df["Segment"].nunique() if "Segment" in df.columns else "-"
            clv_dist    = df["CLV_Band"].value_counts().to_dict() if "CLV_Band" in df.columns else {}

            rows = [
                ("Dataset",       dataset_name),
                ("Generated",     generated),
                ("Total Customers", f"{n_customers:,}"),
                ("Segments Found",  str(n_segments)),
                ("CLV High",      f"{clv_dist.get('High', '-')}") ,
                ("CLV Medium",    f"{clv_dist.get('Medium', '-')}") ,
                ("CLV Low",       f"{clv_dist.get('Low', '-')}") ,
            ]
            if metrics and metrics.get("auc"):
                rows.append(("Model ROC-AUC", f"{metrics['auc']:.4f}"))

            pdf.set_fill_color(245, 245, 250)
            for label, value in rows:
                _set_font("B", 10)
                pdf.cell(60, 7, _s(label), border=0, new_x="RIGHT", new_y="TOP", fill=True)
                _set_font("", 10)
                pdf.cell(0, 7, _s(value), border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(1)

            pdf.ln(8)
            _set_font("I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, _s("Final Year Data Science Project · 2026"), align="C")

        def add_section_title(title: str):
            _set_font("B", 14)
            pdf.set_text_color(83, 74, 183)
            pdf.ln(4)
            pdf.cell(0, 9, _s(title), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(83, 74, 183)
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_text_color(30, 30, 30)
            pdf.ln(4)

        def add_paragraph(text: str):
            _set_font("", 10)
            pdf.set_text_color(50, 50, 50)
            for line in textwrap.wrap(text, width=100):
                pdf.cell(0, 6, _s(line), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        def _ensure_space(min_mm: float = 30) -> None:
            """Add a new page if fewer than min_mm remain on the current page."""
            remaining = pdf.h - pdf.b_margin - pdf.get_y()
            if remaining < min_mm:
                pdf.add_page()

        def try_add_image(filename: str, caption: str = ""):
            path = _notebook_png(filename)
            if path:
                try:
                    _clamp_image(pdf, path)
                    if caption:
                        _ensure_space(10)
                        _set_font("I", 9)
                        pdf.set_text_color(120, 120, 120)
                        pdf.cell(0, 5, _s(caption), align="C", new_x="LMARGIN", new_y="NEXT")
                        pdf.set_text_color(30, 30, 30)
                        pdf.ln(4)
                except Exception as exc:
                    log.warning("Could not embed image %s: %s", filename, exc)
            else:
                _set_font("I", 9)
                pdf.set_text_color(160, 160, 160)
                pdf.cell(0, 6, _s(f"[Chart not available: {filename}]"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(30, 30, 30)

        # Section 1: Cover
        add_cover()

        # Section 2: EDA
        pdf.add_page()
        add_section_title("1. Exploratory Data Analysis")
        add_paragraph(
            "The dataset was explored for distribution shapes, outliers, and feature correlations "
            "before any modelling steps. Key observations are summarised in the charts below."
        )
        try_add_image("eda_univariate.png",    "Figure 1 — Univariate distributions of numeric features")
        try_add_image("eda_correlation.png",   "Figure 2 — Pearson correlation heatmap")
        try_add_image("eda_clv_band.png",      "Figure 3 — CLV band distribution (tertile binning)")

        # Section 3: Clustering
        pdf.add_page()
        add_section_title("2. Customer Segmentation (Clustering)")
        add_paragraph(
            "K-Means and DBSCAN were both applied to scaled features "
            "(spend, recency, frequency, purchase probability, engagement score). "
            "The model with the higher silhouette score was chosen as the primary segmentation."
        )
        try_add_image("clustering_elbow.png",           "Figure 4 — Elbow + silhouette curves (K-Means)")
        try_add_image("clustering_kmeans_pca.png",       "Figure 5 — K-Means clusters projected onto PCA axes")
        try_add_image("clustering_segment_profile.png",  "Figure 6 — Median feature profiles per segment")

        # Clustering metrics table
        if "KMeans_Cluster" in df.columns:
            try:
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                from src.cluster import CLUSTER_FEATURES
                avail = [c for c in CLUSTER_FEATURES if c in df.columns]
                X_c   = df[avail].fillna(0).values
                km_labels = df["KMeans_Cluster"].values
                mask = km_labels != -1
                if mask.sum() > 1 and len(set(km_labels[mask])) >= 2:
                    sil = silhouette_score(X_c[mask], km_labels[mask])
                    dbi = davies_bouldin_score(X_c[mask], km_labels[mask])
                    metrics_df = pd.DataFrame({
                        "Metric": ["Silhouette Score (+ better)", "Davies-Bouldin Index (- better)",
                                   "Number of Clusters"],
                        "Value": [f"{sil:.4f}", f"{dbi:.4f}", str(len(set(km_labels[mask])))]
                    })
                    tmp = _df_to_png(metrics_df, "K-Means Clustering Metrics")
                    _ensure_space(60)
                    _clamp_image(pdf, tmp, max_h=50)
            except Exception as exc:
                log.warning("Clustering metrics table skipped: %s", exc)

        # Section 4: Model
        pdf.add_page()
        add_section_title("3. CLV Band Classification (Gradient Boosting)")
        add_paragraph(
            "A Gradient Boosting classifier (200 estimators, lr=0.05) was trained to predict "
            "each customer's CLV band — Low, Medium, or High — using 12 engineered features. "
            "The model was evaluated on a stratified 20% hold-out test set."
        )
        try_add_image("model_confusion_matrix.png",  "Figure 7 — Confusion matrix on test set")
        try_add_image("model_feature_importance.png","Figure 8 — Top feature importances (Gini)")
        try_add_image("model_roc_curves.png",        "Figure 9 — One-vs-Rest ROC curves per CLV band")
        try_add_image("model_shap.png",              "Figure 10 — SHAP mean |value| feature impact")

        # Classification report table
        if metrics and "report" in metrics:
            try:
                from sklearn.metrics import classification_report
                report_dict = classification_report(
                    metrics["y_test"], metrics["y_pred"],
                    target_names=metrics["classes"], output_dict=True
                )
                rpt_df = pd.DataFrame(report_dict).T.round(3).reset_index()
                rpt_df.columns = ["Class"] + list(rpt_df.columns[1:])
                tmp = _df_to_png(rpt_df, "Classification Report")
                _ensure_space(70)
                _clamp_image(pdf, tmp, max_h=60)
            except Exception as exc:
                log.warning("Classification report table skipped: %s", exc)

        # Section 5: Business Recommendations
        pdf.add_page()
        add_section_title("4. Business Recommendations by Segment")
        add_paragraph(
            "Each customer segment maps to a business interpretation and a recommended "
            "retention or acquisition action. The table below should guide marketing decisions."
        )

        segments_present = (
            df["Segment"].dropna().unique().tolist() if "Segment" in df.columns
            else list(SEGMENT_ACTIONS.keys())
        )
        segments_present = [str(s) for s in segments_present]

        rec_rows = []
        for seg in sorted(set(segments_present)):
            interp, action = SEGMENT_ACTIONS.get(seg, FALLBACK_ACTION)
            n = int((df["Segment"] == seg).sum()) if "Segment" in df.columns else "—"
            rec_rows.append([seg, str(n), interp, action])

        rec_df = pd.DataFrame(rec_rows, columns=["Segment", "Count", "Interpretation", "Action"])
        tmp = _df_to_png(rec_df, "Segment Action Table")
        _ensure_space(90)
        _clamp_image(pdf, tmp, max_h=80)

        _ensure_space(20)
        add_paragraph(
            "Priority: focus retention budget on 'Champions' and 'Loyalists' first. "
            "'At-Risk' customers offer the highest ROI for win-back campaigns. "
            "'Lost' customers should only receive low-cost outreach."
        )

        # Save
        pdf.output(pdf_path)
        log.info("PDF saved to %s", os.path.abspath(pdf_path))
        return os.path.abspath(pdf_path)
    finally:
        import shutil
        if os.path.exists(tmp_plots_dir):
            try:
                shutil.rmtree(tmp_plots_dir)
            except Exception as e:
                log.warning("Failed to clean up temp plots directory: %s", e)


# Smoke test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.preprocess import load_from_db
    from src.features import engineer_features
    from src.cluster import run_clustering

    df = load_from_db()
    df = engineer_features(df)
    df = run_clustering(df)

    path = generate_report(df, dataset_name="Kaggle E-Commerce (smoke test)")
    print(f"Report written to: {path}")