"""
Validates a DataFrame before it enters the training pipeline.
Returns a structured QualityReport dict so callers (Streamlit pages,
train_pipeline.py) can decide whether to abort or warn.

"""

import os

import pandas as pd
import numpy as np
from typing import Any

try:
    from src.logger import get_logger
except ModuleNotFoundError:
    # When running this file directly as a script from the project root,
    # Python sets sys.path[0] to the src directory, so pkg-style import fails.
    from logger import get_logger

log = get_logger(__name__)

# Columns the pipeline absolutely needs (at minimum)
REQUIRED_COLS = [
    "Age",
    "Total Spend",
    "Items Purchased",
    "Days Since Last Purchase",
    "Average Rating",
]

# Columns that are nice-to-have; a warning is raised if absent
OPTIONAL_COLS = [
    "Gender",
    "Membership Type",
    "Satisfaction Level",
    "Discount Applied",
    "City",
]

MIN_ROWS = 50  # fewer rows → model will be unreliable


def _missing_summary(df: pd.DataFrame) -> dict[str, dict]:
    """Return per-column missing value counts and percentages."""
    total = len(df)
    result = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            result[col] = {
                "count": n_missing,
                "pct": round(100 * n_missing / total, 2),
            }
    return result


def _zero_variance_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns where all non-null values are identical."""
    zv = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dropna().nunique() <= 1:
            zv.append(col)
    return zv


def _dtype_issues(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, str]:
    """Return columns that should be numeric but contain non-numeric strings."""
    issues = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(coerced.isnull().sum() - df[col].isnull().sum())
        if n_bad > 0:
            issues[col] = f"{n_bad} non-numeric values (will be coerced to NaN)"
    return issues


def check_quality(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
    optional_cols: list[str] | None = None,
    min_rows: int = MIN_ROWS,
) -> dict[str, Any]:
    """
    Run a full quality check on a DataFrame.

    Args:
        df:            Input DataFrame (raw, before any cleaning).
        required_cols: Columns that must be present. Defaults to REQUIRED_COLS.
        optional_cols: Columns that are nice-to-have. Defaults to OPTIONAL_COLS.
        min_rows:      Minimum acceptable row count.

    Returns:
        QualityReport dict with keys:
            can_train   (bool)   — False if any hard error exists
            errors      (list)   — blocking issues
            warnings    (list)   — non-blocking issues
            stats       (dict)   — row count, column count, missing summary, etc.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLS
    if optional_cols is None:
        optional_cols = OPTIONAL_COLS

    errors:   list[str] = []
    warnings: list[str] = []

    # 1. Basic shape
    n_rows, n_cols = df.shape
    log.info("Quality check — shape: %d rows × %d columns", n_rows, n_cols)

    if n_rows == 0:
        errors.append("DataFrame is empty (0 rows).")
        return {
            "can_train": False,
            "errors": errors,
            "warnings": warnings,
            "stats": {"rows": 0, "cols": n_cols},
        }

    if n_rows < min_rows:
        warnings.append(
            f"Only {n_rows} rows — model reliability may be low (recommended ≥ {min_rows})."
        )

    # 2. Duplicate rows
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate rows detected (will be dropped during preprocessing).")

    # 3. Required columns
    # Normalise column names for comparison (strip + title-case)
    actual_cols = [c.strip().title().replace("  ", " ") for c in df.columns]
    df_norm = df.copy()
    df_norm.columns = actual_cols

    missing_required = [c for c in required_cols if c not in actual_cols]
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")
        log.error("Missing required columns: %s", missing_required)

    missing_optional = [c for c in optional_cols if c not in actual_cols]
    if missing_optional:
        warnings.append(
            f"Optional columns absent (features will be imputed or skipped): {missing_optional}"
        )

    # 4. Missing values
    missing = _missing_summary(df_norm)
    high_missing = {col: v for col, v in missing.items() if v["pct"] > 50}
    if high_missing:
        for col, v in high_missing.items():
            warnings.append(
                f"Column '{col}' has {v['pct']}% missing values — imputation may distort results."
            )

    all_null_cols = [col for col in df_norm.columns if df_norm[col].isnull().all()]
    if all_null_cols:
        errors.append(f"Columns with 100% null values (unusable): {all_null_cols}")

    # 5. Zero-variance columns
    zv_cols = _zero_variance_cols(df_norm)
    present_zv = [c for c in zv_cols if c in required_cols + optional_cols]
    if present_zv:
        warnings.append(
            f"Zero-variance columns (all identical values): {present_zv}. "
            "These add no signal to the model."
        )

    # 6. Dtype issues
    dtype_issues = _dtype_issues(df_norm, required_cols)
    if dtype_issues:
        for col, msg in dtype_issues.items():
            warnings.append(f"Column '{col}': {msg}")

    # 7. Negative values in non-negative columns
    non_negative = ["Total Spend", "Items Purchased", "Days Since Last Purchase", "Age"]
    for col in non_negative:
        if col in df_norm.columns:
            try:
                neg_count = int((pd.to_numeric(df_norm[col], errors="coerce") < 0).sum())
                if neg_count > 0:
                    warnings.append(
                        f"Column '{col}' has {neg_count} negative values — will be clipped to 0."
                    )
            except Exception:
                pass

    # 8. Rating range check
    if "Average Rating" in df_norm.columns:
        try:
            rating_series = pd.to_numeric(df_norm["Average Rating"], errors="coerce").dropna()
            if not rating_series.empty:
                if rating_series.max() > 5:
                    warnings.append(
                        f"'Average Rating' max = {rating_series.max():.1f} — expected ≤ 5. "
                        "Values will be clipped."
                    )
                if rating_series.min() < 1:
                    warnings.append(
                        f"'Average Rating' min = {rating_series.min():.1f} — expected ≥ 1."
                    )
        except Exception:
            pass

    # 9. Summary
    can_train = len(errors) == 0

    stats = {
        "rows": n_rows,
        "cols": n_cols,
        "duplicates": n_dupes,
        "missing_summary": missing,
        "zero_variance_cols": zv_cols,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }

    if can_train:
        log.info("Quality check PASSED — %d error(s), %d warning(s)", 0, len(warnings))
    else:
        log.error(
            "Quality check FAILED — %d error(s): %s", len(errors), errors
        )

    for w in warnings:
        log.warning(w)

    return {
        "can_train": can_train,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def render_quality_report(report: dict[str, Any]) -> None:
    """
    Render the quality report inside a Streamlit app.
    Call this only from within a Streamlit page.
    """
    import streamlit as st

    stats = report["stats"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{stats['rows']:,}")
    col2.metric("Columns", stats["cols"])
    col3.metric("Duplicates", stats.get("duplicates", "—"))
    missing_cols = len(stats.get("missing_summary", {}))
    col4.metric("Cols w/ Missing", missing_cols)

    if report["errors"]:
        for err in report["errors"]:
            st.error(f"{err}")

    if report["warnings"]:
        with st.expander(f"{len(report['warnings'])} Warning(s)", expanded=False):
            for w in report["warnings"]:
                st.warning(w)

    if report["can_train"] and not report["warnings"]:
        st.success("Data quality check passed — ready to train.")
    elif report["can_train"]:
        st.info("Data quality check passed with warnings — training will proceed.")

    # Missing value table
    missing = stats.get("missing_summary", {})
    if missing:
        import pandas as pd
        st.markdown("**Missing Value Summary**")
        mv_df = pd.DataFrame(missing).T.reset_index()
        mv_df.columns = ["Column", "Missing Count", "Missing %"]
        st.dataframe(mv_df, use_container_width=True, hide_index=True)


# 10. Smoke test
if __name__ == "__main__":

    RAW = "data/raw/ecommerce_customer_data.csv"
    if not os.path.exists(RAW):
        print(f"Raw CSV not found at {RAW}. Place the Kaggle CSV there first.")
        sys.exit(1)

    df = pd.read_csv(RAW)
    report = check_quality(df)

    print("\n=== Quality Report ===")
    print(f"Can train : {report['can_train']}")
    print(f"Errors    : {report['errors']}")
    print(f"Warnings  : {report['warnings']}")
    print(f"Stats     : rows={report['stats']['rows']}, cols={report['stats']['cols']}")