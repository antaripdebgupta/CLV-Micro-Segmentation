"""
Live log viewer — reads logs/pipeline.log and displays it with
colour-coded severity badges, line-count control, and auto-refresh.
"""

import os
import time
import streamlit as st

st.set_page_config(page_title="Log Viewer", layout="wide")

LOG_FILE    = "logs/pipeline.log"
MAX_LINES   = 500  # never render more than this to keep the page fast

# Severity colour mapping
_LEVEL_COLORS = {
    "DEBUG":    ("#C0C0C0", "#3A3A3A"),   # (bg, text)
    "INFO":     ("#D4EDDA", "#155724"),
    "WARNING":  ("#FFF3CD", "#856404"),
    "ERROR":    ("#F8D7DA", "#721C24"),
    "CRITICAL": ("#F5C6CB", "#491217"),
}
_DEFAULT_COLOR = ("#E9ECEF", "#343A40")


def _read_log(n_lines: int) -> list[str]:
    """Read the last n_lines from the log file. Returns [] if file absent."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-n_lines:]
    except OSError:
        return []


def _parse_level(line: str) -> str:
    """Extract the log level from a line like '2026-06-05 12:00:00 | INFO     | ...'"""
    parts = line.split("|")
    if len(parts) >= 2:
        level = parts[1].strip().upper()
        for key in _LEVEL_COLORS:
            if level.startswith(key):
                return key
    # Fallback: scan for level keyword anywhere in line
    for key in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
        if key in line.upper():
            return key
    return "INFO"


def _render_line_html(line: str) -> str:
    """Wrap a log line in a coloured <div>."""
    level = _parse_level(line)
    bg, fg = _LEVEL_COLORS.get(level, _DEFAULT_COLOR)
    # Escape HTML characters
    safe = (
        line.rstrip()
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<div style="'
        f'background:{bg};color:{fg};'
        f'font-family:monospace;font-size:12px;'
        f'padding:2px 8px;margin:1px 0;'
        f'border-radius:3px;white-space:pre-wrap;word-break:break-all;'
        f'">{safe}</div>'
    )


def _parse_runs(all_lines: list[str]) -> list[dict]:
    """Parse log lines to find training runs. Returns list of dicts with start/end indices."""
    runs = [{"name": "All Runs (Complete History)", "start": 0, "end": len(all_lines)}]
    current_run = None
    for i, line in enumerate(all_lines):
        is_default = "CLV Micro-Segmentation" in line and "Training Pipeline started" in line
        is_custom = "Custom training started" in line
        
        if is_default or is_custom:
            if current_run:
                current_run["end"] = i
            
            if is_default:
                run_name = "Default Kaggle Pipeline Run"
            else:
                ds_name = "unknown.csv"
                if "dataset:" in line:
                    try:
                        ds_name = line.split("dataset:")[1].split(",")[0].strip().replace("'", "").replace('"', "")
                    except Exception:
                        pass
                run_name = f"Custom Train Run ({ds_name})"
                
            timestamp = ""
            if "|" in line:
                timestamp = " - " + line.split("|")[0].strip()
                
            current_run = {
                "name": f"{run_name}{timestamp}",
                "start": i,
                "end": len(all_lines)
            }
            runs.append(current_run)
            
    if len(runs) > 1:
        return [runs[0]] + list(reversed(runs[1:]))
    return runs


# UI

st.title("Pipeline Log Viewer")
st.markdown(
    "Live view of `logs/pipeline.log`. "
    "Select a specific training run or search to find errors/warnings."
)

# Read log
all_lines = []
if os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        pass

# Parse runs from complete history
runs = _parse_runs(all_lines)
run_names = [r["name"] for r in runs]

# Controls
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 2, 2, 1])

with ctrl1:
    selected_run_name = st.selectbox(
        "Select Training Run",
        options=run_names,
        index=1 if len(run_names) > 1 else 0  # Default to latest run if available
    )

with ctrl2:
    level_filter = st.selectbox(
        "Filter by level",
        options=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=0,
    )

with ctrl3:
    n_lines = st.slider(
        "Max lines to display", min_value=20, max_value=MAX_LINES,
        value=100, step=10
    )

with ctrl4:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

# Optional text search filter
search_query = st.text_input("Search/Filter logs by keyword (e.g. CSV filename, error msg)", value="")

# Manual refresh button
if st.button("🔄 Refresh", use_container_width=True):
    st.rerun()

st.markdown("---")

# Extract the selected run slice
selected_run = next(r for r in runs if r["name"] == selected_run_name)
lines = all_lines[selected_run["start"]:selected_run["end"]]

if not lines:
    if not os.path.exists(LOG_FILE):
        st.info(
            f"`{LOG_FILE}` does not exist yet. "
            "Run `python train_pipeline.py` or any other pipeline step to generate logs."
        )
    else:
        st.info("No logs found for the selected run.")
else:
    # Filter by level
    if level_filter != "ALL":
        _order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        min_idx = _order.index(level_filter)
        lines = [
            l for l in lines
            if _parse_level(l) in _order[min_idx:]
        ]

    # Filter by search query
    if search_query:
        lines = [l for l in lines if search_query.lower() in l.lower()]

    # Limit to max lines
    lines = lines[-n_lines:]

    # Summary badges
    level_counts: dict[str, int] = {}
    for line in lines:
        lvl = _parse_level(line)
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    badge_cols = st.columns(5)
    for i, (lvl, (bg, fg)) in enumerate(_LEVEL_COLORS.items()):
        count = level_counts.get(lvl, 0)
        badge_cols[i].markdown(
            f'<div style="background:{bg};color:{fg};'
            f'text-align:center;padding:6px 4px;border-radius:6px;'
            f'font-weight:bold;font-size:13px;">'
            f'{lvl}<br><span style="font-size:20px">{count}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Rendered log lines
    html_lines = "".join(_render_line_html(l) for l in lines)
    st.markdown(
        f'<div style="max-height:600px;overflow-y:auto;'
        f'border:1px solid #dee2e6;border-radius:6px;padding:8px;">'
        f'{html_lines}</div>',
        unsafe_allow_html=True,
    )

    # Download log
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                "Download full log file",
                data=f.read(),
                file_name="pipeline.log",
                mime="text/plain",
            )
    except OSError:
        pass

# File info
if os.path.exists(LOG_FILE):
    size_kb = os.path.getsize(LOG_FILE) / 1024
    mtime   = os.path.getmtime(LOG_FILE)
    import datetime
    modified = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Log file: `{os.path.abspath(LOG_FILE)}` · {size_kb:.1f} KB · last modified {modified}")

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()