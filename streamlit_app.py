"""
Thin entry-point wrapper for Streamlit Community Cloud, which expects the
default main file to be named 'streamlit_app.py'.

Locally you can still run:  streamlit run app.py
On Streamlit Cloud this file is picked up automatically.

We use exec() instead of 'from app import *' so that Streamlit treats this
file as the real entry-point script (correct __file__, page resolution, etc.).
"""

from pathlib import Path

_app_code = (Path(__file__).parent / "app.py").read_text(encoding="utf-8")
exec(compile(_app_code, "app.py", "exec"), globals())
