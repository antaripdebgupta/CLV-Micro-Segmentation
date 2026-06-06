"""
Thin entry-point wrapper for Streamlit Community Cloud, which expects the
default main file to be named 'streamlit_app.py'.

Locally you can still run:  streamlit run app.py
On Streamlit Cloud this file is picked up automatically.
"""

# Re-export everything from app.py so the behaviour is identical.
from app import *  # noqa: F401,F403
