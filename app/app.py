"""WingCheck entrypoint and page router.

Streamlit Cloud deploys this file (app/app.py), so the deployment setting does
not change. All this file does is make `src/` importable and register the three
pages with Streamlit's native multipage API (st.navigation / st.Page).

  - home.py           beginner-friendly landing page (default)
  - wing_lab.py       Wing Lab guided challenge
  - advanced_mode.py  the original WingCheck toolkit, moved here unchanged

Model loading and prediction logic stay in src/inference.py, and every page
calls it.
"""

import sys
from pathlib import Path

import streamlit as st

# Same root discovery as the original app.py, so `import src.inference` works
# for every page whether Streamlit is launched from the repo root or from app/.
CURRENT_FILE = Path(__file__).resolve()
CANDIDATE_ROOTS = [CURRENT_FILE.parent, CURRENT_FILE.parent.parent]
PROJECT_ROOT = next((root for root in CANDIDATE_ROOTS if (root / "src").exists()), CURRENT_FILE.parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

APP_DIR = CURRENT_FILE.parent
if str(APP_DIR) not in sys.path:  # lets wing_lab.py import wing_lab_core
    sys.path.append(str(APP_DIR))

st.set_page_config(page_title="WingCheck", page_icon="🛩️", layout="wide")

home_page = st.Page("home.py", title="Home", icon="🏠", default=True)
wing_lab_page = st.Page("wing_lab.py", title="Wing Lab", icon="🐋")
advanced_page = st.Page("advanced_mode.py", title="Advanced Mode", icon="🛠️")

nav = st.navigation([home_page, wing_lab_page, advanced_page])
nav.run()
