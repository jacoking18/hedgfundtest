# streamlit_app.py
# ------------------------------------------------------------
# CapNow — Portfolio Syndication Simulator (Model B)
# Updated Theme: Modern blue background with white text
# Added: Fun glove animation (popping up) when raise hits $100k
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

# ===============================
# THEME / STYLES
# ===============================
st.set_page_config(page_title="CapNow – Portfolio Simulator (Model B)", page_icon="💠", layout="wide")

CUSTOM_CSS = """
<style>
:root {
  --bg: #f2f7ff;
  --card: #ffffff;
  --accent: #005bea;
  --accent2: #00c6fb;
  --text: #0b1e3f;
  --muted: #6b7a90;
}

[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #f2f7ff 0%, #dce9ff 100%);} 
section.main > div {padding-top: 1rem;}

.block-container {padding-top: 1.5rem;}
.stMetric, .stDataFrame, .stMarkdown, .stButton > button, .stDownloadButton>button, .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
  border-radius: 14px !important; 
}

.stButton>button, .stDownloadButton>button {background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%); color: #fff; border: 0; padding: .6rem 1rem; font-weight: 600;}
.stButton>button:hover, .stDownloadButton>button:hover {filter: brightness(1.08);} 

[data-testid="stDataFrame"] {background: var(--card); color: var(--text);} 

h1,h2,h3,h4,h5,h6, p, li, div {color: var(--text) !important;}
.small {color: var(--muted) !important; font-size: 0.9rem;}
.badge {display:inline-block; padding: .25rem .6rem; border-radius: 999px; background: rgba(0,91,234,0.15); color: var(--accent); font-weight:600}
.kpi {background: var(--card); padding: .8rem 1rem; border-radius: 16px;}
.card {background: var(--card); padding: 1rem; border-radius: 16px; border: 1px solid rgba(0,0,0,0.05)}
.rule {border-top:1px dashed rgba(0,0,0,0.15); margin: .75rem 0}

.glove {position: fixed; bottom: 20%; left: 50%; transform: translateX(-50%); font-size: 4rem; animation: pop 1.5s ease-out forwards;}
@keyframes pop {
  0% {transform: translateX(-50%) scale(0); opacity: 0;}
  50% {transform: translateX(-50%) scale(1.2); opacity: 1;}
  100% {transform: translateX(-50%) scale(1); opacity: 0;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# CONFIG & CONSTANTS
# ===============================
@dataclass(frozen=True)
class AppDefaults:
    TARGET_CAPITAL: float = 100_000.0
    MIN_TICKET: float = 5_000.0
    FLAT_FEE_PER_INVESTOR: float = 500.0

DEFAULTS = AppDefaults()

# ===============================
# STATE INIT
# ===============================
if "committed" not in st.session_state:
    st.session_state["committed"] = 0.0

# ===============================
# DEMO CAPITAL RAISE
# ===============================
st.markdown("## Admin — Portfolio Studio")
st.markdown("### Raise Capital")

investors = st.session_state.get("investors", [])
name = st.text_input("Investor name")
amount = st.number_input("Contribution $", step=1000.0)
if st.button("Add Investor"):
    if amount >= DEFAULTS.MIN_TICKET:
        investors.append({"name": name, "contribution": amount})
        st.session_state["investors"] = investors
        st.session_state["committed"] = sum(i["contribution"] for i in investors)
    else:
        st.warning("Minimum is $5,000")

committed = st.session_state["committed"]
remaining = DEFAULTS.TARGET_CAPITAL - committed

st.progress(min(committed / DEFAULTS.TARGET_CAPITAL, 1.0), text=f"${committed:,.0f} / ${DEFAULTS.TARGET_CAPITAL:,.0f}")

st.write(f"**Committed:** ${committed:,.0f}")
st.write(f"**Remaining:** ${remaining:,.0f}")

# Trigger glove animation when goal is hit
if math.isclose(committed, DEFAULTS.TARGET_CAPITAL, abs_tol=0.01):
    html('<div class="glove">🧤🎉</div>', height=150)
    st.success("Portfolio goal reached: $100,000! Launch ready 🚀")
