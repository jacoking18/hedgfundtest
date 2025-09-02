# streamlit_app.py
# ------------------------------------------------------------
# CapNow — Portfolio Syndication Simulator (Model B)
# FULL SYSTEM • Dark‑Blue Futuristic Theme • Role Login (Admin/Investor)
# - Admin: Raise Capital, Participation Queue, Deal Studio, Run Simulation
# - Investor: Participate, My Portfolio (read-only)
# - Model B: 20% early skim on completed deals, 40% total to CapNow / 60% to investors, losses absorbed by portfolio
# - Fees: $500 flat per investor (one-time, billed on top; investable principal unchanged)
# - Celebration: balloons + gloves pop when raise hits $100k
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ===============================
# THEME / STYLES — Dark Blue Futuristic
# ===============================
st.set_page_config(page_title="CapNow – Portfolio Simulator (Model B)", page_icon="💠", layout="wide")

CUSTOM_CSS = """
<style>
:root {
  --bg0: #0a1024;        /* deep blue */
  --bg1: #0f1b3a;        /* mid blue */
  --card: #0f1731;       /* card blue */
  --card2: #111d3d;      /* alt card */
  --accent: #4c7dff;     /* electric blue */
  --accent2: #00d5ff;    /* cyan */
  --text: #e8eeff;       /* near-white text */
  --muted: #a2b1d8;      /* muted text */
  --border: #1f2b55;     /* soft border */
}

/***** animated grid background *****/
[data-testid="stAppViewContainer"] {
  color: var(--text);
  background: radial-gradient(1000px 600px at 15% -10%, #0e1a3b 0%, var(--bg0) 60%);
}

/* moving blocks */
[data-testid="stAppViewContainer"]:before,
[data-testid="stAppViewContainer"]:after {
  content:""; position: fixed; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(0deg, rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size: 48px 48px, 48px 48px;
  animation: drift 40s linear infinite;
}
[data-testid="stAppViewContainer"]:after {
  background-image:
    linear-gradient(90deg, rgba(0,213,255,0.08) 1px, transparent 1px),
    linear-gradient(0deg, rgba(76,125,255,0.08) 1px, transparent 1px);
  mix-blend-mode: screen; opacity:.25; animation-duration: 65s;
}
@keyframes drift {0%{background-position:0 0, 0 0} 100%{background-position:480px 480px, 480px 480px}}

section.main > div {padding-top: .5rem;}
.block-container {padding-top: .75rem;}

/* cards, inputs, tables */
.stMetric, .stDataFrame, .stMarkdown, .stButton>button, .stDownloadButton>button,
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
  border-radius: 14px !important;
}

/* buttons */
.stButton>button, .stDownloadButton>button {background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%); color: #02132b; border: 0; padding: .6rem 1rem; font-weight: 700;}
.stButton>button:hover, .stDownloadButton>button:hover {filter: brightness(1.06);} 

/* tables */
[data-testid="stDataFrame"] {background: var(--card); color: var(--text); border: 1px solid var(--border);} 

/* text */
h1,h2,h3,h4,h5,h6 {color: var(--text) !important;}
p, li, div, label {color: var(--text) !important;}
.small {color: var(--muted) !important; font-size: 0.92rem;}
.badge {display:inline-block; padding: .25rem .6rem; border-radius: 999px; background: rgba(76,125,255,0.18); color: #c9d6ff; font-weight:600}
.kpi, .card {background: var(--card); padding: 1rem; border-radius: 16px; border: 1px solid var(--border)}
.rule {border-top:1px dashed var(--border); margin: .75rem 0}

/* celebration gloves pop */
.pop-wrap {display:flex; gap:12px; align-items:center; margin:.5rem 0 .75rem 0}
.glove {font-size: 2.15rem; display:inline-block; animation: pop 900ms ease-out forwards; opacity:0}
.glove:nth-child(2){animation-delay:120ms}
.glove:nth-child(3){animation-delay:240ms}
@keyframes pop {0%{transform:translateY(12px) scale(.6) rotate(-10deg); opacity:0} 50%{opacity:1} 100%{transform:translateY(-4px) scale(1.05) rotate(0deg); opacity:1}}
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
    FACTOR_FALLBACK: float = 1.40
    BD_PER_YEAR: int = 252
    CAPNOW_EARLY: float = 0.20
    CAPNOW_TOTAL: float = 0.40

DEFAULTS = AppDefaults()

# ===============================
# STATE KEYS & INIT
# ===============================
KEY_AUTH = "auth"
KEY_INVESTORS = "investors"
KEY_REQUESTS = "requests"
KEY_DEALS = "deals"
KEY_PORT = "portfolio"


def init_state():
    if KEY_AUTH not in st.session_state:
        st.session_state[KEY_AUTH] = {"role": None, "user": None}
    if KEY_INVESTORS not in st.session_state:
        st.session_state[KEY_INVESTORS] = []
    if KEY_REQUESTS not in st.session_state:
        st.session_state[KEY_REQUESTS] = []
    if KEY_DEALS not in st.session_state:
        st.session_state[KEY_DEALS] = []
    if KEY_PORT not in st.session_state:
        st.session_state[KEY_PORT] = {"launched": False}

# ===============================
# UTIL — MONEY / PERCENT
# ===============================

def dollars(x: float) -> str:
    try:
        return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"
    except Exception:
        return "$0.00"

def percent(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "0.00%"

# ===============================
# MODEL B — ENGINE
# ===============================
@dataclass
class CycleRow:
    cycle: int
    label: str
    start_principal: float
    deployed: float
    factor: float
    days: int
    completion: float
    profit: float
    capnow_early: float
    reinvested: float
    end_principal: float
    defaulted: bool


def simulate_model_b(start_capital: float, deals: List[Dict], capnow_early_rate: float = DEFAULTS.CAPNOW_EARLY, bd_year: int = DEFAULTS.BD_PER_YEAR):
    S = float(start_capital)
    used_days = 0
    rows: List[CycleRow] = []
    total_profit = 0.0
    total_early = 0.0

    for i, d in enumerate(deals, start=1):
        label = d.get("label", f"Deal {i}")
        factor = float(d.get("factor", DEFAULTS.FACTOR_FALLBACK))
        days = int(d.get("days", 90))
        size_amt = float(d.get("size_amt", S))
        defaulted = bool(d.get("default", False))

        remaining_days = max(bd_year - used_days, 0)
        completion = 0.0 if remaining_days <= 0 else min(1.0, remaining_days / max(days, 1))
        deployed = min(S, size_amt)

        if defaulted:
            P = 0.0; skim = 0.0; reinvested = 0.0; S_end = S - deployed
        else:
            P_full = deployed * (factor - 1.0)
            P = P_full * completion
            skim = capnow_early_rate * P if completion >= 1.0 - 1e-9 else 0.0
            reinvested = P - skim
            S_end = S + reinvested
            total_profit += P; total_early += skim

        rows.append(CycleRow(i, label, S, deployed, factor, days, completion, P, skim, reinvested, S_end, defaulted))
        S = S_end
        used_days += min(days, remaining_days)

    tl = pd.DataFrame([asdict(r) for r in rows])
    return tl, total_profit, total_early


def final_splits(initial_capital: float, total_profit: float, total_early: float, capnow_total_share: float = DEFAULTS.CAPNOW_TOTAL, flat_fees: float = 0.0):
    capnow_total_profit_take = capnow_total_share * total_profit
    capnow_final_component = capnow_total_profit_take - total_early
    investors_total = initial_capital + (1.0 - capnow_total_share) * total_profit
    capnow_total = total_early + capnow_final_component + flat_fees
    return investors_total, capnow_total, capnow_final_component

# ===============================
# CAP TABLE / PAYOUTS
# ===============================

def cap_table_df(investors: List[Dict], target_capital: float) -> pd.DataFrame:
    df = pd.DataFrame(investors)
    if df.empty:
        return pd.DataFrame(columns=["Name", "Email", "Contribution", "% Ownership", "Entry Fee ($500)", "Net Investable"])
    df["% Ownership"] = df["contribution"] / target_capital
    df["Entry Fee ($500)"] = DEFAULTS.FLAT_FEE_PER_INVESTOR
    df["Net Investable"] = df["contribution"]
    df = df.rename(columns={"name": "Name", "email": "Email", "contribution": "Contribution"})[
        ["Name", "Email", "Contribution", "% Ownership", "Entry Fee ($500)", "Net Investable"]
    ]
    return df


def payouts_df(cap_table: pd.DataFrame, investors_total_distribution: float) -> pd.DataFrame:
    if cap_table.empty:
        return pd.DataFrame(columns=["Name", "Email", "Contribution", "% Ownership", "Payout", "ROI %", "Entry Fee ($500)"])
    out = cap_table.copy()
    out["Payout"] = (out["% Ownership"] * investors_total_distribution).round(2)
    out["ROI %"] = ((out["Payout"] - out["Contribution"]) / out["Contribution"]) * 100.0
    return out[["Name", "Email", "Contribution", "% Ownership", "Payout", "ROI %", "Entry Fee ($500)"]]

# ===============================
# AUTH (demo)
# ===============================
ADMIN_USER = "admin@capnow"
ADMIN_PASS = "admin2025"


def login_widget():
    auth = st.session_state[KEY_AUTH]
    if auth["role"]:
        with st.sidebar:
            st.markdown(f"**Logged in as:** `{auth['user']}` · **Role:** `{auth['role']}`")
            if st.button("Log out"):
                st.session_state[KEY_AUTH] = {"role": None, "user": None}
                st.rerun()
        return

    with st.sidebar.expander("Login", expanded=True):
        role = st.selectbox("Role", ["admin", "investor"])
        email = st.text_input("Email / Username")
        password = st.text_input("Password (demo)", type="password")
        if st.button("Sign in"):
            if role == "admin" and email == ADMIN_USER and password == ADMIN_PASS:
                st.session_state[KEY_AUTH] = {"role": "admin", "user": email}
                st.rerun()
            elif role == "investor" and email:
                st.session_state[KEY_AUTH] = {"role": "investor", "user": email}
                st.rerun()
            else:
                st.warning("Invalid credentials (demo admin: admin@capnow / admin2025)")

# celebration helper

def celebrate_reach_100k():
    st.balloons()
    st.markdown('<div class="pop-wrap"><span class="glove">🥊</span><span class="glove">🥊</span><span class="glove">🥊</span></div>', unsafe_allow_html=True)

# ===============================
# ADMIN PAGES
# ===============================

def page_admin():
    st.markdown("## 💠 Admin — Portfolio Studio")

    tabs = st.tabs(["Raise Capital", "Participation Queue", "Deal Studio", "Run Simulation"])  # flow

    # --- Raise Capital ---
    with tabs[0]:
        st.markdown("#### Raise Capital")
        investors = st.session_state[KEY_INVESTORS]
        port = st.session_state[KEY_PORT]

        # Add investor manually
        with st.expander("Add Investor (manual)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Name", key="adm_name")
            with c2:
                email = st.text_input("Email", key="adm_email")
            with c3:
                contribution = st.number_input("Contribution $", min_value=0.0, step=1000.0, key="adm_contrib")
            c4, c5 = st.columns(2)
            with c4:
                auto_trim = st.checkbox("Auto-trim to fill $100k", value=True)
            with c5:
                if st.button("Add / Update"):
                    if not name or contribution < DEFAULTS.MIN_TICKET:
                        st.warning("Name required and minimum ticket is $5,000.")
                    else:
                        existing = next((i for i in investors if i.get("email") == email and email), None)
                        if existing:
                            existing["name"], existing["contribution"] = name, contribution
                        else:
                            total_now = sum(i["contribution"] for i in investors)
                            remaining = DEFAULTS.TARGET_CAPITAL - total_now
                            amt = min(contribution, remaining) if auto_trim else contribution
                            investors.append({"name": name, "email": email, "contribution": float(amt)})
                        st.success("Investor saved.")

        # Cap table & progress
        df = cap_table_df(investors, DEFAULTS.TARGET_CAPITAL)
        total = df["Contribution"].sum() if not df.empty else 0.0
        launched = port.get("launched", False)
        st.progress(min(total / DEFAULTS.TARGET_CAPITAL, 1.0), text=f"{dollars(total)} / {dollars(DEFAULTS.TARGET_CAPITAL)}")

        nice = df.copy()
        if not nice.empty:
            nice["Contribution"], nice["% Ownership"], nice["Entry Fee ($500)"], nice["Net Investable"] = (
                nice["Contribution"].map(dollars), (nice["% Ownership"]*100).map(lambda v: f"{v:.2f}%"),
                nice["Entry Fee ($500)"].map(dollars), nice["Net Investable"].map(dollars)
            )
        st.dataframe(nice, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Investors", len(investors))
        with c2: st.metric("Committed", dollars(total))
        with c3: st.metric("Remaining", dollars(max(DEFAULTS.TARGET_CAPITAL-total, 0)))

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        if not launched:
            if math.isclose(total, DEFAULTS.TARGET_CAPITAL, abs_tol=0.01):
                celebrate_reach_100k()
                if st.button("🚀 Launch Portfolio (lock roster)"):
                    st.session_state[KEY_PORT]["launched"] = True
                    st.success("Portfolio launched.")
            else:
                st.info("Launch is enabled when total committed equals $100,000.")
        else:
            st.success("Portfolio is live. You can still play scenarios in the Deal Studio; distributions are computed at year end.")

    # --- Participation Queue ---
    with tabs[1]:
        st.markdown("#### Participation Queue")
        reqs = st.session_state[KEY_REQUESTS]
        if not reqs:
            st.info("No pending requests. Investors can submit from the Investor portal.")
        else:
            rdf = pd.DataFrame(reqs)
            if not rdf.empty:
                rdf = rdf.rename(columns={"requested_amount": "Requested $"})
                rdf["Requested $"] = rdf["Requested $"] .map(dollars)
            st.dataframe(rdf, use_container_width=True)

            for idx, r in enumerate(reqs):
                colA, colB, colC, colD = st.columns([3,2,1,1])
                with colA: st.write(f"**{r['name']}** · {r.get('email','')} · Request: {dollars(r['requested_amount'])}")
                with colB: trim = st.checkbox(f"Auto-trim to remaining", key=f"trim_{idx}", value=True)
                with colC:
                    if st.button("Approve", key=f"approve_{idx}"):
                        invs = st.session_state[KEY_INVESTORS]
                        total_now = sum(i["contribution"] for i in invs)
                        remaining = DEFAULTS.TARGET_CAPITAL - total_now
                        amt = min(r["requested_amount"], remaining) if trim else r["requested_amount"]
                        invs.append({"name": r["name"], "email": r.get("email",""), "contribution": float(amt)})
                        reqs.pop(idx); st.rerun()
                with colD:
                    if st.button("Reject", key=f"reject_{idx}"):
                        reqs.pop(idx); st.rerun()

    # --- Deal Studio ---
    with tabs[2]:
        st.markdown("#### Deal Studio (build your year)")
        st.caption("Add deals with their own size, factor and duration. Early skim applies on completed deals; defaults are absorbed by portfolio.")

        with st.expander("Quick Builder"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: qb_n = st.number_input("# deals", min_value=1, value=5, step=1)
            with c2: qb_size = st.number_input("deal size $", min_value=0.0, value=10_000.0, step=1_000.0)
            with c3: qb_days = st.number_input("biz days", min_value=1, value=60, step=5)
            with c4: qb_factor = st.number_input("factor", min_value=0.01, value=1.40, step=0.01, format="%.2f")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Set: N × $size"):
                    st.session_state[KEY_DEALS] = [
                        {"label": f"Batch {i+1}", "factor": float(qb_factor), "size_amt": float(qb_size), "days": int(qb_days), "default": False}
                        for i in range(int(qb_n))
                    ]; st.rerun()
            with b2:
                if st.button("Append: N × $size"):
                    cur = st.session_state.get(KEY_DEALS, [])
                    cur += [{"label": f"Batch {len(cur)+i+1}", "factor": float(qb_factor), "size_amt": float(qb_size), "days": int(qb_days), "default": False} for i in range(int(qb_n))]
                    st.session_state[KEY_DEALS] = cur; st.rerun()
            with b3:
                if st.button("Preset: 10×$10k + 2×$50k"):
                    st.session_state[KEY_DEALS] = (
                        [{"label": f"$10k {i+1}", "factor": 1.39, "size_amt": 10_000.0, "days": 60, "default": False} for i in range(10)] +
                        [{"label": f"$50k {i+1}", "factor": 1.40, "size_amt": 50_000.0, "days": 110, "default": False} for i in range(2)]
                    ); st.rerun()

        deals = st.session_state.get(KEY_DEALS, [])
        df = pd.DataFrame(deals)
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, key="deal_editor",
            column_config={
                "label": st.column_config.TextColumn("label"),
                "factor": st.column_config.NumberColumn("factor", step=0.01, format="%.2f"),
                "size_amt": st.column_config.NumberColumn("deal size $", step=1000, format="$%0.0f"),
                "days": st.column_config.NumberColumn("biz days", step=5, min_value=1),
                "default": st.column_config.CheckboxColumn("default"),
            },
        )
        st.session_state[KEY_DEALS] = edited.fillna({"factor": DEFAULTS.FACTOR_FALLBACK, "size_amt": 0, "days": 60, "default": False}).to_dict(orient="records")

    # --- Run Simulation ---
    with tabs[3]:
        st.markdown("#### Run Simulation (Model B)")
        investors = st.session_state[KEY_INVESTORS]
        deals = st.session_state.get(KEY_DEALS, [])
        cap = DEFAULTS.TARGET_CAPITAL
        fees_total = len(investors) * DEFAULTS.FLAT_FEE_PER_INVESTOR
        tl, profit, early = simulate_model_b(cap, deals, DEFAULTS.CAPNOW_EARLY, DEFAULTS.BD_PER_YEAR)
        inv_total, capnow_total, capnow_final = final_splits(cap, profit, early, DEFAULTS.CAPNOW_TOTAL, fees_total)

        if not tl.empty:
            pretty = tl.copy()
            for col in ["start_principal", "deployed", "profit", "capnow_early", "reinvested", "end_principal"]:
                pretty[col] = pretty[col].map(dollars)
            pretty["completion"] = (pretty["completion"]*100).map(lambda v: f"{v:.0f}%")
            st.dataframe(pretty, use_container_width=True)
        else:
            st.info("Add deals in the Deal Studio to simulate.")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Realized Profit", dollars(profit))
        with c2: st.metric("CapNow Early Skims (sum)", dollars(early))
        with c3: st.metric("CapNow Final Component", dollars(capnow_final))

        c4, c5, c6 = st.columns(3)
        with c4: st.metric("Investors' Total Distribution", dollars(inv_total))
        with c5: st.metric("CapNow – All-in (incl. fees)", dollars(capnow_total))
        with c6: st.metric("Fees Collected ($500 × N)", dollars(fees_total))

        cap_df = cap_table_df(investors, cap)
        pay = payouts_df(cap_df, inv_total)
        if not pay.empty:
            nice = pay.copy()
            nice["Contribution"] = nice["Contribution"].map(dollars)
            nice["% Ownership"] = (nice["% Ownership"]*100).map(lambda v: f"{v:.2f}%")
            nice["Payout"] = nice["Payout"].map(dollars)
            nice["Entry Fee ($500)"] = nice["Entry Fee ($500)"].map(dollars)
            nice["ROI %"] = nice["ROI %"].map(lambda v: f"{v:.2f}%")
            st.dataframe(nice, use_container_width=True)
        else:
            st.info("No investors yet.")

# ===============================
# INVESTOR PAGES
# ===============================

def page_investor():
    st.markdown("## 💫 Investor Portal")
    auth = st.session_state[KEY_AUTH]
    email = auth.get("user") or ""

    tabs = st.tabs(["Participate", "My Portfolio"]) 

    with tabs[0]:
        st.markdown("#### Participate in Portfolio Raise")
        st.caption("Target: $100,000 · Min ticket $5,000 · $500 flat entry fee")
        name = st.text_input("Your name", key="req_name")
        req_email = st.text_input("Your email", value=email, key="req_email")
        amt = st.number_input("Request amount $", min_value=0.0, step=1000.0, key="req_amt")
        if st.button("Submit Request"):
            if not name or amt < DEFAULTS.MIN_TICKET:
                st.warning("Please enter your name and request at least $5,000.")
            else:
                st.session_state[KEY_REQUESTS].append({"name": name, "email": req_email, "requested_amount": float(amt)})
                st.success("Request submitted. The admin will review and approve.")

    with tabs[1]:
        st.markdown("#### My Portfolio")
        invs = [i for i in st.session_state[KEY_INVESTORS] if i.get("email") == email]
        if not invs:
            st.info("No approved investment on record for this login.")
            return

        cap_df = cap_table_df(st.session_state[KEY_INVESTORS], DEFAULTS.TARGET_CAPITAL)
        deals = st.session_state.get(KEY_DEALS, [])
        tl, profit, early = simulate_model_b(DEFAULTS.TARGET_CAPITAL, deals, DEFAULTS.CAPNOW_EARLY, DEFAULTS.BD_PER_YEAR)
        inv_total, capnow_total, capnow_final = final_splits(DEFAULTS.TARGET_CAPITAL, profit, early, DEFAULTS.CAPNOW_TOTAL, len(st.session_state[KEY_INVESTORS]) * DEFAULTS.FLAT_FEE_PER_INVESTOR)
        pay = payouts_df(cap_df, inv_total)

        my = pay[pay["Email"] == email]
        if my.empty:
            st.info("You're not in the latest cap table yet. Check back after approval.")
            return

        row = my.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Contribution", dollars(row["Contribution"]))
        with c2: st.metric("Ownership", f"{row['% Ownership']*100:.2f}%")
        with c3: st.metric("Projected Payout", dollars(row["Payout"]))
        with c4: st.metric("ROI %", f"{row['ROI %']:.2f}%")

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown("**Portfolio Performance (Latest Simulation)**")
        if not tl.empty:
            pretty = tl.copy()
            for col in ["start_principal", "deployed", "profit", "capnow_early", "reinvested", "end_principal"]:
                pretty[col] = pretty[col].map(dollars)
            pretty["completion"] = (pretty["completion"]*100).map(lambda v: f"{v:.0f}%")
            st.dataframe(pretty, use_container_width=True)
        else:
            st.info("Admin has not added deals to simulate yet.")

# ===============================
# ROUTER
# ===============================
init_state()
login_widget()
role = st.session_state[KEY_AUTH]["role"]

if role == "admin":
    page_admin()
elif role == "investor":
    page_investor()
else:
    st.markdown("## 💠 CapNow — Portfolio Simulator (Model B)")
    st.write("Use the sidebar to **log in** as admin or investor.")
    st.markdown("- Admin demo: **admin@capnow / admin2025**")
    st.markdown("- Investor: enter any email (no password in demo)")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("**Highlights**")
    st.markdown("- Target: $100,000 · Min ticket $5,000 · $500 entry fee per investor")
    st.markdown("- Profit split: 40% CapNow / 60% Investors · Early skim 20% per completed deal")
    st.markdown("- Portfolio absorbs losses on defaults")
