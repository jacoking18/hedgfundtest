# streamlit_app.py
# ------------------------------------------------------------
# CapNow — Portfolio Syndication Simulator (Model B)
# FINTECH-STYLE UI (2025) • Light Grey Theme • ZERO logic changes
# - Same business logic & behavior; only visuals/components improved.
# - Adds: Ledger start date picker + date jump (US business days)
# - Adds: Supabase persistence (investors, requests, ledger_deals)
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import date

import pandas as pd
import streamlit as st

# ===============================
# THEME / STYLES — Light Grey Fintech
# ===============================
st.set_page_config(page_title="CapNow – Portfolio Simulator (Model B)", page_icon="💠", layout="wide")

LIGHT_GREY_CSS = """
<style>
:root{
  --bg0:#f6f7fb; --bg1:#ffffff; --grid:#e6e9ef; --card:#ffffff; --card2:#f0f2f7;
  --accent:#3a7bd5; --accent2:#00c6ff; --accent3:#82ffd2; --warn:#ffc658; --error:#ff6b6b;
  --txt:#0f172a; --muted:#6b7280; --border:#d7deea;
}

/* background */
[data-testid="stAppViewContainer"]{color:var(--txt);background:var(--bg0);} 

/* layout */
.block-container{padding-top:.6rem}
section.main > div{padding-top:.25rem}

/* cards + inputs */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem}
.card.alt{background:var(--card2)}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem}
.rule{border-top:1px dashed var(--border);margin:.75rem 0}
.badge{display:inline-flex;gap:.4rem;align-items:center;background:rgba(58,123,213,.10);color:var(--txt);border:1px solid rgba(58,123,213,.28);padding:.22rem .6rem;border-radius:999px;font-weight:600}

.stButton>button,.stDownloadButton>button{border:0;border-radius:8px;padding:.55rem 1rem;font-weight:600;color:#fff;background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);} 
.stButton>button:hover{filter:brightness(1.06)}

/* table container */
[data-testid="stDataFrame"]{background:var(--card);border:1px solid var(--border);color:var(--txt)}

/* sidebar login tiles */
.login-tile{display:block;text-align:center;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.7rem .6rem;transition:transform .08s ease, filter .2s ease}
.login-tile:hover{transform:translateY(-2px);filter:brightness(1.04)}
.login-tile .em{display:block;font-size:1rem;font-weight:700;color:var(--txt)}
.login-tile .sm{display:block;color:var(--muted);font-size:.8rem}

/* celebration */
.pop-wrap{display:flex;gap:10px;align-items:center;margin:.3rem 0 .6rem 0}
.glove{font-size:1.9rem;opacity:0;animation:pop .9s ease-out forwards}
.glove:nth-child(2){animation-delay:120ms}.glove:nth-child(3){animation-delay:240ms}
@keyframes pop{0%{transform:translateY(10px) scale(.6) rotate(-10deg);opacity:0}50%{opacity:1}100%{transform:translateY(-3px) scale(1.03) rotate(0)} }
</style>
"""
st.markdown(LIGHT_GREY_CSS, unsafe_allow_html=True)

# ===============================
# CONFIG & CONSTANTS (unchanged)
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
# STATE (unchanged business logic; adds start_date only)
# ===============================
KEY_AUTH = "auth"
KEY_INVESTORS = "investors"
KEY_REQUESTS = "requests"
KEY_DEALS = "deals"              # discrete studio list
KEY_PORT = "portfolio"
# Continuous ledger state
KEY_LEDGER = "ledger_deals"       # list of deals in ledger
KEY_DAY = "ledger_day"            # current business day index (0..252)
KEY_CASH = "ledger_cash"          # available cash
KEY_SKIM = "ledger_early_skim"    # capnow early skim accumulated
# New (UI only): start date to map business days → calendar dates
KEY_STARTDATE = "ledger_start_date"


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
    if KEY_LEDGER not in st.session_state:
        st.session_state[KEY_LEDGER] = []
    if KEY_DAY not in st.session_state:
        st.session_state[KEY_DAY] = 0
    if KEY_CASH not in st.session_state:
        st.session_state[KEY_CASH] = 0.0
    if KEY_SKIM not in st.session_state:
        st.session_state[KEY_SKIM] = 0.0
    if KEY_STARTDATE not in st.session_state:
        st.session_state[KEY_STARTDATE] = date.today()

# ===============================
# UTIL — MONEY / PERCENT (unchanged)
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
# DATE MAPPING (UI helper; does not affect ledger math)
# ===============================
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_CB = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def bizday_to_date(start: date, day_index: int) -> date:
    try:
        ts = pd.Timestamp(start) + (day_index * _CB)
        return ts.date()
    except Exception:
        return start

def date_to_biz_index(start: date, target: date) -> int:
    try:
        start_ts = pd.Timestamp(start)
        target_ts = pd.Timestamp(target)
        if target_ts < start_ts:
            return 0
        rng = pd.date_range(start=start_ts, end=target_ts, freq=_CB)
        # day 0 = start date
        return max(0, len(rng) - 1)
    except Exception:
        return 0

# ===============================
# MODEL B — DISCRETE ENGINE (unchanged)
# ===============================
from dataclasses import dataclass as _dataclass
@_dataclass
class CycleRow:
    cycle: int; label: str; start_principal: float; deployed: float; factor: float; days: int
    completion: float; profit: float; capnow_early: float; reinvested: float; end_principal: float; defaulted: bool


def simulate_model_b(start_capital: float, deals: List[Dict], capnow_early_rate: float = DEFAULTS.CAPNOW_EARLY, bd_year: int = DEFAULTS.BD_PER_YEAR):
    S = float(start_capital); used_days = 0; rows: List[CycleRow] = []; total_profit = 0.0; total_early = 0.0
    for i, d in enumerate(deals, start=1):
        label = d.get("label", f"Deal {i}"); factor = float(d.get("factor", DEFAULTS.FACTOR_FALLBACK)); days = int(d.get("days", 90))
        size_amt = float(d.get("size_amt", S)); defaulted = bool(d.get("default", False))
        remaining_days = max(bd_year - used_days, 0); completion = 0.0 if remaining_days <= 0 else min(1.0, remaining_days / max(days, 1))
        deployed = min(S, size_amt)
        if defaulted:
            P = 0.0; skim = 0.0; reinvested = 0.0; S_end = S - deployed
        else:
            P_full = deployed * (factor - 1.0); P = P_full * completion; skim = capnow_early_rate * P if completion >= 1.0 - 1e-9 else 0.0
            reinvested = P - skim; S_end = S + reinvested; total_profit += P; total_early += skim
        rows.append(CycleRow(i, label, S, deployed, factor, days, completion, P, skim, reinvested, S_end, defaulted))
        S = S_end; used_days += min(days, remaining_days)
    tl = pd.DataFrame([asdict(r) for r in rows]); return tl, total_profit, total_early


def final_splits(initial_capital: float, total_profit: float, total_early: float, capnow_total_share: float = DEFAULTS.CAPNOW_TOTAL, flat_fees: float = 0.0):
    capnow_total_profit_take = capnow_total_share * total_profit
    capnow_final_component = capnow_total_profit_take - total_early
    investors_total = initial_capital + (1.0 - capnow_total_share) * total_profit
    capnow_total = total_early + capnow_final_component + flat_fees
    return investors_total, capnow_total, capnow_final_component

# ===============================
# CONTINUOUS CASH FLOW LEDGER (unchanged logic)
# ===============================

def ledger_add_deal(label: str, amount: float, factor: float, term_days: int, start_day: int, default: bool = False) -> bool:
    cash = st.session_state[KEY_CASH]
    if amount > cash:
        st.warning("Not enough available cash to start this deal.")
        return False
    end_day = start_day + term_days; daily = amount * factor / term_days; gross = amount * factor
    deal = {"id": len(st.session_state[KEY_LEDGER]) + 1, "label": label or f"Deal {len(st.session_state[KEY_LEDGER]) + 1}",
            "amount": float(amount), "factor": float(factor), "term": int(term_days), "start_day": int(start_day),
            "end_day": int(end_day), "daily": float(daily), "gross": float(gross), "collected": 0.0,
            "completed": False, "default": bool(default)}
    st.session_state[KEY_LEDGER].append(deal); st.session_state[KEY_CASH] = cash - amount; return True


def ledger_advance_to_day(target_day: int):
    day = st.session_state[KEY_DAY]
    if target_day < day:
        st.info("You moved backward; no changes applied."); st.session_state[KEY_DAY] = target_day; return
    cash = st.session_state[KEY_CASH]; skim = st.session_state[KEY_SKIM]
    for d in range(day + 1, target_day + 1):
        for deal in st.session_state[KEY_LEDGER]:
            if deal["completed"]: continue
            if deal["default"]:
                if d == max(day + 1, deal["start_day"]): deal["completed"] = True
                continue
            if d > deal["start_day"] and d <= deal["end_day"]:
                to_add = deal["daily"];  # drip in
                if deal["collected"] + to_add > deal["gross"]: to_add = deal["gross"] - deal["collected"]
                cash += to_add; deal["collected"] += to_add
            if d >= deal["end_day"] and not deal["completed"]:
                profit = max(0.0, deal["gross"] - deal["amount"]); early = DEFAULTS.CAPNOW_EARLY * profit
                cash -= early; skim += early; deal["completed"] = True
    st.session_state[KEY_CASH] = cash; st.session_state[KEY_SKIM] = skim; st.session_state[KEY_DAY] = target_day


def ledger_dataframe() -> pd.DataFrame:
    if not st.session_state[KEY_LEDGER]:
        return pd.DataFrame(columns=["id","label","amount","factor","term","start_day","end_day","daily","gross","collected","days_left","completed","default"])
    df = pd.DataFrame(st.session_state[KEY_LEDGER]); df["days_left"] = (df["end_day"] - st.session_state[KEY_DAY]).clip(lower=0); return df


def ledger_realized_profit() -> float:
    prof = 0.0
    for deal in st.session_state[KEY_LEDGER]:
        collected = deal["collected"]; principal_part = min(collected, deal["amount"]); profit_part = max(0.0, collected - principal_part)
        prof += profit_part
    return prof


def ledger_outstanding_principal() -> float:
    out = 0.0
    for deal in st.session_state[KEY_LEDGER]:
        principal_repaid = min(deal["collected"], deal["amount"])
        out += max(0.0, deal["amount"] - principal_repaid)
    return out

# NEW (UI metric only; logic neutral): remaining (principal+profit) not yet received

def ledger_to_be_collected() -> float:
    rem = 0.0
    for deal in st.session_state[KEY_LEDGER]:
        if deal.get("default", False):
            continue
        rem += max(0.0, deal.get("gross", 0.0) - deal.get("collected", 0.0))
    return rem

# ===============================
# CAP TABLE / PAYOUTS (unchanged)
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
# AUTH — Click-to-enter tiles (no passwords)
# ===============================

def login_widget():
    auth = st.session_state[KEY_AUTH]
    if auth["role"]:
        with st.sidebar:
            st.markdown(f"**Logged in as:** `{auth['user'] or auth['role']}` · **Role:** `{auth['role']}`")
            if st.button("Log out"):
                st.session_state[KEY_AUTH] = {"role": None, "user": None}; st.rerun()
        return
    with st.sidebar.expander("Enter", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("↪ Admin", use_container_width=True):
                st.session_state[KEY_AUTH] = {"role": "admin", "user": "admin"}; st.rerun()
        with c2:
            if st.button("↪ Investor", use_container_width=True):
                st.session_state[KEY_AUTH] = {"role": "investor", "user": "guest@investor"}; st.rerun()

# Celebration (unchanged)

def celebrate_reach_100k():
    st.balloons()
    st.markdown('<div class="pop-wrap"><span class="glove">🥊</span><span class="glove">🥊</span><span class="glove">🥊</span></div>', unsafe_allow_html=True)

# ===============================
# SUPABASE (persistence) — additive, no logic change
# ===============================
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

@st.cache_resource
def sb_admin() -> Client | None:
    try:
        if create_client is None:
            return None
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None

SUPA_ENABLED = sb_admin() is not None

# Writers (safe no-ops if Supabase not configured)

def upsert_investors_list(investors: List[Dict]) -> None:
    if not SUPA_ENABLED or not investors:
        return
    payload = []
    for i in investors:
        payload.append({
            "name": i.get("name", ""),
            "email": i.get("email", ""),
            "contribution": float(i.get("contribution", 0) or 0),
        })
    try:
        sb_admin().table("investors").upsert(payload, on_conflict="email").execute()
    except Exception:
        pass


def insert_request_row(name: str, email: str, amount: float) -> None:
    if not SUPA_ENABLED:
        return
    try:
        sb_admin().table("requests").insert({
            "name": name,
            "email": email,
            "requested_amount": float(amount or 0),
        }).execute()
    except Exception:
        pass


def insert_ledger_deal_row(deal: Dict) -> None:
    if not SUPA_ENABLED or not deal:
        return
    try:
        sb_admin().table("ledger_deals").insert({
            "label": deal.get("label", "Deal"),
            "amount": float(deal.get("amount", 0) or 0),
            "factor": float(deal.get("factor", 1.0) or 1.0),
            "term": int(deal.get("term", 0) or 0),
            "start_day": int(deal.get("start_day", 0) or 0),
            "end_day": int(deal.get("end_day", 0) or 0),
            "collected": float(deal.get("collected", 0) or 0),
            "completed": bool(deal.get("completed", False)),
            "defaulted": bool(deal.get("default", False)),
        }).execute()
    except Exception:
        pass


def save_portfolio_run(summary, deals, payouts, capnow_earnings):
    if not SUPA_ENABLED:
        st.warning("Supabase is not enabled.")
        return
    try:
        sb_admin().table("portfolio_runs").insert({
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": summary,             # dict
            "deals": deals,                 # list of dicts
            "payouts": payouts,             # list of dicts
            "capnow_earnings": float(capnow_earnings),
        }).execute()
        st.success("Portfolio run saved to Supabase!")
    except Exception as e:
        st.warning(f"Failed to save run: {e}")


# ===============================
# ADMIN PAGES (UI polish only)
# ===============================

def page_admin():
    st.markdown("## 💠 Admin — Portfolio Studio")
    tabs = st.tabs(["Raise Capital", "Participation Queue", "Deal Studio", "Ledger (continuous)", "Run / Finalize"])  # flow

    # --- Raise Capital ---
    with tabs[0]:
        st.markdown("### 🧩 Raise Capital")
        investors = st.session_state[KEY_INVESTORS]; port = st.session_state[KEY_PORT]
        with st.expander("Add Investor (manual)"):
            c1, c2, c3 = st.columns(3)
            with c1: name = st.text_input("Name", key="adm_name")
            with c2: email = st.text_input("Email", key="adm_email")
            with c3: contribution = st.number_input("Contribution $", min_value=0.0, step=1000.0, key="adm_contrib")
            c4, c5 = st.columns(2)
            with c4: auto_trim = st.checkbox("Auto-trim to fill $100k", value=True)
            with c5:
                if st.button("Add / Update", use_container_width=True):
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
                        # NEW: upsert to Supabase (no logic change)
                        upsert_investors_list(st.session_state[KEY_INVESTORS])

        df = cap_table_df(investors, DEFAULTS.TARGET_CAPITAL)
        total = df["Contribution"].sum() if not df.empty else 0.0
        launched = port.get("launched", False)
        st.progress(min(total / DEFAULTS.TARGET_CAPITAL, 1.0), text=f"{dollars(total)} / {dollars(DEFAULTS.TARGET_CAPITAL)}")

        nice = df.copy()
        if not nice.empty:
            nice["Contribution"] = nice["Contribution"].map(dollars)
            nice["% Ownership"] = (nice["% Ownership"]*100).map(lambda v: f"{v:.2f}%")
            nice["Entry Fee ($500)"] = nice["Entry Fee ($500)"].map(dollars)
            nice["Net Investable"] = nice["Net Investable"].map(dollars)
        st.dataframe(nice, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Investors", len(investors))
        with c2: st.metric("Committed", dollars(total))
        with c3: st.metric("Remaining", dollars(max(DEFAULTS.TARGET_CAPITAL-total, 0)))

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        if not launched:
            if math.isclose(total, DEFAULTS.TARGET_CAPITAL, abs_tol=0.01):
                celebrate_reach_100k()
                if st.button("🚀 Launch Portfolio (lock roster)", use_container_width=True):
                    st.session_state[KEY_PORT]["launched"] = True
                    st.session_state[KEY_CASH] = DEFAULTS.TARGET_CAPITAL
                    st.success("Portfolio launched. Ledger initialized with $100,000 available.")
            else:
                st.info("Launch is enabled when total committed equals $100,000.")
        else:
            st.success("Portfolio is live. Use the Ledger tab to deploy & advance time.")

    # --- Participation Queue ---
    with tabs[1]:
        st.markdown("### 📨 Participation Queue")
        reqs = st.session_state[KEY_REQUESTS]
        if not reqs:
            st.info("No pending requests. Investors can submit from the Investor portal.")
        else:
            rdf = pd.DataFrame(reqs)
            if not rdf.empty:
                rdf = rdf.rename(columns={"requested_amount": "Requested $"}); rdf["Requested $"] = rdf["Requested $"] .map(dollars)
            st.dataframe(rdf, use_container_width=True)
            for idx, r in enumerate(list(reqs)):
                colA, colB, colC, colD = st.columns([3,2,1,1])
                with colA: st.write(f"**{r['name']}** · {r.get('email','')} · Request: {dollars(r['requested_amount'])}")
                with colB: trim = st.checkbox("Auto-trim to remaining", key=f"trim_{idx}", value=True)
                with colC:
                    if st.button("Approve", key=f"approve_{idx}"):
                        invs = st.session_state[KEY_INVESTORS]
                        total_now = sum(i["contribution"] for i in invs); remaining = DEFAULTS.TARGET_CAPITAL - total_now
                        amt = min(r["requested_amount"], remaining) if trim else r["requested_amount"]
                        invs.append({"name": r["name"], "email": r.get("email",""), "contribution": float(amt)})
                        reqs.remove(r); st.rerun()
                with colD:
                    if st.button("Reject", key=f"reject_{idx}"):
                        reqs.remove(r); st.rerun()

    # --- Deal Studio (discrete, optional) ---
    with tabs[2]:
        st.markdown("### 🧪 Deal Studio (discrete sandbox)")
        st.caption("Build a list of deals and simulate sequentially. Use Ledger for daily cashflow.")
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

    # --- Ledger (continuous) ---
    with tabs[3]:
        st.markdown("### 💹 Ledger — Continuous Cash Flow (daily collections)")
        port = st.session_state[KEY_PORT]
        if not port.get("launched", False):
            st.info("Launch the portfolio in Raise Capital to initialize $100,000 cash.")
        c0, c1, c2, c3, c4, c5 = st.columns(6)
        with c0:
            # NEW: start date picker (UI only)
            sd = st.date_input("Start date", value=st.session_state[KEY_STARTDATE], help="Used to map business days to calendar (US holidays excluded)")
            if sd != st.session_state[KEY_STARTDATE]:
                st.session_state[KEY_STARTDATE] = sd
        with c1: st.metric("Day", st.session_state[KEY_DAY])
        with c2: st.metric("Simulated Date", bizday_to_date(st.session_state[KEY_STARTDATE], st.session_state[KEY_DAY]).strftime("%b %d, %Y"))
        with c3: st.metric("Available Cash", dollars(st.session_state[KEY_CASH]))
        with c4: st.metric("Outstanding Principal", dollars(ledger_outstanding_principal()))
        with c5: st.metric("CapNow Early Skims", dollars(st.session_state[KEY_SKIM]))

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        # Controls: advance day & add deal
        colA, colB = st.columns([2,3])
        with colA:
            target = st.slider("Advance to business day", min_value=0, max_value=DEFAULTS.BD_PER_YEAR, value=st.session_state[KEY_DAY], step=1)
            if st.button("Advance"):
                ledger_advance_to_day(target)
            # NEW: jump by calendar date (does not alter math; just computes index)
            jump_date = st.date_input("Jump to date", value=bizday_to_date(st.session_state[KEY_STARTDATE], st.session_state[KEY_DAY]))
            if st.button("Go to date"):
                idx = date_to_biz_index(st.session_state[KEY_STARTDATE], jump_date)
                idx = min(max(0, idx), DEFAULTS.BD_PER_YEAR)
                ledger_advance_to_day(idx)
        with colB:
            st.markdown("**Start New Deal (at current day)**")
            d1, d2, d3, d4, d5 = st.columns(5)
            with d1: label = st.text_input("label", value=f"Deal {len(st.session_state[KEY_LEDGER])+1}")
            with d2: amt = st.number_input("amount $", min_value=0.0, value=10_000.0, step=1000.0)
            with d3: factor = st.number_input("factor", min_value=1.0, value=1.40, step=0.01, format="%.2f")
            with d4: term = st.number_input("term (biz days)", min_value=1, value=90, step=5)
            with d5: dflt = st.checkbox("default?", value=False)
            if st.button("Deploy now"):
                ok = ledger_add_deal(label, amt, factor, term, st.session_state[KEY_DAY], dflt)
                if ok:
                    st.success("Deal deployed.")
                    # NEW: persist deployment
                    insert_ledger_deal_row(st.session_state[KEY_LEDGER][-1])

        # Ledger table
        ldf = ledger_dataframe()
        if not ldf.empty:
            show = ldf.copy()
            for c in ["amount","daily","gross","collected"]: show[c] = show[c].map(dollars)
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No deals in ledger yet. Deploy one to start daily collections.")

        # Quick actions
        cqa1, cqa2, cqa3 = st.columns(3)
        with cqa1:
            if st.button("Advance 10 days"): ledger_advance_to_day(min(DEFAULTS.BD_PER_YEAR, st.session_state[KEY_DAY] + 10))
        with cqa2:
            if st.button("Advance 30 days"): ledger_advance_to_day(min(DEFAULTS.BD_PER_YEAR, st.session_state[KEY_DAY] + 30))
        with cqa3:
            if st.button("Reset Ledger"):
                st.session_state[KEY_LEDGER] = []; st.session_state[KEY_DAY] = 0
                st.session_state[KEY_CASH] = DEFAULTS.TARGET_CAPITAL if port.get("launched", False) else 0.0
                st.session_state[KEY_SKIM] = 0.0; st.success("Ledger reset.")

    # --- Run / Finalize ---
    with tabs[4]:
        st.markdown("### ✅ Run / Finalize Year")
        investors = st.session_state[KEY_INVESTORS]; fees_total = len(investors) * DEFAULTS.FLAT_FEE_PER_INVESTOR
        realized_profit = ledger_realized_profit(); early = st.session_state[KEY_SKIM]
        inv_total, capnow_total, capnow_final = final_splits(DEFAULTS.TARGET_CAPITAL, realized_profit, early, DEFAULTS.CAPNOW_TOTAL, fees_total)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Realized Profit (YTD)", dollars(realized_profit))
        with c2: st.metric("CapNow Early Skims (sum)", dollars(early))
        with c3: st.metric("CapNow Final Component", dollars(capnow_final))
        c4, c5, c6 = st.columns(3)
        with c4: st.metric("Investors' Total Distribution", dollars(inv_total))
        with c5: st.metric("CapNow – All-in (incl. fees)", dollars(capnow_total))
        with c6: st.metric("Fees Collected ($500 × N)", dollars(fees_total))
        cap_df = cap_table_df(investors, DEFAULTS.TARGET_CAPITAL); pay = payouts_df(cap_df, inv_total)
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

        st.dataframe(nice, use_container_width=True)

        # ----- Save Portfolio Run to Supabase -----
        summary = {
            "realized_profit": float(realized_profit),
            "capnow_early": float(early),
            "capnow_final_component": float(capnow_final),
            "investors_total_distribution": float(inv_total),
            "capnow_total": float(capnow_total),
            "fees_total": float(fees_total),
        }
        deals_list = ldf.to_dict(orient="records") if 'ldf' in locals() else []
        payouts_list = pay.to_dict(orient="records") if not pay.empty else []
        if st.button("💾 Save Portfolio Run"):
            save_portfolio_run(
                summary=summary,
                deals=deals_list,
                payouts=payouts_list,
                capnow_earnings=capnow_total
            )

# ===============================
# INVESTOR PAGES (UI polish only)
# ===============================

def page_investor():
    st.markdown("## ✨ Investor Portal")
    email = st.session_state[KEY_AUTH].get("user") or ""
    tabs = st.tabs(["Participate", "My Portfolio"])
    with tabs[0]:
        st.markdown("### Participate in Portfolio Raise")
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
                # NEW: persist request (no logic change)
                insert_request_row(name, req_email, amt)
    with tabs[1]:
        st.markdown("### My Portfolio")
        invs = [i for i in st.session_state[KEY_INVESTORS] if i.get("email") == email]
        if not invs:
            st.info("No approved investment on record for this login."); return
        deals = st.session_state.get(KEY_DEALS, [])
        tl, profit, early = simulate_model_b(DEFAULTS.TARGET_CAPITAL, deals, DEFAULTS.CAPNOW_EARLY, DEFAULTS.BD_PER_YEAR)
        inv_total, capnow_total, capnow_final = final_splits(DEFAULTS.TARGET_CAPITAL, profit, early, DEFAULTS.CAPNOW_TOTAL, len(st.session_state[KEY_INVESTORS]) * DEFAULTS.FLAT_FEE_PER_INVESTOR)
        cap_df = cap_table_df(st.session_state[KEY_INVESTORS], DEFAULTS.TARGET_CAPITAL); pay = payouts_df(cap_df, inv_total)
        my = pay[pay["Email"] == email]
        if my.empty:
            st.info("You're not in the latest cap table yet. Check back after approval."); return
        row = my.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Contribution", dollars(row["Contribution"]))
        with c2: st.metric("Ownership", f"{row['% Ownership']*100:.2f}%")
        with c3: st.metric("Projected Payout", dollars(row["Payout"]))
        with c4: st.metric("ROI %", f"{row['ROI %']:.2f}%")
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown("**Portfolio Performance (Discrete Sandbox)**")
        if not tl.empty:
            pretty = tl.copy()
            for col in ["start_principal", "deployed", "profit", "capnow_early", "reinvested", "end_principal"]:
                pretty[col] = pretty[col].map(dollars)
            pretty["completion"] = (pretty["completion"]*100).map(lambda v: f"{v:.0f}%")
            st.dataframe(pretty, use_container_width=True)
        else:
            st.info("Admin has not added deals to the discrete sandbox.")

# ===============================
# ROUTER (unchanged behavior)
# ===============================
init_state()

# click-to-enter auth
auth = st.session_state[KEY_AUTH]
if not auth["role"]:
    with st.sidebar.expander("Enter", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↪ Admin", use_container_width=True):
                st.session_state[KEY_AUTH] = {"role": "admin", "user": "admin"}; st.rerun()
        with col2:
            if st.button("↪ Investor", use_container_width=True):
                st.session_state[KEY_AUTH] = {"role": "investor", "user": "guest@investor"}; st.rerun()
else:
    with st.sidebar:
        st.markdown(f"**Logged in as:** `{auth['user'] or auth['role']}` · **Role:** `{auth['role']}`")
        if st.button("Log out", use_container_width=True):
            st.session_state[KEY_AUTH] = {"role": None, "user": None}; st.rerun()
    # Small indicator for Supabase status (UI only)
    if SUPA_ENABLED:
        st.sidebar.success("DB: Supabase connected")
    else:
        st.sidebar.info("DB: Supabase not configured")

role = st.session_state[KEY_AUTH]["role"]
if role == "admin":
    page_admin()
elif role == "investor":
    page_investor()
else:
    st.markdown("# 💠 CapNow — Portfolio Simulator (Model B)")
    st.write("Use the sidebar to enter as admin or investor.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("### Highlights")
    st.markdown("- Target: $100,000 · Min ticket $5,000 · $500 entry fee per investor")
    st.markdown("- Profit split: 40% CapNow / 60% Investors · Early skim 20% per completed deal")
    st.markdown("- Portfolio absorbs losses on defaults")
