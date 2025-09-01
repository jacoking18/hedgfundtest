# streamlit_app.py
# ---------------------------------------------
# Portfolio Syndication Simulator (Model B)
# - Page 1: Capital Raise (add investors until target reached)
# - Page 2: Admin/Simulate (cycle math, payouts)
# - Page 3: Investor View (simple investor-facing cap table + payouts)
# ---------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import List, Dict

import pandas as pd
import streamlit as st

# ===============================
# BIN 1 — CONFIG & CONSTANTS
# ===============================
@dataclass(frozen=True)
class AppDefaults:
    TARGET_CAPITAL: float = 100_000.0
    MIN_TICKET: float = 5_000.0
    MGMT_FEE_RATE: float = 0.01
    FACTOR: float = 1.40
    BD_PER_CYCLE: int = 90
    BD_PER_YEAR: int = 252
    CAPNOW_EARLY: float = 0.20
    CAPNOW_TOTAL: float = 0.40

DEFAULTS = AppDefaults()

# ===============================
# BIN 2 — STATE HELPERS
# ===============================
INVESTORS_KEY = "investors"
LOCKED_KEY = "locked"
PARAMS_KEY = "params"

@dataclass
class RuntimeParams:
    factor: float = DEFAULTS.FACTOR
    bd_cycle: int = DEFAULTS.BD_PER_CYCLE
    bd_year: int = DEFAULTS.BD_PER_YEAR
    capnow_early: float = DEFAULTS.CAPNOW_EARLY
    capnow_total: float = DEFAULTS.CAPNOW_TOTAL
    mgmt_fee_rate: float = DEFAULTS.MGMT_FEE_RATE


def _init_session():
    if INVESTORS_KEY not in st.session_state:
        st.session_state[INVESTORS_KEY] = []
    if LOCKED_KEY not in st.session_state:
        st.session_state[LOCKED_KEY] = False
    if PARAMS_KEY not in st.session_state:
        st.session_state[PARAMS_KEY] = RuntimeParams()

# ===============================
# BIN 3 — PURE CALC FUNCTIONS
# ===============================
@dataclass
class CycleRow:
    cycle: int
    start_principal: float
    profit: float
    capnow_early: float
    reinvested: float
    end_principal: float


def compute_cycles(start_capital: float, factor: float, bd_cycle: int, bd_year: int, capnow_early_rate: float):
    n_full = bd_year // bd_cycle
    remainder = bd_year - n_full * bd_cycle
    r_last = remainder / bd_cycle if bd_cycle > 0 else 0.0

    S = float(start_capital)
    rows: List[CycleRow] = []
    profits_full = []
    early_skims = []

    for i in range(1, n_full + 1):
        P = S * (factor - 1.0)
        skim = capnow_early_rate * P
        reinvested = (P - skim)
        S_end = S + reinvested
        rows.append(CycleRow(i, S, P, skim, reinvested, S_end))
        profits_full.append(P)
        early_skims.append(skim)
        S = S_end

    P_partial = S * (factor - 1.0) * r_last
    if r_last > 0:
        rows.append(CycleRow(n_full + 1, S, P_partial, 0.0, P_partial, S + P_partial))

    total_profit = sum(profits_full) + P_partial
    total_early = sum(early_skims)

    timeline_df = pd.DataFrame([asdict(r) for r in rows])
    return timeline_df, total_profit, total_early


def final_distribution(target_capital: float, total_realized_profit: float, total_early_skims: float, capnow_total_rate: float, mgmt_fee_rate: float):
    capnow_total_profit_take = capnow_total_rate * total_realized_profit
    capnow_final_component = capnow_total_profit_take - total_early_skims
    investors_total = target_capital + (1.0 - capnow_total_rate) * total_realized_profit
    capnow_total = total_early_skims + capnow_final_component + mgmt_fee_rate * target_capital
    return investors_total, capnow_total


def build_cap_table(investors: List[Dict], target_capital: float, mgmt_fee_rate: float) -> pd.DataFrame:
    df = pd.DataFrame(investors)
    if df.empty:
        return pd.DataFrame(columns=["Name", "Contribution", "% Ownership", "Mgmt Fee", "Net Investable"])
    df["% Ownership"] = df["contribution"] / target_capital
    df["Mgmt Fee"] = df["contribution"] * mgmt_fee_rate
    df["Net Investable"] = df["contribution"]
    df = df.rename(columns={"name": "Name", "contribution": "Contribution"})[
        ["Name", "Contribution", "% Ownership", "Mgmt Fee", "Net Investable"]
    ]
    return df


def compute_investor_payouts(cap_table: pd.DataFrame, investors_total_distribution: float) -> pd.DataFrame:
    if cap_table.empty:
        return pd.DataFrame(columns=["Name", "Contribution", "% Ownership", "Payout", "ROI %"])
    out = cap_table.copy()
    out["Payout"] = (out["% Ownership"] * investors_total_distribution).round(2)
    out["ROI %"] = ((out["Payout"] - out["Contribution"]) / out["Contribution"]) * 100.0
    return out[["Name", "Contribution", "% Ownership", "Payout", "ROI %", "Mgmt Fee"]]


def dollars(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"

def percent(x: float) -> str:
    return f"{x*100:.2f}%"

# ===============================
# BIN 4 — UI COMPONENTS
# ===============================
class UI:
    @staticmethod
    def header(title: str, subtitle: str | None = None):
        st.markdown(f"## {title}")
        if subtitle:
            st.caption(subtitle)

    @staticmethod
    def divider():
        st.markdown("---")

# ===============================
# BIN 5 — PAGE 1: CAPITAL RAISE
# ===============================
def page_raise_capital(target_capital: float):
    UI.header("Page 1 — Raise Capital")
    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    locked: bool = st.session_state[LOCKED_KEY]

    name = st.text_input("Investor name", key="name_input", disabled=locked)
    input_mode = st.radio("Input mode", ["$ Amount", "% of Portfolio"], horizontal=True, disabled=locked)
    amount_value = st.number_input("Amount", min_value=0.0, step=100.0, key="amount_input", disabled=locked)
    auto_trim = st.checkbox("Auto-trim last contribution", value=True, disabled=locked)

    def _current_total():
        return sum(i["contribution"] for i in investors)

    def _add_investor():
        nonlocal investors
        if locked:
            return
        nm = name.strip()
        if not nm:
            return
        if any(i["name"].lower() == nm.lower() for i in investors):
            st.warning("Duplicate name")
            return
        contrib = float(amount_value) if input_mode == "$ Amount" else float(amount_value) * 0.01 * target_capital
        if contrib < DEFAULTS.MIN_TICKET:
            st.warning("Below minimum ticket")
            return
        remaining = target_capital - _current_total()
        if contrib > remaining:
            if auto_trim:
                contrib = remaining
            else:
                return
        investors.append({"name": nm, "contribution": round(contrib, 2)})
        st.session_state[INVESTORS_KEY] = investors

    st.button("Add Investor", on_click=_add_investor, disabled=locked)

    total = _current_total()
    st.progress(min(total / target_capital, 1.0), text=f"{dollars(total)} / {dollars(target_capital)}")

    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    show_df = cap_table.copy()
    show_df["Contribution"] = show_df["Contribution"].map(dollars)
    show_df["% Ownership"] = show_df["% Ownership"].map(percent)
    show_df["Mgmt Fee"] = show_df["Mgmt Fee"].map(dollars)
    show_df["Net Investable"] = show_df["Net Investable"].map(dollars)
    st.dataframe(show_df, use_container_width=True)

    if st.button("Reset Roster", disabled=locked):
        st.session_state[INVESTORS_KEY] = []
        st.rerun()

# ===============================
# BIN 6 — PAGE 2: ADMIN
# ===============================
def page_admin(target_capital: float):
    UI.header("Page 2 — Admin & Simulation")
    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    if not investors:
        st.info("No investors yet.")
        return
    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    show_table = cap_table.copy()
    show_table["Contribution"] = show_table["Contribution"].map(dollars)
    show_table["% Ownership"] = show_table["% Ownership"].map(percent)
    show_table["Mgmt Fee"] = show_table["Mgmt Fee"].map(dollars)
    show_table["Net Investable"] = show_table["Net Investable"].map(dollars)
    st.dataframe(show_table, use_container_width=True)

    timeline_df, total_profit, total_early = compute_cycles(target_capital, DEFAULTS.FACTOR, DEFAULTS.BD_PER_CYCLE, DEFAULTS.BD_PER_YEAR, DEFAULTS.CAPNOW_EARLY)
    st.dataframe(timeline_df, use_container_width=True)
    investors_total, capnow_total = final_distribution(target_capital, total_profit, total_early, DEFAULTS.CAPNOW_TOTAL, DEFAULTS.MGMT_FEE_RATE)
    st.metric("Investors Total", dollars(investors_total))
    st.metric("CapNow Total", dollars(capnow_total))

    payouts_df = compute_investor_payouts(cap_table, investors_total)
    show_payouts = payouts_df.copy()
    show_payouts["Contribution"] = show_payouts["Contribution"].map(dollars)
    show_payouts["% Ownership"] = show_payouts["% Ownership"].map(percent)
    show_payouts["Mgmt Fee"] = show_payouts["Mgmt Fee"].map(dollars)
    show_payouts["Payout"] = show_payouts["Payout"].map(dollars)
    show_payouts["ROI %"] = show_payouts["ROI %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(show_payouts, use_container_width=True)

# ===============================
# BIN 7 — PAGE 3: INVESTOR VIEW
# ===============================
def page_investor_view(target_capital: float):
    UI.header("Page 3 — Investor View")
    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    if not investors:
        st.info("No investors yet.")
        return
    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    show_table = cap_table.copy()
    show_table["Contribution"] = show_table["Contribution"].map(dollars)
    show_table["% Ownership"] = show_table["% Ownership"].map(percent)
    show_table["Mgmt Fee"] = show_table["Mgmt Fee"].map(dollars)
    show_table["Net Investable"] = show_table["Net Investable"].map(dollars)
    st.dataframe(show_table, use_container_width=True)

    timeline_df, total_profit, total_early = compute_cycles(target_capital, DEFAULTS.FACTOR, DEFAULTS.BD_PER_CYCLE, DEFAULTS.BD_PER_YEAR, DEFAULTS.CAPNOW_EARLY)
    investors_total, _ = final_distribution(target_capital, total_profit, total_early, DEFAULTS.CAPNOW_TOTAL, DEFAULTS.MGMT_FEE_RATE)
    payouts_df = compute_investor_payouts(cap_table, investors_total)
    show_payouts = payouts_df.copy()
    show_payouts["Contribution"] = show_payouts["Contribution"].map(dollars)
    show_payouts["% Ownership"] = show_payouts["% Ownership"].map(percent)
    show_payouts["Mgmt Fee"] = show_payouts["Mgmt Fee"].map(dollars)
    show_payouts["Payout"] = show_payouts["Payout"].map(dollars)
    show_payouts["ROI %"] = show_payouts["ROI %"].map(lambda x: f"{x:.2f}%")
    st.subheader("Investor Payouts")
    st.dataframe(show_payouts, use_container_width=True)

    names = payouts_df["Name"].tolist()
    choice = st.selectbox("Select investor", names)
    row = payouts_df[payouts_df["Name"] == choice].iloc[0]
    st.metric("Contribution", dollars(row["Contribution"]))
    st.metric("Ownership", percent(row["% Ownership"]))
    st.metric("Payout", dollars(row["Payout"]))
    st.metric("ROI %", f"{row['ROI %']:.2f}%")

# ===============================
# MAIN APP
# ===============================
def main():
    st.set_page_config(page_title="Portfolio Syndication Simulator", layout="wide")
    _init_session()

    page = st.sidebar.radio("Go to", ["Page 1 — Raise Capital", "Page 2 — Admin & Simulation", "Page 3 — Investor View"])
    target = DEFAULTS.TARGET_CAPITAL

    with st.sidebar.expander("Quick Demo Roster"):
        if st.button("Load 20×$5,000 (exact $100k)"):
            st.session_state[INVESTORS_KEY] = [{"name": f"Investor {i+1}", "contribution": 5_000.0} for i in range(20)]
            st.session_state[LOCKED_KEY] = False
            st.rerun()
        if st.button("Clear Roster"):
            st.session_state[INVESTORS_KEY] = []
            st.session_state[LOCKED_KEY] = False
            st.rerun()
        st.markdown("**Edit roster (name & contribution)**")
        current_df = pd.DataFrame(st.session_state[INVESTORS_KEY])
        edited = st.data_editor(current_df, num_rows="dynamic", use_container_width=True, key="roster_editor")
        if st.button("Save roster changes"):
            new_list: List[Dict] = []
            if not edited.empty:
                for _, r in edited.iterrows():
                    nm = str(r.get("name", "")).strip()
                    contrib = float(r.get("contribution", 0) or 0)
                    if nm and contrib > 0:
                        new_list.append({"name": nm, "contribution": round(contrib, 2)})
            st.session_state[INVESTORS_KEY] = new_list
            st.session_state[LOCKED_KEY] = False
            st.rerun()

    if page.startswith("Page 1"):
        page_raise_capital(target)
    elif page.startswith("Page 2"):
        page_admin(target)
    else:
        page_investor_view(target)

if __name__ == "__main__":
    main()
