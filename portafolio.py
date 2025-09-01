# streamlit_app.py
# ---------------------------------------------
# Portfolio Syndication Simulator (Model B)
# - Page 1: Capital Raise
# - Page 2: Admin/Simulate
# - Page 3: Investor View
# - Flexible Deal Editor: play with factors, deal sizes, durations, defaults
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
    factor: float
    profit: float
    capnow_early: float
    reinvested: float
    end_principal: float
    defaulted: bool


def compute_flexible_cycles(start_capital: float, deals: List[Dict], capnow_early_rate: float, bd_per_cycle: int, bd_per_year: int):
    S = float(start_capital)
    rows: List[CycleRow] = []
    total_profit = 0.0
    total_early = 0.0
    used_days = 0

    for i, d in enumerate(deals, start=1):
        f = float(d.get("factor", 1.40))
        defaulted = bool(d.get("default", False))
        size_amt = float(d.get("size_amt", start_capital))  # absolute amount to deploy
        days = int(d.get("days", bd_per_cycle))

        # ensure we do not exceed current portfolio size
        deployed_base = min(S, size_amt)

        # Remaining days for the year & effective r for this cycle
        remaining_days = max(bd_per_year - used_days, 0)
        if remaining_days <= 0:
            r_eff = 0.0
        else:
            r_raw = days / bd_per_cycle if bd_per_cycle > 0 else 1.0
            r_cap = min(1.0, remaining_days / max(days, 1))
            r_eff = max(0.0, min(1.0, r_raw * r_cap))

        if defaulted:
            P = 0.0
            skim = 0.0
            reinvested = 0.0
            S_end = S - deployed_base
        else:
            P = deployed_base * (f - 1.0) * r_eff
            skim = capnow_early_rate * P if abs(r_eff - 1.0) < 1e-9 else 0.0
            reinvested = (P - skim)
            S_end = S + reinvested
            total_profit += P
            total_early += skim

        rows.append(CycleRow(i, S, f, P, skim, reinvested, S_end, defaulted))
        S = S_end
        used_days += min(days, remaining_days)

    timeline_df = pd.DataFrame([asdict(r) for r in rows])
    if len(deals) > 0:
        meta = pd.DataFrame({
            "cycle": list(range(1, len(deals)+1)),
            "factor": [float(d.get("factor", 1.40)) for d in deals],
            "size_amt": [float(d.get("size_amt", start_capital)) for d in deals],
            "days": [int(d.get("days", bd_per_cycle)) for d in deals],
            "defaulted": [bool(d.get("default", False)) for d in deals],
        })
        timeline_df = meta.merge(timeline_df, on="cycle", suffixes=("_cfg", ""))

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
# BIN 5 — PAGES
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
    st.dataframe(cap_table, use_container_width=True)


def page_admin(target_capital: float):
    UI.header("Page 2 — Admin & Simulation")
    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    if not investors:
        st.info("No investors yet.")
        return

    st.subheader("Deal Editor")
    deals = st.session_state.get("deals", [
        {"factor": 1.40, "size_amt": 10_000, "days": DEFAULTS.BD_PER_CYCLE, "default": False},
        {"factor": 1.40, "size_amt": 10_000, "days": DEFAULTS.BD_PER_CYCLE, "default": False},
    ])
    deals_df = pd.DataFrame(deals)
    edited = st.data_editor(
        deals_df,
        num_rows="dynamic",
        use_container_width=True,
        key="deals_editor",
        column_config={
            "factor": st.column_config.NumberColumn("factor", help="Deal factor e.g., 1.40 or 14.6", step=0.01, format="%.2f"),
            "size_amt": st.column_config.NumberColumn("deal size $", help="Absolute amount of portfolio deployed in this deal", step=1000, format="$%0.0f"),
            "days": st.column_config.NumberColumn("biz days", help="Business days for this deal", step=5, min_value=1),
            "default": st.column_config.CheckboxColumn("default", help="Lose deployed portion this deal"),
        }
    )
    st.session_state["deals"] = edited.to_dict(orient="records")

    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    st.dataframe(cap_table, use_container_width=True)

    timeline_df, total_profit, total_early = compute_flexible_cycles(
        target_capital,
        st.session_state["deals"],
        DEFAULTS.CAPNOW_EARLY,
        DEFAULTS.BD_PER_CYCLE,
        DEFAULTS.BD_PER_YEAR,
    )
    pretty_tl = timeline_df.copy()
    if not pretty_tl.empty:
        for col in ["start_principal", "profit", "capnow_early", "reinvested", "end_principal"]:
            pretty_tl[col] = pretty_tl[col].map(dollars)
        if "size_amt" in pretty_tl.columns:
            pretty_tl["size_amt"] = pretty_tl["size_amt"].map(dollars)
        if "days" in pretty_tl.columns:
            pretty_tl["days"] = pretty_tl["days"].astype(int)
        if "factor_cfg" in pretty_tl.columns:
            pretty_tl["factor_cfg"] = pretty_tl["factor_cfg"].map(lambda x: f"{x:.2f}")
        if "defaulted" in pretty_tl.columns:
            pretty_tl["defaulted"] = pretty_tl["defaulted"].map(lambda v: "❌ default" if v else "—")
    st.dataframe(pretty_tl, use_container_width=True)

    # Show averages
    if not timeline_df.empty:
        avg_factor = timeline_df["factor"].mean()
        avg_profit = timeline_df["profit"].mean()
        avg_days = pd.DataFrame(st.session_state["deals"]).get("days", pd.Series([DEFAULTS.BD_PER_CYCLE]*len(timeline_df))).mean()
        avg_size = pd.DataFrame(st.session_state["deals"]).get("size_amt", pd.Series([target_capital]*len(timeline_df))).mean()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Average Factor", f"{avg_factor:.2f}")
        with c2:
            st.metric("Avg Deal Profit", dollars(avg_profit))
        with c3:
            st.metric("Avg Deal Size", dollars(avg_size))
        with c4:
            st.metric("Avg Cycle Days", f"{avg_days:.0f} days")

    investors_total, capnow_total = final_distribution(target_capital, total_profit, total_early, DEFAULTS.CAPNOW_TOTAL, DEFAULTS.MGMT_FEE_RATE)
    st.metric("Investors Total", dollars(investors_total))
    st.metric("CapNow Total", dollars(capnow_total))

    payouts_df = compute_investor_payouts(cap_table, investors_total)
    st.dataframe(payouts_df, use_container_width=True)


def page_investor_view(target_capital: float):
    UI.header("Page 3 — Investor View")
    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    if not investors:
        st.info("No investors yet.")
        return
    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    st.dataframe(cap_table, use_container_width=True)

    deals = st.session_state.get("deals", [])
    timeline_df, total_profit, total_early = compute_flexible_cycles(
        target_capital,
        deals,
        DEFAULTS.CAPNOW_EARLY,
        DEFAULTS.BD_PER_CYCLE,
        DEFAULTS.BD_PER_YEAR,
    )
    investors_total, _ = final_distribution(target_capital, total_profit, total_early, DEFAULTS.CAPNOW_TOTAL, DEFAULTS.MGMT_FEE_RATE)
    payouts_df = compute_investor_payouts(cap_table, investors_total)
    st.dataframe(payouts_df, use_container_width=True)

    if not payouts_df.empty:
        choice = st.selectbox("Select investor", payouts_df["Name"].tolist())
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

    if page.startswith("Page 1"):
        page_raise_capital(target)
    elif page.startswith("Page 2"):
        page_admin(target)
    else:
        page_investor_view(target)

if __name__ == "__main__":
    main()
