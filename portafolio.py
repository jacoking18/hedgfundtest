# streamlit_app.py
# ---------------------------------------------
# Portfolio Syndication Simulator (Model B)
# - Page 1: Capital Raise (add investors until target reached)
# - Page 2: Admin/Simulate (cycle math, payouts)
#
# Design goals:
# - Reusable "bins" (config, state, calc, ui) to minimize change impact
# - Pure functions for calculations; thin UI layer
# - Single-file for easy deploy; structured for maintainability
# ---------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ===============================
# BIN 1 — CONFIG & CONSTANTS
# ===============================
@dataclass(frozen=True)
class AppDefaults:
    TARGET_CAPITAL: float = 100_000.0
    MIN_TICKET: float = 5_000.0              # hard floor per investor
    MGMT_FEE_RATE: float = 0.01              # 1% on top (doesn't reduce investable capital)
    FACTOR: float = 1.40                     # 1.40 conservative default
    BD_PER_CYCLE: int = 90                   # 90 business days
    BD_PER_YEAR: int = 252                   # business day year
    CAPNOW_EARLY: float = 0.20               # 20% skim each completed cycle
    CAPNOW_TOTAL: float = 0.40               # 40% of realized profits overall
    ALLOW_PERCENT_INPUT: bool = True         # allow adding by % as well as $ amount

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
        st.session_state[INVESTORS_KEY] = []  # list[dict(name, contribution)]
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


def compute_cycles(
    start_capital: float,
    factor: float,
    bd_cycle: int,
    bd_year: int,
    capnow_early_rate: float,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Returns (timeline_df, total_realized_profit, sum_early_skims, S_after_full, partial_profit)
    Where S_after_full is the principal after all *full* cycles (before partial).
    """
    n_full = bd_year // bd_cycle
    remainder = bd_year - n_full * bd_cycle
    r_last = remainder / bd_cycle if bd_cycle > 0 else 0.0

    S = float(start_capital)
    rows: List[CycleRow] = []
    profits_full = []
    early_skims = []

    # Full cycles
    for i in range(1, n_full + 1):
        P = S * (factor - 1.0)
        skim = capnow_early_rate * P
        reinvested = (P - skim)
        S_end = S + reinvested
        rows.append(CycleRow(
            cycle=i,
            start_principal=S,
            profit=P,
            capnow_early=skim,
            reinvested=reinvested,
            end_principal=S_end,
        ))
        profits_full.append(P)
        early_skims.append(skim)
        S = S_end

    # Partial cycle
    P_partial = S * (factor - 1.0) * r_last
    if r_last > 0:
        rows.append(CycleRow(
            cycle=n_full + 1,
            start_principal=S,
            profit=P_partial,
            capnow_early=0.0,  # no early skim on partial at year-end
            reinvested=P_partial,  # for display only; real split happens at year end
            end_principal=S + P_partial,
        ))

    total_profit = sum(profits_full) + P_partial
    total_early = sum(early_skims)

    timeline_df = pd.DataFrame([asdict(r) for r in rows]) if rows else pd.DataFrame(
        columns=[
            "cycle",
            "start_principal",
            "profit",
            "capnow_early",
            "reinvested",
            "end_principal",
        ]
    )
    return timeline_df, total_profit, total_early, S, P_partial


def final_distribution(
    target_capital: float,
    total_realized_profit: float,
    total_early_skims: float,
    capnow_total_rate: float,
    mgmt_fee_rate: float,
) -> Tuple[float, float, float]:
    """Compute final splits.
    Returns (investors_total_distribution, capnow_total_take, capnow_final_component)
    """
    capnow_total_profit_take = capnow_total_rate * total_realized_profit
    capnow_final_component = capnow_total_profit_take - total_early_skims
    investors_total = target_capital + (1.0 - capnow_total_rate) * total_realized_profit
    capnow_total = total_early_skims + capnow_final_component + mgmt_fee_rate * target_capital
    return investors_total, capnow_total, capnow_final_component


def build_cap_table(investors: List[Dict], target_capital: float, mgmt_fee_rate: float) -> pd.DataFrame:
    df = pd.DataFrame(investors)
    if df.empty:
        return pd.DataFrame(columns=["Name", "Contribution", "% Ownership", "Mgmt Fee", "Net Investable"])
    df["% Ownership"] = df["Contribution"] / target_capital
    df["Mgmt Fee"] = df["Contribution"] * mgmt_fee_rate
    df["Net Investable"] = df["Contribution"]  # fee billed on top; does not reduce investable funds
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
    # Reorder
    out = out[["Name", "Contribution", "% Ownership", "Payout", "ROI %", "Mgmt Fee"]]
    return out


def dollars(x: float) -> str:
    return f"${x:,.2f}"


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

    @staticmethod
    def info_badge(label: str, value: str):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**{label}**")
        with c2:
            st.markdown(value)


# ===============================
# BIN 5 — PAGE 1: CAPITAL RAISE
# ===============================

def page_raise_capital(target_capital: float):
    st.sidebar.subheader("Navigation")
    st.sidebar.write("You're on: **Page 1 – Raise Capital**")

    UI.header("Page 1 — Raise Capital", "Add investors until the portfolio reaches the target. Minimum ticket $5,000.")

    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    locked: bool = st.session_state[LOCKED_KEY]

    colA, colB = st.columns([2, 1])
    with colA:
        name = st.text_input("Investor name", key="name_input", disabled=locked)
    with colB:
        input_mode = st.radio("Input mode", ["$ Amount", "% of Portfolio"], horizontal=True, disabled=locked)

    amount_value = st.number_input(
        "Amount (in selected mode)", min_value=0.0, step=100.0, key="amount_input", disabled=locked
    )

    auto_trim = st.checkbox("Auto-trim last contribution to hit target exactly", value=True, disabled=locked)

    def _current_total():
        return sum(i["contribution"] for i in investors)

    def _add_investor():
        nonlocal investors
        if locked:
            return
        nm = name.strip()
        if not nm:
            st.warning("Enter a name.")
            return
        if any(i["name"].lower() == nm.lower() for i in investors):
            st.warning("Name already exists. Use a unique name.")
            return
        # Compute contribution
        if input_mode == "$ Amount":
            contrib = float(amount_value)
        else:
            contrib = float(amount_value) * 0.01 * target_capital
        if contrib <= 0:
            st.warning("Contribution must be > 0.")
            return
        # Enforce min ticket
        if contrib < DEFAULTS.MIN_TICKET:
            st.warning(f"Minimum ticket is {dollars(DEFAULTS.MIN_TICKET)}.")
            return
        # Oversubscription handling
        remaining = target_capital - _current_total()
        if contrib > remaining:
            if auto_trim:
                contrib = remaining
            else:
                st.warning("This exceeds the target. Enable auto-trim or enter a smaller amount.")
                return
        if remaining <= 0:
            st.info("Target already reached.")
            return
        investors.append({"name": nm, "contribution": round(contrib, 2)})
        st.session_state[INVESTORS_KEY] = investors
        st.success(f"Added {nm} — {dollars(contrib)}")

    st.button("Add Investor", on_click=_add_investor, disabled=locked)

    UI.divider()

    total = _current_total()
    st.progress(min(total / target_capital, 1.0), text=f"Committed: {dollars(total)} / {dollars(target_capital)}")

    cap_table = build_cap_table(investors, target_capital, DEFAULTS.MGMT_FEE_RATE)
    st.dataframe(cap_table, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("# Investors", len(investors))
    with c2:
        st.metric("Committed", dollars(total))
    with c3:
        st.metric("Remaining", dollars(max(target_capital - total, 0)))
    with c4:
        st.metric("Mgmt Fees (total)", dollars(total * DEFAULTS.MGMT_FEE_RATE))

    UI.divider()

    colL, colR = st.columns([1, 1])
    with colL:
        if st.button("Reset Roster", disabled=locked):
            st.session_state[INVESTORS_KEY] = []
            st.experimental_rerun()
    with colR:
        can_start = math.isclose(total, target_capital, rel_tol=0, abs_tol=0.01)
        if st.button("Lock & Go to Admin", disabled=(not can_start)):
            st.session_state[LOCKED_KEY] = True
            st.experimental_rerun()


# ===============================
# BIN 6 — PAGE 2: ADMIN & SIMULATION
# ===============================

def page_admin(target_capital: float):
    st.sidebar.subheader("Navigation")
    st.sidebar.write("You're on: **Page 2 – Admin & Simulation**")

    UI.header("Page 2 — Admin & Simulation", "Review cap table, set parameters, run cycles, and compute payouts.")

    investors: List[Dict] = st.session_state[INVESTORS_KEY]
    locked: bool = st.session_state[LOCKED_KEY]

    if not investors:
        st.info("No investors yet. Add them on Page 1.")
        return

    # Params sidebar
    with st.sidebar:
        st.markdown("### Parameters")
        rp: RuntimeParams = st.session_state[PARAMS_KEY]
        rp.factor = st.number_input("Factor (e.g., 1.40)", value=rp.factor, step=0.01, format="%.2f")
        rp.bd_cycle = st.number_input("Business days per cycle", min_value=1, value=rp.bd_cycle, step=1)
        rp.bd_year = st.number_input("Business days per year", min_value=1, value=rp.bd_year, step=1)
        rp.capnow_early = st.slider("CapNow early skim per full cycle", 0.0, 0.5, rp.capnow_early, 0.01)
        rp.capnow_total = st.slider("CapNow total profit share", 0.0, 0.6, rp.capnow_total, 0.01)
        rp.mgmt_fee_rate = st.slider("Mgmt fee (on top)", 0.0, 0.05, rp.mgmt_fee_rate, 0.001)
        st.session_state[PARAMS_KEY] = rp

        st.markdown("---")
        start_clicked = st.button("Start Portfolio (lock roster)")
        if start_clicked:
            st.session_state[LOCKED_KEY] = True

    cap_table = build_cap_table(investors, target_capital, st.session_state[PARAMS_KEY].mgmt_fee_rate)

    # Show cap table
    st.subheader("Cap Table (Read-only)")
    st.dataframe(cap_table, use_container_width=True)

    # Safety: ensure target reached
    committed = cap_table["Contribution"].sum() if not cap_table.empty else 0.0
    if not math.isclose(committed, target_capital, rel_tol=0, abs_tol=0.01):
        st.error("Total committed does not equal the target. Go back to Page 1 and finish raising.")
        return

    UI.divider()

    # Compute cycle timeline
    rp = st.session_state[PARAMS_KEY]
    timeline_df, total_profit, total_early, S_after_full, P_partial = compute_cycles(
        start_capital=target_capital,
        factor=rp.factor,
        bd_cycle=rp.bd_cycle,
        bd_year=rp.bd_year,
        capnow_early_rate=rp.capnow_early,
    )

    st.subheader("Cycle Timeline")
    if timeline_df.empty:
        st.info("No cycles computed. Check your parameters.")
        return

    # Pretty formatting for display
    show_df = timeline_df.copy()
    for col in ["start_principal", "profit", "capnow_early", "reinvested", "end_principal"]:
        show_df[col] = show_df[col].map(dollars)
    st.dataframe(show_df, use_container_width=True)

    # Final splits
    investors_total, capnow_total, capnow_final = final_distribution(
        target_capital=target_capital,
        total_realized_profit=total_profit,
        total_early_skims=total_early,
        capnow_total_rate=rp.capnow_total,
        mgmt_fee_rate=rp.mgmt_fee_rate,
    )

    UI.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total realized profit", dollars(total_profit))
    with c2:
        st.metric("CapNow – early skims (sum)", dollars(total_early))
    with c3:
        st.metric("CapNow – final component", dollars(capnow_final))

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Investors' total distribution", dollars(investors_total))
    with c5:
        st.metric("CapNow – all-in (incl. mgmt fee)", dollars(capnow_total))
    with c6:
        # Compute effective cycles
        n_full = rp.bd_year // rp.bd_cycle
        r_last = (rp.bd_year - n_full * rp.bd_cycle) / rp.bd_cycle if rp.bd_cycle > 0 else 0
        st.metric("Effective cycles", f"{n_full} + {r_last:.2f}")

    UI.divider()

    # Per-investor payouts
    st.subheader("Per-Investor Payouts")
    payouts_df = compute_investor_payouts(cap_table, investors_total)

    # Nicely formatted view
    show_payouts = payouts_df.copy()
    show_payouts["Contribution"] = show_payouts["Contribution"].map(dollars)
    show_payouts["Payout"] = show_payouts["Payout"].map(dollars)
    show_payouts["Mgmt Fee"] = show_payouts["Mgmt Fee"].map(dollars)
    show_payouts["% Ownership"] = (show_payouts["% Ownership"] * 100).map(lambda x: f"{x:.2f}%")
    show_payouts["ROI %"] = show_payouts["ROI %"].map(lambda x: f"{x:.2f}%")

    st.dataframe(show_payouts, use_container_width=True)

    # Downloads
    UI.divider()
    st.download_button("Download Cap Table (CSV)", cap_table.to_csv(index=False).encode("utf-8"), file_name="cap_table.csv")
    st.download_button("Download Cycle Timeline (CSV)", timeline_df.to_csv(index=False).encode("utf-8"), file_name="cycle_timeline.csv")
    st.download_button("Download Payouts (CSV)", payouts_df.to_csv(index=False).encode("utf-8"), file_name="payouts.csv")

    # Footer info
    UI.divider()
    st.caption("Model B: 20% early skim per full cycle, 40% total profit share to CapNow at year-end (final component adjusts). Mgmt fee is billed on top and does not reduce investable funds.")


# ===============================
# BIN 7 — MAIN APP
# ===============================

def main():
    st.set_page_config(page_title="Portfolio Syndication Simulator — Model B", layout="wide")
    _init_session()

    st.sidebar.title("Portfolio Simulator")
    page = st.sidebar.radio("Go to", ["Page 1 — Raise Capital", "Page 2 — Admin & Simulation"]) 

    target = DEFAULTS.TARGET_CAPITAL

    # Quick demo roster helper
    with st.sidebar.expander("Quick Demo Roster"):
        if st.button("Load 20×$5,000 (exact $100k)", use_container_width=True):
            st.session_state[INVESTORS_KEY] = [
                {"name": f"Investor {i+1}", "contribution": 5_000.0} for i in range(20)
            ]
            st.session_state[LOCKED_KEY] = False
            st.experimental_rerun()
        if st.button("Clear Roster", use_container_width=True):
            st.session_state[INVESTORS_KEY] = []
            st.session_state[LOCKED_KEY] = False
            st.experimental_rerun()

    if page.startswith("Page 1"):
        page_raise_capital(target)
    else:
        page_admin(target)


if __name__ == "__main__":
    main()
