"""V3.1 monthly SPY absolute-trend research.

V2.5's cross-sectional sector selection was unprofitable. V3.1 removes that
selection and tests the broad-market absolute trend directly: once per month,
hold SPY only when it is above a rising 200-day EMA and has positive six-month
momentum. Entry is next open, the catastrophe stop is three ATR, and the
maximum hold is 20 sessions. The $1,000 simulator stays cash-only, integer-
share, costed, gap-aware, and research-only. Development is 2010-2017;
2018-2025 remains sealed unless every frozen gate passes.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22
import develop_v25 as v25


VERSION = "V3.1"
VARIANT = "MONTHLY_SPY_ABSOLUTE_TREND"
MOMENTUM_SESSIONS = 126
FAMILY = v21.Family("MONTHLY_SPY_TREND", 3.0, 100.0, 20)
RESULTS_DIR = Path("backtest_results_v31")


def load_data():
    data = v21.load_data()
    spy = data["SPY"]
    spy["momentum126"] = spy["close"] / spy["close"].shift(MOMENTUM_SESSIONS) - 1
    return {"SPY": spy}


def build_signals(data, start, end):
    spy = data["SPY"]
    rows = []
    for date in v25.monthly_signal_dates(spy, start, end):
        matches = spy.index[spy["date"] == date]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        if idx < v22.REGIME_SLOPE_SESSIONS:
            continue
        row = spy.loc[idx]
        prior_ema = spy.loc[idx - v22.REGIME_SLOPE_SESSIONS, "ema200"]
        if pd.isna(row["atr14"]) or pd.isna(row["momentum126"]) or pd.isna(prior_ema):
            continue
        if not (row["close"] > row["ema200"] > prior_ema and row["momentum126"] > 0):
            continue
        rows.append(
            {
                "family": FAMILY.name,
                "variant": VARIANT,
                "symbol": "SPY",
                "signal_idx": idx,
                "signal_time": date,
                "atr": float(row["atr14"]),
                "strength": float(row["momentum126"] * 100),
            }
        )
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength"])


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_{suffix}.csv")


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    dev = {"version": VERSION, "variant": VARIANT, "momentum_sessions": MOMENTUM_SESSIONS, **asdict(FAMILY), **v22.fixed_period_summary(trades, skips, "DEVELOPMENT")}
    write_period("v31_development", signals, trades, skips, dev)
    if not dev["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev}]).to_csv(RESULTS_DIR / "v31_candidate.csv", index=False)
        print(pd.DataFrame([dev]).to_string(index=False))
        print("V3.1 failed frozen development gates; validation was not opened.")
        return
    pd.DataFrame([{"status": "LOCKED", **dev}]).to_csv(RESULTS_DIR / "v31_candidate.csv", index=False)
    signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    val = {"version": VERSION, "variant": VARIANT, "momentum_sessions": MOMENTUM_SESSIONS, **asdict(FAMILY), **v22.fixed_period_summary(trades, skips, "VALIDATION")}
    write_period("v31_validation", signals, trades, skips, val)
    print("V3.1 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([val]).to_string(index=False))


if __name__ == "__main__":
    main()
