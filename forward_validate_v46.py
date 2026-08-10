"""V4.6 forward-only paper validation for the locked V3.4 candidate.

The historical holdouts are consumed. This module therefore reads only data
from a fixed 2026-forward boundary (plus a pre-boundary indicator warm-up),
applies the unchanged V3.4 alternating SPY/sector rule, and reports completed
hypothetical trades for a separate $1,000 account. It cannot place orders.
The forward record is immature until it contains at least 50 completed trades
and three calendar years; no result before then can pass.
"""

from dataclasses import asdict
from pathlib import Path
import os
import time

import pandas as pd

import develop_v21 as v21
import develop_v32 as v32
import develop_v34 as v34


VERSION = "V4.6"
VARIANT = "FORWARD_ONLY_UNCHANGED_V34"
FORWARD_START = pd.Timestamp("2026-01-02")
WARMUP_START = "2025-01-01"
SYMBOLS = v34.SYMBOLS
FAMILY = v34.FAMILY
MIN_FORWARD_TRADES = 50
MIN_FORWARD_YEARS = 3
RESULTS_DIR = Path("forward_results_v46")


def current_download_end():
    return (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def load_forward_data():
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing")
    end = current_download_end()
    data = {}
    for position, symbol in enumerate(SYMBOLS):
        frame = v32.fetch_window(symbol, WARMUP_START, end, key)
        frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        frame = v21.add_indicators(frame)
        frame["momentum126"] = frame["close"] / frame["close"].shift(v32.MOMENTUM_SESSIONS) - 1
        data[symbol] = frame
        if position + 1 < len(SYMBOLS):
            time.sleep(8)
    return data


def completed_signals(data, end):
    signals = v34.build_signals(data, FORWARD_START, end)
    if signals.empty:
        return signals
    complete = signals.apply(
        lambda row: int(row["signal_idx"]) + FAMILY.max_hold_sessions < len(data[row["symbol"]]),
        axis=1,
    )
    return signals.loc[complete].reset_index(drop=True)


def fixed_year_stats(trades, start, end):
    years = list(range(start.year, end.year + 1))
    if trades.empty:
        return 0, len(years)
    work = trades.copy()
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    positive = (work.groupby("year")["trade_pnl"].sum().reindex(years, fill_value=0) > 0).sum()
    return int(positive), len(years)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_forward_data()
    last_date = min(frame["date"].max() for frame in data.values())
    signals = completed_signals(data, last_date)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = v21.summarize(trades, skips, "FORWARD_PAPER")
    positive_years, years = fixed_year_stats(trades, FORWARD_START, last_date)
    summary.update({"version": VERSION, "variant": VARIANT, "forward_only": True,
                    "forward_start": FORWARD_START.date().isoformat(),
                    "data_through": last_date.date().isoformat(),
                    "positive_years": positive_years, "years_observed": years,
                    "minimum_trades": MIN_FORWARD_TRADES,
                    "minimum_years": MIN_FORWARD_YEARS, **asdict(FAMILY)})
    mature = bool(summary.get("trades", 0) >= MIN_FORWARD_TRADES and years >= MIN_FORWARD_YEARS)
    passed = bool(
        mature
        and summary["bootstrap_mean_95_ci_low_pct"] > 0
        and summary["profit_factor"] >= 1.50
        and summary["account_max_drawdown_pct"] >= -25
        and summary["max_ticker_profit_contribution_pct"] <= 60
        and summary["max_month_profit_contribution_pct"] <= 35
        and positive_years >= years - 1
    )
    summary["forward_mature"] = mature
    summary["forward_pass"] = passed
    status = "FORWARD_PASS" if passed else ("FORWARD_FAIL_MATURE" if mature else "FORWARD_IMMATURE")
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v46_forward_summary.csv", index=False)
    pd.DataFrame([{"status": status,
                   "reason": "ALL_FORWARD_GATES_PASSED" if passed else
                             ("FORWARD_GATES_FAILED" if mature else "MORE_NEW_DATA_REQUIRED")}]).to_csv(
        RESULTS_DIR / "v46_forward_status.csv", index=False
    )
    signals.to_csv(RESULTS_DIR / "v46_forward_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v46_forward_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v46_forward_skips.csv", index=False)
    print("V4.6 FORWARD-ONLY PAPER VALIDATION")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(status)


if __name__ == "__main__":
    main()
