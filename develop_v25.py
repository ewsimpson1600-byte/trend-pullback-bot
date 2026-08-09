"""V2.5 monthly ETF relative-momentum rotation research.

After pooled sector mean reversion diluted V2.2's edge, V2.5 changes signal
structure rather than pruning ETFs after observing results. Once per month it
selects the strongest positive six-month momentum ETF that is above its
200-day EMA, provided SPY is above a rising 200-day EMA. Entry is next open,
the planned stop is three ATR, and the maximum hold is 20 sessions. The
simulator remains $1,000 cash-only, integer-share, costed, gap-aware and
research-only. Rules are frozen on 2010-2017 before any 2018-2025 evaluation.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22
import develop_v24 as v24


VERSION = "V2.5"
VARIANT = "MONTHLY_RELATIVE_MOMENTUM_ROTATION"
SYMBOLS = v24.SYMBOLS
MOMENTUM_SESSIONS = 126
FAMILY = v21.Family("MONTHLY_ROTATION", 3.0, 100.0, 20)
RESULTS_DIR = Path("backtest_results_v25")


def load_data():
    data = v24.load_data()
    for symbol, frame in data.items():
        frame["momentum126"] = frame["close"] / frame["close"].shift(MOMENTUM_SESSIONS) - 1
        data[symbol] = frame
    return data


def monthly_signal_dates(spy, start, end):
    period = spy.loc[spy["date"].between(start, end), ["date"]].copy()
    if period.empty:
        return []
    period["month"] = period["date"].dt.to_period("M")
    return period.groupby("month")["date"].max().tolist()


def build_signals(data, start, end):
    spy = data["SPY"]
    spy_by_date = spy.set_index("date")
    rows = []
    for date in monthly_signal_dates(spy, start, end):
        if date not in spy_by_date.index:
            continue
        spy_idx = int(spy.index[spy["date"] == date][0])
        if spy_idx < v22.REGIME_SLOPE_SESSIONS:
            continue
        spy_row = spy.loc[spy_idx]
        prior_ema = spy.loc[spy_idx - v22.REGIME_SLOPE_SESSIONS, "ema200"]
        if pd.isna(prior_ema) or not (spy_row["close"] > spy_row["ema200"] > prior_ema):
            continue
        for symbol, frame in data.items():
            matches = frame.index[frame["date"] == date]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            row = frame.loc[idx]
            if pd.isna(row["atr14"]) or pd.isna(row["momentum126"]) or pd.isna(row["ema200"]):
                continue
            if row["close"] <= row["ema200"] or row["momentum126"] <= 0:
                continue
            rows.append(
                {
                    "family": FAMILY.name,
                    "variant": VARIANT,
                    "symbol": symbol,
                    "signal_idx": idx,
                    "signal_time": date,
                    "atr": float(row["atr14"]),
                    "strength": float(row["momentum126"] * 100),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength"])
    return pd.DataFrame(rows).sort_values(
        ["signal_time", "strength", "symbol"], ascending=[True, False, True]
    ).reset_index(drop=True)


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(
            RESULTS_DIR / f"{prefix}_by_{suffix}.csv"
        )


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    dev_signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_trades, dev_skips = v21.run_account(dev_signals, data, FAMILY)
    dev_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "universe_size": len(SYMBOLS),
        "momentum_sessions": MOMENTUM_SESSIONS,
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v25_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v25_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.5 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v25_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "universe_size": len(SYMBOLS),
        "momentum_sessions": MOMENTUM_SESSIONS,
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v25_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.5 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
