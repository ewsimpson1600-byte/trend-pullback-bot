"""V2.8 cross-asset breakout trend-following research.

This is an independent, long/cash ETF-share strategy rather than another
mean-reversion threshold revision. It trades a predeclared liquid cross-asset
universe after a 55-session closing breakout, provided the asset is above its
200-day EMA and has positive six-month momentum. Entries occur next session;
the simulator remains $1,000 cash-only, integer-share, costed, gap-aware, and
research-only. Rules are frozen on 2010-2017 before 2018-2025 can be opened.
"""

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22


VERSION = "V2.8"
VARIANT = "CROSS_ASSET_BREAKOUT_TREND"
SYMBOLS = ("SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ")
BREAKOUT_SESSIONS = 55
MOMENTUM_SESSIONS = 126
FAMILY = v21.Family("CROSS_ASSET_TREND", 2.5, 4.0, 20)
RESULTS_DIR = Path("backtest_results_v28")


def load_data():
    v21.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = None
    data = {}
    for symbol in SYMBOLS:
        path = v21.CACHE_DIR / f"{symbol}_1day.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
        else:
            key = key or v21.require_api_key()
            frame = v21.fetch_daily(symbol, key)
            frame.to_csv(path, index=False)
            time.sleep(8)
        frame = v21.add_indicators(frame)
        frame["prior_high55"] = frame["high"].shift(1).rolling(BREAKOUT_SESSIONS).max()
        frame["momentum126"] = frame["close"] / frame["close"].shift(MOMENTUM_SESSIONS) - 1
        data[symbol] = frame
    return data


def build_signals(data, start, end):
    rows = []
    for symbol, frame in data.items():
        ready = (
            frame["date"].between(start, end)
            & frame[["atr14", "ema200", "prior_high55", "momentum126"]].notna().all(axis=1)
        )
        valid = (
            ready
            & (frame["close"] > frame["prior_high55"])
            & (frame["close"] > frame["ema200"])
            & (frame["momentum126"] > 0)
        )
        breakout_pct = (frame["close"] / frame["prior_high55"] - 1) * 100
        strength = breakout_pct + frame["momentum126"] * 10
        for idx in frame.index[valid]:
            rows.append(
                {
                    "family": FAMILY.name,
                    "variant": VARIANT,
                    "symbol": symbol,
                    "signal_idx": int(idx),
                    "signal_time": frame.at[idx, "date"],
                    "atr": float(frame.at[idx, "atr14"]),
                    "strength": float(strength.at[idx]),
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
        "breakout_sessions": BREAKOUT_SESSIONS,
        "momentum_sessions": MOMENTUM_SESSIONS,
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v28_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v28_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.8 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v28_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "universe_size": len(SYMBOLS),
        "breakout_sessions": BREAKOUT_SESSIONS,
        "momentum_sessions": MOMENTUM_SESSIONS,
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v28_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.8 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
