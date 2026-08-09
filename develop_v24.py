"""V2.4 diversified ETF mean-reversion research.

V2.3 showed that strict next-day confirmation destroyed sample size. V2.4
returns to the frozen V2.2 regime-filtered mean-reversion rules and makes one
structural change: the universe expands from three broad ETFs to include the
nine long-history Select Sector SPDR ETFs. Rules are fixed on 2010-2017;
2018-2025 remains sealed unless every development gate passes.
"""

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22


VERSION = "V2.4"
VARIANT = "DIVERSIFIED_ETF_MEAN_REVERSION"
SYMBOLS = ("SPY", "QQQ", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
FAMILY = v22.FAMILY
RESULTS_DIR = Path("backtest_results_v24")


def load_data():
    """Reuse cached broad ETFs and fetch only missing daily sector histories."""
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
        data[symbol] = v21.add_indicators(frame)
    return data


def build_signals(data, start, end):
    signals = v22.build_signals(data, start, end)
    if signals.empty:
        return signals
    signals = signals.copy()
    signals["variant"] = VARIANT
    return signals


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
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v24_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v24_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.4 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v24_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "universe_size": len(SYMBOLS),
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v24_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.4 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
