"""V2.6 sector-breadth-filtered ETF mean-reversion research.

V2.4 showed that trading every sector diluted the mean-reversion edge, while
also making long-history sector data available. V2.6 uses those sectors only
as an independent breadth regime: at least five of nine sectors must be above
their 200-day EMA. Actual candidates remain SPY, QQQ, and IWM under the frozen
V2.2 rules. Development is 2010-2017; 2018-2025 remains sealed unless every
development gate passes.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22
import develop_v24 as v24


VERSION = "V2.6"
VARIANT = "SECTOR_BREADTH_MEAN_REVERSION"
SECTOR_SYMBOLS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
MIN_SECTORS_ABOVE_EMA200 = 5
FAMILY = v22.FAMILY
RESULTS_DIR = Path("backtest_results_v26")


def breadth_frame(data):
    columns = []
    for symbol in SECTOR_SYMBOLS:
        frame = data[symbol].set_index("date")
        columns.append((frame["close"] > frame["ema200"]).rename(symbol))
    breadth = pd.concat(columns, axis=1).dropna()
    breadth["sectors_above_ema200"] = breadth.sum(axis=1)
    breadth["risk_on"] = breadth["sectors_above_ema200"] >= MIN_SECTORS_ABOVE_EMA200
    return breadth


def build_signals(data, start, end):
    broad_data = {symbol: data[symbol] for symbol in v21.SYMBOLS}
    signals = v22.build_signals(broad_data, start, end)
    if signals.empty:
        return signals
    breadth = breadth_frame(data)
    allowed = set(pd.to_datetime(breadth.index[breadth["risk_on"]]).normalize())
    filtered = signals.loc[pd.to_datetime(signals["signal_time"]).dt.normalize().isin(allowed)].copy()
    filtered["variant"] = VARIANT
    filtered["min_sectors_above_ema200"] = MIN_SECTORS_ABOVE_EMA200
    return filtered.reset_index(drop=True)


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
    data = v24.load_data()
    dev_signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_trades, dev_skips = v21.run_account(dev_signals, data, FAMILY)
    dev_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "sector_count": len(SECTOR_SYMBOLS),
        "min_sectors_above_ema200": MIN_SECTORS_ABOVE_EMA200,
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v26_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v26_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.6 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v26_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "sector_count": len(SECTOR_SYMBOLS),
        "min_sectors_above_ema200": MIN_SECTORS_ABOVE_EMA200,
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v26_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.6 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
