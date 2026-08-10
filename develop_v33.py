"""V3.3 diversified sector absolute-trend rotation.

V3.2's risk-adjusted return ranking concentrated exposure. V3.3 removes
return chasing: once monthly, each legacy sector ETF independently qualifies
only with positive six-month momentum above a rising 200-day EMA, then the
least-used eligible sector is selected. This equal-opportunity rule satisfies
the one-position constraint without pruning tickers. It develops on 2010-2025
before one possible opening of the still-unused 2002-2009 stress holdout.
"""

from dataclasses import asdict
from pathlib import Path
import os
import time

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v32 as v32


VERSION = "V3.3"
VARIANT = "DIVERSIFIED_SECTOR_ABSOLUTE_TREND"
SYMBOLS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
FAMILY = v21.Family("SECTOR_ABSOLUTE_TREND", 3.0, 100.0, 20)
CACHE_DIR = Path("backtest_data_v33_history")
RESULTS_DIR = Path("backtest_results_v33")


def load_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = os.getenv("TWELVE_DATA_API_KEY")
    data = {}
    for symbol in SYMBOLS:
        path = CACHE_DIR / f"{symbol}_2000_2025.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
        else:
            if not key:
                raise RuntimeError("TWELVE_DATA_API_KEY is missing")
            pieces = []
            for start, end in v32.DOWNLOAD_WINDOWS:
                pieces.append(v32.fetch_window(symbol, start, end, key))
                time.sleep(8)
            frame = pd.concat(pieces).drop_duplicates("date").sort_values("date").reset_index(drop=True)
            frame.to_csv(path, index=False)
        frame = v21.add_indicators(frame)
        frame["momentum126"] = frame["close"] / frame["close"].shift(v32.MOMENTUM_SESSIONS) - 1
        data[symbol] = frame
    return data


def eligible_on_date(frame, date):
    matches = frame.index[frame["date"] == date]
    if len(matches) == 0:
        return None
    idx = int(matches[0])
    if idx < v32.EMA_SLOPE_SESSIONS:
        return None
    row = frame.loc[idx]
    prior_ema = frame.loc[idx - v32.EMA_SLOPE_SESSIONS, "ema200"]
    if pd.isna(row["momentum126"]) or pd.isna(row["atr14"]) or pd.isna(prior_ema):
        return None
    if not (row["close"] > row["ema200"] > prior_ema and row["momentum126"] > 0):
        return None
    return idx, row


def build_signals(data, start, end):
    usage = {symbol: 0 for symbol in SYMBOLS}
    rows = []
    calendar = data["XLB"]
    for date in v25.monthly_signal_dates(calendar, start, end):
        eligible = []
        for symbol in SYMBOLS:
            result = eligible_on_date(data[symbol], date)
            if result is not None:
                eligible.append((usage[symbol], symbol, result))
        if not eligible:
            continue
        _, symbol, (idx, row) = min(eligible, key=lambda item: (item[0], item[1]))
        usage[symbol] += 1
        rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                     "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
                     "strength": float(-usage[symbol]), "selection_count": usage[symbol]})
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength", "selection_count"])


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
    signals = build_signals(data, v32.DEVELOPMENT_START, v32.DEVELOPMENT_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    dev = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY),
           **v32.summarize_period(trades, skips, "DEVELOPMENT", v32.DEVELOPMENT_START, v32.DEVELOPMENT_END)}
    write_period("v33_development", signals, trades, skips, dev)
    if not dev["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "NEW_DEVELOPMENT_GATES_FAILED", **dev}]).to_csv(RESULTS_DIR / "v33_candidate.csv", index=False)
        print(pd.DataFrame([dev]).to_string(index=False)); print("V3.3 failed development; historical holdout stayed sealed."); return
    pd.DataFrame([{"status": "LOCKED", **dev}]).to_csv(RESULTS_DIR / "v33_candidate.csv", index=False)
    signals = build_signals(data, v32.VALIDATION_START, v32.VALIDATION_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    val = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY),
           **v32.summarize_period(trades, skips, "HISTORICAL_VALIDATION", v32.VALIDATION_START, v32.VALIDATION_END)}
    write_period("v33_historical_validation", signals, trades, skips, val)
    print("V3.3 LOCKED HISTORICAL VALIDATION RESULT"); print(pd.DataFrame([val]).to_string(index=False))


if __name__ == "__main__":
    main()
