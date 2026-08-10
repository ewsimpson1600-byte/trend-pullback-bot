"""V3.5 equity-core, sector, and Treasury trend rotation.

V3.4 cleared return, profit-factor, confidence, drawdown, and concentration
development requirements but missed annual consistency by one year. V3.5
adds an independent Treasury trend slot to the existing alternating framework:
monthly slots cycle SPY, diversified sector, and diversified Treasury. Assets
still require positive six-month momentum above a rising 200-day EMA, and
only one position may exist. Development is 2010-2025; the unused 2003-2009
stress holdout opens once only if development qualifies.
"""

from dataclasses import asdict
from pathlib import Path
import os
import time

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v32 as v32
import develop_v33 as v33


VERSION = "V3.5"
VARIANT = "EQUITY_SECTOR_TREASURY_TREND_ROTATION"
SECTOR_SYMBOLS = v33.SYMBOLS
TREASURY_SYMBOLS = ("IEF", "TLT")
SYMBOLS = ("SPY",) + SECTOR_SYMBOLS + TREASURY_SYMBOLS
VALIDATION_START = pd.Timestamp("2003-01-02")
VALIDATION_END = v32.VALIDATION_END
FAMILY = v21.Family("THREE_SLEEVE_TREND", 3.0, 100.0, 20)
CACHE_DIR = Path("backtest_data_v35_treasury_history")
RESULTS_DIR = Path("backtest_results_v35")


def load_treasuries():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = os.getenv("TWELVE_DATA_API_KEY")
    data = {}
    for symbol in TREASURY_SYMBOLS:
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


def load_data():
    broad = v32.load_data()
    return {"SPY": broad["SPY"], **v33.load_data(), **load_treasuries()}


def make_signal(symbol, idx, row, date, slot):
    return {"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
            "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
            "strength": 1.0, "slot": slot}


def least_used_eligible(data, symbols, usage, date):
    eligible = []
    for symbol in symbols:
        result = v33.eligible_on_date(data[symbol], date)
        if result is not None:
            eligible.append((usage[symbol], symbol, result))
    if not eligible:
        return None
    _, symbol, result = min(eligible, key=lambda item: (item[0], item[1]))
    usage[symbol] += 1
    return symbol, result


def build_signals(data, start, end):
    sector_usage = {symbol: 0 for symbol in SECTOR_SYMBOLS}
    treasury_usage = {symbol: 0 for symbol in TREASURY_SYMBOLS}
    rows = []
    for slot_index, date in enumerate(v25.monthly_signal_dates(data["SPY"], start, end)):
        slot = slot_index % 3
        if slot == 0:
            result = v33.eligible_on_date(data["SPY"], date)
            if result is not None:
                idx, row = result
                rows.append(make_signal("SPY", idx, row, date, "SPY_CORE"))
        elif slot == 1:
            selected = least_used_eligible(data, SECTOR_SYMBOLS, sector_usage, date)
            if selected:
                symbol, (idx, row) = selected
                rows.append(make_signal(symbol, idx, row, date, "SECTOR"))
        else:
            selected = least_used_eligible(data, TREASURY_SYMBOLS, treasury_usage, date)
            if selected:
                symbol, (idx, row) = selected
                rows.append(make_signal(symbol, idx, row, date, "TREASURY"))
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength", "slot"])


def validation_summary(trades, skips):
    result = v32.summarize_period(trades, skips, "HISTORICAL_VALIDATION", VALIDATION_START, VALIDATION_END)
    result["pass"] = bool(result["trades"] >= 50 and result.get("bootstrap_mean_95_ci_low_pct", -100) > 0
                          and result.get("profit_factor", 0) >= 1.50
                          and result.get("account_max_drawdown_pct", -100) >= -25
                          and result.get("max_ticker_profit_contribution_pct", 100) <= 60
                          and result.get("max_month_profit_contribution_pct", 100) <= 35
                          and result["positive_years"] >= result["years_tested"] - 1)
    return result


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy(); work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m"); work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_{suffix}.csv")


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False); trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False); skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False); save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); data = load_data()
    signals = build_signals(data, v32.DEVELOPMENT_START, v32.DEVELOPMENT_END); trades, skips = v21.run_account(signals, data, FAMILY)
    dev = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY), **v32.summarize_period(trades, skips, "DEVELOPMENT", v32.DEVELOPMENT_START, v32.DEVELOPMENT_END)}
    write_period("v35_development", signals, trades, skips, dev)
    if not dev["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "NEW_DEVELOPMENT_GATES_FAILED", **dev}]).to_csv(RESULTS_DIR / "v35_candidate.csv", index=False)
        print(pd.DataFrame([dev]).to_string(index=False)); print("V3.5 failed development; historical holdout stayed sealed."); return
    pd.DataFrame([{"status": "LOCKED", **dev}]).to_csv(RESULTS_DIR / "v35_candidate.csv", index=False)
    signals = build_signals(data, VALIDATION_START, VALIDATION_END); trades, skips = v21.run_account(signals, data, FAMILY)
    val = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY), **validation_summary(trades, skips)}
    write_period("v35_historical_validation", signals, trades, skips, val)
    print("V3.5 LOCKED HISTORICAL VALIDATION RESULT"); print(pd.DataFrame([val]).to_string(index=False))


if __name__ == "__main__":
    main()
