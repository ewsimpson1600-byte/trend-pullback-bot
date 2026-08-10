"""V3.2 broad-index dual-momentum with a new historical holdout.

The original 2018-2025 holdout was opened by V3.1 and is never treated as
untouched again. V3.2 locks a new diversified rule on 2010-2025, then opens
the previously unused 2002-2009 dot-com/financial-crisis history once. Each
month it ranks SPY, QQQ, and IWM by six-month momentum divided by ATR percent,
while requiring positive momentum and a rising 200-day EMA. The simulator is
$1,000 cash-only, integer-share, costed, gap-aware, and research-only.
"""

from dataclasses import asdict
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import requests

import develop_v21 as v21
import develop_v25 as v25


VERSION = "V3.2"
VARIANT = "BROAD_INDEX_RISK_ADJUSTED_DUAL_MOMENTUM"
SYMBOLS = ("SPY", "QQQ", "IWM")
DOWNLOAD_WINDOWS = (("2000-01-01", "2013-01-01"), ("2013-01-01", "2026-01-01"))
DEVELOPMENT_START = pd.Timestamp("2010-01-04")
DEVELOPMENT_END = pd.Timestamp("2025-12-31")
VALIDATION_START = pd.Timestamp("2002-01-02")
VALIDATION_END = pd.Timestamp("2009-12-31")
MOMENTUM_SESSIONS = 126
EMA_SLOPE_SESSIONS = 20
FAMILY = v21.Family("BROAD_INDEX_DUAL_MOMENTUM", 3.0, 100.0, 20)
CACHE_DIR = Path("backtest_data_v32_history")
RESULTS_DIR = Path("backtest_results_v32")


def fetch_window(symbol, start, end, api_key):
    for attempt in range(5):
        response = requests.get(
            v21.API_URL,
            params={"symbol": symbol, "interval": "1day", "start_date": start, "end_date": end,
                    "timezone": "America/New_York", "apikey": api_key, "format": "JSON",
                    "order": "ASC", "outputsize": 5000},
            timeout=60,
        )
        if response.status_code == 429:
            time.sleep(min(30 * 2 ** attempt, 300))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise RuntimeError(f"Twelve Data error for {symbol}: {payload}")
        frame = pd.DataFrame(payload.get("values") or [])
        if frame.empty:
            raise RuntimeError(f"No data for {symbol} in {start}:{end}")
        frame["date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[["date", "open", "high", "low", "close", "volume"]].dropna()
    raise RuntimeError(f"Rate limit persisted for {symbol} in {start}:{end}")


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
            for start, end in DOWNLOAD_WINDOWS:
                pieces.append(fetch_window(symbol, start, end, key))
                time.sleep(8)
            frame = pd.concat(pieces).drop_duplicates("date").sort_values("date").reset_index(drop=True)
            frame.to_csv(path, index=False)
        frame = v21.add_indicators(frame)
        frame["momentum126"] = frame["close"] / frame["close"].shift(MOMENTUM_SESSIONS) - 1
        frame["atr_pct"] = frame["atr14"] / frame["close"]
        data[symbol] = frame
    return data


def build_signals(data, start, end):
    rows = []
    for date in v25.monthly_signal_dates(data["SPY"], start, end):
        for symbol, frame in data.items():
            matches = frame.index[frame["date"] == date]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            if idx < EMA_SLOPE_SESSIONS:
                continue
            row = frame.loc[idx]
            prior_ema = frame.loc[idx - EMA_SLOPE_SESSIONS, "ema200"]
            if pd.isna(row["momentum126"]) or pd.isna(row["atr_pct"]) or pd.isna(prior_ema):
                continue
            if not (row["close"] > row["ema200"] > prior_ema and row["momentum126"] > 0 and row["atr_pct"] > 0):
                continue
            rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                         "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
                         "strength": float(row["momentum126"] / row["atr_pct"])})
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength"])


def summarize_period(trades, skips, period, start, end):
    result = v21.summarize(trades, skips, "DEVELOPMENT" if period == "DEVELOPMENT" else "VALIDATION")
    years = list(range(start.year, end.year + 1))
    if trades.empty:
        positive_years = 0
    else:
        work = trades.copy()
        work["year"] = pd.to_datetime(work["entry_time"]).dt.year
        positive_years = int((work.groupby("year")["trade_pnl"].sum().reindex(years, fill_value=0) > 0).sum())
    result.update({"period": period, "positive_years": positive_years, "years_tested": len(years)})
    if period == "DEVELOPMENT":
        result["pass"] = bool(result["trades"] >= 80 and result.get("account_return_pct", 0) > 0
                              and result.get("profit_factor", 0) >= 1.15
                              and result.get("account_max_drawdown_pct", -100) >= -25
                              and positive_years >= 12)
    else:
        result["pass"] = bool(result["trades"] >= 50 and result.get("bootstrap_mean_95_ci_low_pct", -100) > 0
                              and result.get("profit_factor", 0) >= 1.50
                              and result.get("account_max_drawdown_pct", -100) >= -25
                              and result.get("max_ticker_profit_contribution_pct", 100) <= 60
                              and result.get("max_month_profit_contribution_pct", 100) <= 35
                              and positive_years >= 7)
    return result


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
    signals = build_signals(data, DEVELOPMENT_START, DEVELOPMENT_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    dev = {"version": VERSION, "variant": VARIANT, "momentum_sessions": MOMENTUM_SESSIONS,
           **asdict(FAMILY), **summarize_period(trades, skips, "DEVELOPMENT", DEVELOPMENT_START, DEVELOPMENT_END)}
    write_period("v32_development", signals, trades, skips, dev)
    if not dev["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "NEW_DEVELOPMENT_GATES_FAILED", **dev}]).to_csv(RESULTS_DIR / "v32_candidate.csv", index=False)
        print(pd.DataFrame([dev]).to_string(index=False)); print("V3.2 failed development; historical holdout stayed sealed."); return
    pd.DataFrame([{"status": "LOCKED", **dev}]).to_csv(RESULTS_DIR / "v32_candidate.csv", index=False)
    signals = build_signals(data, VALIDATION_START, VALIDATION_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    val = {"version": VERSION, "variant": VARIANT, "momentum_sessions": MOMENTUM_SESSIONS,
           **asdict(FAMILY), **summarize_period(trades, skips, "HISTORICAL_VALIDATION", VALIDATION_START, VALIDATION_END)}
    write_period("v32_historical_validation", signals, trades, skips, val)
    print("V3.2 LOCKED HISTORICAL VALIDATION RESULT"); print(pd.DataFrame([val]).to_string(index=False))


if __name__ == "__main__":
    main()
