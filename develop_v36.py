"""V3.6 blocked cross-asset dual-momentum screening.

All historical holdouts have been opened, so this module explicitly does not
claim untouched validation. It applies one fixed cross-asset rule from 2007-
2025 and reports five blocked regime folds. Monthly candidates must have
positive six-month momentum above a rising 200-day EMA; the single position
is the highest momentum-to-ATR-percent asset. A screening pass is only
eligible for future forward paper validation.
"""

from dataclasses import asdict
from pathlib import Path
import os
import time

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v32 as v32


VERSION = "V3.6"
VARIANT = "BLOCKED_CROSS_ASSET_DUAL_MOMENTUM"
SYMBOLS = ("SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ")
START = pd.Timestamp("2007-01-03")
END = pd.Timestamp("2025-12-31")
FOLDS = (("2007-01-03", "2010-12-31"), ("2011-01-03", "2014-12-31"),
         ("2015-01-02", "2018-12-31"), ("2019-01-02", "2022-12-30"),
         ("2023-01-03", "2025-12-31"))
FAMILY = v21.Family("CROSS_ASSET_DUAL_MOMENTUM", 3.0, 100.0, 20)
CACHE_DIR = Path("backtest_data_v36_history")
RESULTS_DIR = Path("backtest_results_v36")


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
                pieces.append(v32.fetch_window(symbol, start, end, key)); time.sleep(8)
            frame = pd.concat(pieces).drop_duplicates("date").sort_values("date").reset_index(drop=True)
            frame.to_csv(path, index=False)
        frame = v21.add_indicators(frame)
        frame["momentum126"] = frame["close"] / frame["close"].shift(v32.MOMENTUM_SESSIONS) - 1
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
            if idx < v32.EMA_SLOPE_SESSIONS:
                continue
            row = frame.loc[idx]; prior_ema = frame.loc[idx - v32.EMA_SLOPE_SESSIONS, "ema200"]
            if pd.isna(row["momentum126"]) or pd.isna(row["atr_pct"]) or pd.isna(prior_ema):
                continue
            if row["close"] > row["ema200"] > prior_ema and row["momentum126"] > 0 and row["atr_pct"] > 0:
                rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                             "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
                             "strength": float(row["momentum126"] / row["atr_pct"])})
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength"])


def fixed_year_stats(trades, start, end):
    years = list(range(start.year, end.year + 1)); work = trades.copy()
    if work.empty:
        return 0, len(years)
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    return int((work.groupby("year")["trade_pnl"].sum().reindex(years, fill_value=0) > 0).sum()), len(years)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); data = load_data()
    signals = build_signals(data, START, END); trades, skips = v21.run_account(signals, data, FAMILY)
    summary = v21.summarize(trades, skips, "VALIDATION"); positive_years, years = fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years, **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v36_signals.csv", index=False); trades.to_csv(RESULTS_DIR / "v36_trades.csv", index=False)
    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        fold_start, fold_end = pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        fold_signals = build_signals(data, fold_start, fold_end)
        fold_trades, fold_skips = v21.run_account(fold_signals, data, FAMILY)
        fold = v21.summarize(fold_trades, fold_skips, "VALIDATION")
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows); fold_frame.to_csv(RESULTS_DIR / "v36_folds.csv", index=False)
    positive_folds = int((fold_frame["account_return_pct"] > 0).sum())
    summary["positive_folds"] = positive_folds; summary["folds_tested"] = len(FOLDS)
    summary["screening_pass"] = bool(summary["trades"] >= 100
        and summary["bootstrap_mean_95_ci_low_pct"] > 0 and summary["profit_factor"] >= 1.50
        and summary["account_max_drawdown_pct"] >= -25 and summary["max_ticker_profit_contribution_pct"] <= 60
        and summary["max_month_profit_contribution_pct"] <= 35 and positive_years >= 15
        and positive_folds >= 4 and fold_frame["account_max_drawdown_pct"].min() >= -25)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v36_screening_summary.csv", index=False)
    pd.DataFrame([{"status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
                   "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"] else "BLOCKED_SCREENING_GATES_FAILED"}]).to_csv(RESULTS_DIR / "v36_candidate.csv", index=False)
    print("V3.6 BLOCKED SCREENING RESULT"); print(pd.DataFrame([summary]).to_string(index=False)); print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
