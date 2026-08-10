"""V3.9 blocked cross-asset volatility-expansion screening.

This precommitted family buys a 20-session closing breakout only when the
previous session's Bollinger bandwidth was in its lowest trailing quintile
and price is above a rising 200-day EMA.  It tests post-compression volatility
expansion rather than retuning momentum, mean reversion, or regime switching.
All historical results are screening only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V3.9"
VARIANT = "BLOCKED_CROSS_ASSET_VOLATILITY_EXPANSION"
SYMBOLS = v36.SYMBOLS
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
BREAKOUT_SESSIONS = 20
COMPRESSION_LOOKBACK = 252
COMPRESSION_QUANTILE = 0.20
FAMILY = v21.Family("VOLATILITY_EXPANSION", 2.0, 4.0, 20)
RESULTS_DIR = Path("backtest_results_v39")


def add_expansion_indicators(frame):
    x = frame.copy()
    x["prior_high20_expansion"] = x["high"].shift(1).rolling(BREAKOUT_SESSIONS).max()
    bandwidth = 4.0 * x["std20"] / x["sma20"]
    prior_bandwidth = bandwidth.shift(1)
    x["prior_bandwidth"] = prior_bandwidth
    x["compression_cutoff"] = prior_bandwidth.rolling(COMPRESSION_LOOKBACK).quantile(
        COMPRESSION_QUANTILE
    )
    x["prior_ema200"] = x["ema200"].shift(v36.v32.EMA_SLOPE_SESSIONS)
    return x


def build_signals(data, start, end):
    rows = []
    for symbol, raw in data.items():
        frame = add_expansion_indicators(raw)
        ready = (
            frame["date"].between(start, end)
            & frame[["atr14", "ema200", "prior_ema200", "prior_high20_expansion",
                     "prior_bandwidth", "compression_cutoff"]].notna().all(axis=1)
        )
        valid = (
            ready
            & (frame["prior_bandwidth"] <= frame["compression_cutoff"])
            & (frame["close"] > frame["prior_high20_expansion"])
            & (frame["close"] > frame["ema200"])
            & (frame["ema200"] > frame["prior_ema200"])
        )
        strength = (
            (frame["close"] / frame["prior_high20_expansion"] - 1) * 100
            + (frame["compression_cutoff"] / frame["prior_bandwidth"] - 1)
        )
        for idx in frame.index[valid]:
            rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                         "signal_idx": int(idx), "signal_time": frame.at[idx, "date"],
                         "atr": float(frame.at[idx, "atr14"]),
                         "strength": float(strength.at[idx])})
    columns = ["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["signal_time", "strength", "symbol"], ascending=[True, False, True]
    ).reset_index(drop=True)


def run_period(data, start, end):
    signals = build_signals(data, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v36.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = v37.fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years,
                    "breakout_sessions": BREAKOUT_SESSIONS,
                    "compression_lookback": COMPRESSION_LOOKBACK,
                    "compression_quantile": COMPRESSION_QUANTILE, **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v39_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v39_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v39_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, fold_skips, fold = run_period(
            data, pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        )
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v39_folds.csv", index=False)
    positive_folds = int((fold_frame["account_return_pct"] > 0).sum())
    summary["positive_folds"] = positive_folds
    summary["folds_tested"] = len(FOLDS)
    summary["screening_pass"] = bool(
        summary["trades"] >= 100
        and summary["bootstrap_mean_95_ci_low_pct"] > 0
        and summary["profit_factor"] >= 1.50
        and summary["account_max_drawdown_pct"] >= -25
        and summary["max_ticker_profit_contribution_pct"] <= 60
        and summary["max_month_profit_contribution_pct"] <= 35
        and positive_years >= 15
        and positive_folds >= 4
        and fold_frame["account_max_drawdown_pct"].min() >= -25
    )
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v39_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v39_candidate.csv", index=False)
    print("V3.9 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
