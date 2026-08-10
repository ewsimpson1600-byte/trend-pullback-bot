"""V4.2 blocked equal-opportunity cross-asset trend screening.

Return ranking in V3.6 and low-volatility sector selection in V4.1 were not
regime robust.  V4.2 removes both preferences: each month, among assets with
positive six-month momentum above a rising 200-day EMA, it selects the least-
used eligible asset.  This fixed exposure-balancing rule uses the unchanged
V3.6 universe, trend eligibility, sizing, and exits.  Historical results are
screening only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v33 as v33
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V4.2"
VARIANT = "BLOCKED_EQUAL_OPPORTUNITY_CROSS_ASSET_TREND"
SYMBOLS = v36.SYMBOLS
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = v36.FAMILY
RESULTS_DIR = Path("backtest_results_v42")


def build_signals(data, start, end):
    usage = {symbol: 0 for symbol in SYMBOLS}
    rows = []
    for date in v25.monthly_signal_dates(data["SPY"], start, end):
        eligible = []
        for symbol in SYMBOLS:
            result = v33.eligible_on_date(data[symbol], date)
            if result is not None:
                eligible.append((usage[symbol], symbol, result))
        if not eligible:
            continue
        _, symbol, (idx, row) = min(eligible, key=lambda item: (item[0], item[1]))
        usage[symbol] += 1
        rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                     "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
                     "strength": float(-usage[symbol]), "selection_count": usage[symbol]})
    columns = ["family", "variant", "symbol", "signal_idx", "signal_time", "atr",
               "strength", "selection_count"]
    return pd.DataFrame(rows, columns=columns)


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
                    "universe_size": len(SYMBOLS), **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v42_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v42_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v42_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(data, pd.Timestamp(raw_start), pd.Timestamp(raw_end))
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v42_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v42_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v42_candidate.csv", index=False)
    print("V4.2 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
