"""V4.5 blocked cross-asset volume-breakout screening.

V4.4 showed that the original trend-pullback family did not generalize across
asset classes. V4.5 applies the final unchanged V2.1 entry family to the fixed
V3.6 universe: a 20-session breakout above a rising long-term trend with at
least 1.2x prior average volume. Stops, targets, holding period, sizing, and
costs remain frozen. Historical results are screening only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V4.5"
VARIANT = "BLOCKED_CROSS_ASSET_VOLUME_BREAKOUT"
SYMBOLS = v36.SYMBOLS
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = next(item for item in v21.FAMILIES if item.name == "VOLUME_BREAKOUT")
RESULTS_DIR = Path("backtest_results_v45")


def run_period(data, start, end):
    signals = v21.build_signals(data, FAMILY, start, end)
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
    signals.to_csv(RESULTS_DIR / "v45_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v45_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v45_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(data, pd.Timestamp(raw_start), pd.Timestamp(raw_end))
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v45_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v45_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v45_candidate.csv", index=False)
    print("V4.5 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
