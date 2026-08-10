"""V4.0 blocked robustness screen of the locked V3.4 rule.

V3.4 was the strongest diversified trend candidate before the historical
holdouts were consumed, but missed its development year-consistency gate by
one year.  This module changes no V3.4 signal, selection, sizing, or exit rule.
It applies that already-defined alternating SPY/sector rule to the same five
fixed blocks used by V3.6-V3.9.  Results are screening evidence only and can
only nominate the unchanged rule for new forward paper validation.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v34 as v34
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V4.0"
VARIANT = "BLOCKED_UNCHANGED_V34_ALTERNATING_TREND"
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = v34.FAMILY
SYMBOLS = v34.SYMBOLS
RESULTS_DIR = Path("backtest_results_v40")


def run_period(data, start, end):
    signals = v34.build_signals(data, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v34.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = v37.fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "source_rule": v34.VERSION, "positive_years": positive_years,
                    "years_tested": years, "universe_size": len(SYMBOLS), **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v40_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v40_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v40_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(
            data, pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        )
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v40_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v40_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v40_candidate.csv", index=False)
    print("V4.0 BLOCKED ROBUSTNESS RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
