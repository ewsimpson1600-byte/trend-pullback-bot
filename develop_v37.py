"""V3.7 blocked cross-asset mean-reversion screening.

V3.6 showed that a diversified momentum ranking did not produce a reliable
return distribution.  This is one precommitted, structurally different test:
the original V2.1 oversold-rebound rule is applied unchanged to a fixed
cross-asset universe and evaluated in the same five non-overlapping blocks.
All historical holdouts are consumed, so a pass is screening evidence only
and can at most nominate the rule for future forward paper validation.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v36 as v36


VERSION = "V3.7"
VARIANT = "BLOCKED_CROSS_ASSET_MEAN_REVERSION"
SYMBOLS = v36.SYMBOLS
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = next(item for item in v21.FAMILIES if item.name == "MEAN_REVERSION")
RESULTS_DIR = Path("backtest_results_v37")


def fixed_year_stats(trades, start, end):
    years = list(range(start.year, end.year + 1))
    if trades.empty:
        return 0, len(years)
    work = trades.copy()
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    positive = (work.groupby("year")["trade_pnl"].sum().reindex(years, fill_value=0) > 0).sum()
    return int(positive), len(years)


def run_period(data, start, end):
    signals = v21.build_signals(data, FAMILY, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v36.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years, **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v37_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v37_trades.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        fold_start, fold_end = pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        _, fold_trades, _, fold = run_period(data, fold_start, fold_end)
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v37_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v37_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"] else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v37_candidate.csv", index=False)
    print("V3.7 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
