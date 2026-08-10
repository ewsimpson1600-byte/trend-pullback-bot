"""V5.3 historical robustness check for the frozen V5.2 candidate.

V5.2 passed its predeclared 2024-2025 fast screen. This stage applies the
exact same three-lower-closes rules to 2010-2023 and four fixed chronological
folds. Every historical window has already been exposed by earlier research,
so this is robustness screening only, never validation. No parameter may be
changed from the V5.2 candidate based on these results.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import screen_v52 as v52


VERSION = "V5.3"
VARIANT = v52.VARIANT
FAMILY = v52.FAMILY
SYMBOLS = v52.SYMBOLS
START = pd.Timestamp("2010-01-04")
END = pd.Timestamp("2023-12-29")
FOLDS = (
    (pd.Timestamp("2010-01-04"), pd.Timestamp("2013-12-31")),
    (pd.Timestamp("2014-01-02"), pd.Timestamp("2017-12-29")),
    (pd.Timestamp("2018-01-02"), pd.Timestamp("2020-12-31")),
    (pd.Timestamp("2021-01-04"), pd.Timestamp("2023-12-29")),
)

# Frozen before the robustness data are evaluated.
MIN_TRADES = 100
MIN_PROFIT_FACTOR = 1.50
MAX_DRAWDOWN_PCT = -25.0
MAX_TICKER_CONTRIBUTION_PCT = 60.0
MAX_MONTH_CONTRIBUTION_PCT = 35.0
MIN_POSITIVE_YEARS = 10
MIN_POSITIVE_FOLDS = 3
RESULTS_DIR = Path("robustness_results_v53")


def run_period(data, start, end):
    signals = v52.build_signals(data, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = v21.summarize(trades, skips, "HISTORICAL_ROBUSTNESS")
    return signals, trades, skips, summary


def robustness_pass(summary, fold_frame):
    return bool(
        summary["trades"] >= MIN_TRADES
        and summary["bootstrap_mean_95_ci_low_pct"] > 0
        and summary["profit_factor"] >= MIN_PROFIT_FACTOR
        and summary["account_max_drawdown_pct"] >= MAX_DRAWDOWN_PCT
        and summary["max_ticker_profit_contribution_pct"] <= MAX_TICKER_CONTRIBUTION_PCT
        and summary["max_month_profit_contribution_pct"] <= MAX_MONTH_CONTRIBUTION_PCT
        and summary["positive_years"] >= MIN_POSITIVE_YEARS
        and int((fold_frame["account_return_pct"] > 0).sum()) >= MIN_POSITIVE_FOLDS
        and fold_frame["account_max_drawdown_pct"].min() >= MAX_DRAWDOWN_PCT
    )


def save_breakdowns(trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field in ("symbol", "month", "year", "exit_reason"):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(
            RESULTS_DIR / f"v53_by_{field}.csv"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    fold_rows = []
    for number, (start, end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(data, start, end)
        fold_rows.append({"fold": number, "start": start.date().isoformat(),
                          "end": end.date().isoformat(), **fold})
    folds = pd.DataFrame(fold_rows)
    summary.update({
        "version": VERSION,
        "variant": VARIANT,
        "robustness_only": True,
        "start": START.date().isoformat(),
        "end": END.date().isoformat(),
        "universe_size": len(SYMBOLS),
        "consecutive_lower_closes": v52.CONSECUTIVE_LOWER_CLOSES,
        "positive_folds": int((folds["account_return_pct"] > 0).sum()),
        "folds_tested": len(FOLDS),
        **asdict(FAMILY),
    })
    summary["robustness_pass"] = robustness_pass(summary, folds)
    summary["next_stage"] = (
        "FORWARD_ONLY_VALIDATION" if summary["robustness_pass"]
        else "REJECTED_AFTER_HISTORICAL_ROBUSTNESS"
    )
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v53_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v53_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v53_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v53_skips.csv", index=False)
    folds.to_csv(RESULTS_DIR / "v53_folds.csv", index=False)
    save_breakdowns(trades)
    print("V5.3 FROZEN HISTORICAL ROBUSTNESS")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
