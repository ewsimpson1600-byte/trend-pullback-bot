"""V4.1 blocked low-volatility core/sector trend screening.

V4.0 confirmed that V3.4's diversified architecture had positive aggregate
expectancy but weak early-regime loss quality.  This precommitted structural
test retains alternating SPY and sector slots and every frozen trend, sizing,
and exit rule.  In sector slots it replaces usage rotation with the eligible
sector having the lowest ATR percentage, a defensive low-volatility selection
principle that does not rank or optimize past returns.  Historical output is
screening only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v34 as v34
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V4.1"
VARIANT = "BLOCKED_LOW_VOLATILITY_CORE_SECTOR_TREND"
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = v34.FAMILY
SYMBOLS = v34.SYMBOLS
RESULTS_DIR = Path("backtest_results_v41")


def build_signals(data, start, end):
    rows = []
    for slot_index, date in enumerate(v25.monthly_signal_dates(data["SPY"], start, end)):
        if slot_index % 2 == 0:
            result = v34.v33.eligible_on_date(data["SPY"], date)
            if result is not None:
                idx, row = result
                signal = v34.make_signal("SPY", idx, row, date, "SPY_CORE")
                signal["variant"] = VARIANT
                rows.append(signal)
            continue
        eligible = []
        for symbol in v34.SECTOR_SYMBOLS:
            result = v34.v33.eligible_on_date(data[symbol], date)
            if result is None:
                continue
            idx, row = result
            atr_pct = float(row["atr14"] / row["close"])
            eligible.append((atr_pct, symbol, idx, row))
        if eligible:
            _, symbol, idx, row = min(eligible, key=lambda item: (item[0], item[1]))
            signal = v34.make_signal(symbol, idx, row, date, "LOW_VOL_SECTOR")
            signal["variant"] = VARIANT
            signal["atr_pct"] = float(row["atr14"] / row["close"])
            rows.append(signal)
    columns = ["family", "variant", "symbol", "signal_idx", "signal_time", "atr",
               "strength", "slot", "atr_pct"]
    return pd.DataFrame(rows).reindex(columns=columns)


def run_period(data, start, end):
    signals = build_signals(data, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v34.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = v37.fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years,
                    "universe_size": len(SYMBOLS), **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v41_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v41_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v41_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(data, pd.Timestamp(raw_start), pd.Timestamp(raw_end))
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v41_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v41_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v41_candidate.csv", index=False)
    print("V4.1 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
