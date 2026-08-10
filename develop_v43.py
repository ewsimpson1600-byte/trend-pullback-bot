"""V4.3 blocked canonical global dual-momentum screening.

V4.2 showed that forcing equal cross-asset exposure destroyed expectancy.
V4.3 changes family to a canonical monthly global dual-momentum rule: choose
the stronger of U.S. and international equities when its trailing 12-month
return is positive; otherwise use intermediate Treasuries when their trailing
12-month return is positive; otherwise stay in cash.  The lookback and asset
roles are predeclared, with no grid search. Historical results are screening
only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V4.3"
VARIANT = "BLOCKED_CANONICAL_GLOBAL_DUAL_MOMENTUM"
EQUITIES = ("SPY", "EFA")
DEFENSIVE = "IEF"
SYMBOLS = EQUITIES + (DEFENSIVE,)
MOMENTUM_SESSIONS = 252
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
FAMILY = v21.Family("GLOBAL_DUAL_MOMENTUM", 3.0, 100.0, 20)
RESULTS_DIR = Path("backtest_results_v43")


def add_momentum(data):
    enriched = {}
    for symbol in SYMBOLS:
        frame = data[symbol].copy()
        frame["momentum252"] = frame["close"] / frame["close"].shift(MOMENTUM_SESSIONS) - 1
        enriched[symbol] = frame
    return enriched


def signal_for(frame, symbol, date, momentum, role):
    matches = frame.index[frame["date"] == date]
    if len(matches) == 0:
        return None
    idx = int(matches[0])
    row = frame.loc[idx]
    if pd.isna(row["atr14"]):
        return None
    return {"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
            "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
            "strength": float(momentum), "role": role}


def build_signals(data, start, end):
    rows = []
    for date in v25.monthly_signal_dates(data["SPY"], start, end):
        equity = []
        for symbol in EQUITIES:
            matches = data[symbol].index[data[symbol]["date"] == date]
            if len(matches):
                momentum = data[symbol].at[int(matches[0]), "momentum252"]
                if pd.notna(momentum):
                    equity.append((float(momentum), symbol))
        if equity:
            momentum, symbol = max(equity, key=lambda item: (item[0], item[1]))
            if momentum > 0:
                signal = signal_for(data[symbol], symbol, date, momentum, "EQUITY")
                if signal:
                    rows.append(signal)
                continue
        matches = data[DEFENSIVE].index[data[DEFENSIVE]["date"] == date]
        if len(matches):
            momentum = data[DEFENSIVE].at[int(matches[0]), "momentum252"]
            if pd.notna(momentum) and float(momentum) > 0:
                signal = signal_for(data[DEFENSIVE], DEFENSIVE, date, float(momentum), "DEFENSIVE")
                if signal:
                    rows.append(signal)
    columns = ["family", "variant", "symbol", "signal_idx", "signal_time", "atr",
               "strength", "role"]
    return pd.DataFrame(rows, columns=columns)


def run_period(data, start, end):
    signals = build_signals(data, start, end)
    trades, skips = v21.run_account(signals, data, FAMILY)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = add_momentum(v36.load_data())
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = v37.fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years,
                    "momentum_sessions": MOMENTUM_SESSIONS, "universe_size": len(SYMBOLS),
                    **asdict(FAMILY)})
    signals.to_csv(RESULTS_DIR / "v43_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v43_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v43_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, _, fold = run_period(data, pd.Timestamp(raw_start), pd.Timestamp(raw_end))
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v43_folds.csv", index=False)
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v43_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v43_candidate.csv", index=False)
    print("V4.3 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
