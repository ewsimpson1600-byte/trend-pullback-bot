"""V3.0 dual-sleeve ETF portfolio research.

V3.0 combines two independently tested, diversifying share strategies without
stacking leverage: two-thirds of capital uses V2.6 sector-breadth mean
reversion and one-third uses V2.8 cross-asset breakout trend following. Each
sleeve retains its frozen signals, exits, costs, and 80% cash-use/2% planned
risk limits, so aggregate limits remain the same. Development is 2010-2017;
2018-2025 stays sealed unless every frozen development gate passes.
"""

from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22
import develop_v24 as v24
import develop_v26 as v26
import develop_v28 as v28


VERSION = "V3.0"
VARIANT = "DUAL_SLEEVE_MEAN_REVERSION_TREND"
MEAN_REVERSION_WEIGHT = 2 / 3
TREND_WEIGHT = 1 / 3
RESULTS_DIR = Path("backtest_results_v30")


def load_data():
    return {**v24.load_data(), **v28.load_data()}


def combine_sleeves(mean_trades, trend_trades):
    frames = []
    for sleeve, trades in (("MEAN_REVERSION", mean_trades), ("TREND", trend_trades)):
        if trades.empty:
            continue
        x = trades.copy()
        x["sleeve"] = sleeve
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["exit_time", "entry_time", "sleeve", "symbol"]
    ).reset_index(drop=True)
    combined["account_balance"] = v21.STARTING_ACCOUNT + combined["trade_pnl"].cumsum()
    peak = combined["account_balance"].cummax().clip(lower=v21.STARTING_ACCOUNT)
    combined["account_drawdown_pct"] = (combined["account_balance"] / peak - 1) * 100
    return combined


def run_portfolio(data, start, end):
    mean_signals = v26.build_signals(data, start, end)
    trend_signals = v28.build_signals(data, start, end)
    mean_trades, mean_skips = v21.run_account(
        mean_signals, data, v26.FAMILY, v21.STARTING_ACCOUNT * MEAN_REVERSION_WEIGHT
    )
    trend_trades, trend_skips = v21.run_account(
        trend_signals, data, v28.FAMILY, v21.STARTING_ACCOUNT * TREND_WEIGHT
    )
    combined = combine_sleeves(mean_trades, trend_trades)
    skips = pd.concat([mean_skips, trend_skips], ignore_index=True) if not mean_skips.empty or not trend_skips.empty else pd.DataFrame()
    signals = pd.concat(
        [mean_signals.assign(sleeve="MEAN_REVERSION"), trend_signals.assign(sleeve="TREND")],
        ignore_index=True,
    ).sort_values(["signal_time", "sleeve", "strength"], ascending=[True, True, False])
    return signals, combined, skips


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year"), ("sleeve", "sleeve")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(
            RESULTS_DIR / f"{prefix}_by_{suffix}.csv"
        )


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def summary(trades, skips, period):
    result = v22.fixed_period_summary(trades, skips, period)
    result["mean_reversion_weight"] = MEAN_REVERSION_WEIGHT
    result["trend_weight"] = TREND_WEIGHT
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    dev_signals, dev_trades, dev_skips = run_portfolio(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_summary = {"version": VERSION, "variant": VARIANT, **summary(dev_trades, dev_skips, "DEVELOPMENT")}
    write_period("v30_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v30_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V3.0 failed frozen development gates; validation was not opened.")
        return
    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v30_candidate.csv", index=False)
    val_signals, val_trades, val_skips = run_portfolio(data, v21.VALIDATION_START, v21.VALIDATION_END)
    val_summary = {"version": VERSION, "variant": VARIANT, **summary(val_trades, val_skips, "VALIDATION")}
    write_period("v30_validation", val_signals, val_trades, val_skips, val_summary)
    print("V3.0 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([val_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
