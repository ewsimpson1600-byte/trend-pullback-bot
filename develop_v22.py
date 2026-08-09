"""V2.2 regime-filtered ETF mean-reversion research.

V2.1 identified long-only ETF mean reversion as the only promising family,
but it was inconsistent across development years. V2.2 makes one structural,
predeclared change: entries are allowed only while SPY is above a rising
200-day EMA. Rules are fixed on 2010-2017 before any 2018-2025 evaluation.
The inherited simulator remains cash-only, integer-share, gap-aware, and
research-only; it cannot submit orders.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21


VERSION = "V2.2"
VARIANT = "REGIME_FILTERED_MEAN_REVERSION"
FAMILY = next(item for item in v21.FAMILIES if item.name == "MEAN_REVERSION")
REGIME_SLOPE_SESSIONS = 20
RESULTS_DIR = Path("backtest_results_v22")


def risk_on_dates(data):
    """Dates when the broad market is above a rising long-term trend."""
    spy = data["SPY"]
    valid = (
        spy["ema200"].notna()
        & spy["ema200"].shift(REGIME_SLOPE_SESSIONS).notna()
        & (spy["close"] > spy["ema200"])
        & (spy["ema200"] > spy["ema200"].shift(REGIME_SLOPE_SESSIONS))
    )
    return set(pd.to_datetime(spy.loc[valid, "date"]).dt.normalize())


def build_signals(data, start, end):
    signals = v21.build_signals(data, FAMILY, start, end)
    if signals.empty:
        return signals
    allowed = risk_on_dates(data)
    filtered = signals.loc[pd.to_datetime(signals["signal_time"]).dt.normalize().isin(allowed)].copy()
    filtered["variant"] = VARIANT
    return filtered.reset_index(drop=True)


def fixed_period_summary(trades, skips, period):
    """Apply frozen gates while counting no-trade calendar years as nonpositive."""
    result = v21.summarize(trades, skips, period)
    start = v21.DEVELOPMENT_START if period == "DEVELOPMENT" else v21.VALIDATION_START
    end = v21.DEVELOPMENT_END if period == "DEVELOPMENT" else v21.VALIDATION_END
    years = list(range(start.year, end.year + 1))
    if trades.empty:
        positive_years = 0
    else:
        work = trades.copy()
        work["year"] = pd.to_datetime(work["entry_time"]).dt.year
        annual = work.groupby("year")["trade_pnl"].sum().reindex(years, fill_value=0.0)
        positive_years = int((annual > 0).sum())
    result["positive_years"] = positive_years
    result["years_tested"] = len(years)
    if period == "DEVELOPMENT":
        result["pass"] = bool(
            result["trades"] >= v21.MIN_DEVELOPMENT_TRADES
            and result.get("account_return_pct", 0) > 0
            and result.get("profit_factor", 0) >= v21.MIN_DEVELOPMENT_PROFIT_FACTOR
            and result.get("account_max_drawdown_pct", -100) >= v21.MAX_DEVELOPMENT_DRAWDOWN
            and positive_years >= len(years) - 2
        )
    else:
        result["pass"] = bool(
            result["trades"] >= 50
            and result.get("bootstrap_mean_95_ci_low_pct", -100) > 0
            and result.get("profit_factor", 0) >= 1.50
            and result.get("account_max_drawdown_pct", -100) >= -25
            and result.get("max_ticker_profit_contribution_pct", 100) <= 60
            and result.get("max_month_profit_contribution_pct", 100) <= 35
            and positive_years >= len(years) - 1
        )
    return result


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(
            RESULTS_DIR / f"{prefix}_by_{suffix}.csv"
        )


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    dev_signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_trades, dev_skips = v21.run_account(dev_signals, data, FAMILY)
    dev_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "regime_slope_sessions": REGIME_SLOPE_SESSIONS,
        **asdict(FAMILY),
        **fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v22_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v22_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.2 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v22_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "regime_slope_sessions": REGIME_SLOPE_SESSIONS,
        **asdict(FAMILY),
        **fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v22_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.2 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
