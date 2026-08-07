"""V1.4.2 frozen out-of-sample backtest for 2024 through 2025."""

from pathlib import Path

import numpy as np
import pandas as pd

import develop_v13_expanded as engine


# Frozen V1.4 candidate rules.
engine.RVOL_MIN = 3.0
engine.SIGNAL_END = "10:00"
engine.OPTION_STOP_RETURN = None
engine.TRAILING_STOP_ACTIVATION_RETURN = None
engine.TRAILING_STOP_DISTANCE = None

# Untouched period selected after the 2026 development analysis.
engine.DOWNLOAD_START = "2023-12-01"
engine.DOWNLOAD_END = "2025-12-01"
engine.TEST_START = pd.Timestamp("2024-01-02", tz="America/New_York")
engine.TEST_END = pd.Timestamp(
    "2025-11-26 16:00", tz="America/New_York"
)

engine.CACHE_DIR = Path("backtest_data_v142_oos")
RESULTS_DIR = Path("backtest_results_v142")
engine.RESULTS_DIR = RESULTS_DIR

STARTING_ACCOUNT = 1000.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 142


def bootstrap_mean_ci(returns_pct):
    values = np.asarray(returns_pct, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def simulate_one_contract_account(trades):
    balance = STARTING_ACCOUNT
    peak = balance
    rows = []

    for _, trade in trades.sort_values("entry_time").iterrows():
        contract_cost = float(trade["modeled_option_entry"]) * 100.0
        can_enter = contract_cost <= balance
        pnl = 0.0

        if can_enter:
            pnl = contract_cost * float(trade["option_return_pct"]) / 100.0
            balance += pnl

        peak = max(peak, balance)
        drawdown = balance / peak - 1.0
        rows.append(
            {
                "entry_time": trade["entry_time"],
                "symbol": trade["symbol"],
                "contract_cost": contract_cost,
                "can_enter": can_enter,
                "trade_pnl": pnl,
                "account_balance": balance,
                "drawdown_pct": drawdown * 100.0,
            }
        )

    return pd.DataFrame(rows)


def main():
    engine.ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = engine.download_all_data()
    data = {
        symbol: engine.add_indicators(raw_data[symbol])
        for symbol in engine.SYMBOLS
    }
    market = engine.create_market_confirmation(data)
    signals = engine.build_signals(data, market)
    trades = engine.run_portfolio(signals, data)

    signals.to_csv(RESULTS_DIR / "v142_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v142_trades.csv", index=False)

    summary = engine.summarize(trades)
    ci_low, ci_high = bootstrap_mean_ci(trades["option_return_pct"])
    summary["bootstrap_mean_95_ci_low_pct"] = ci_low
    summary["bootstrap_mean_95_ci_high_pct"] = ci_high
    summary["test_start"] = str(engine.TEST_START)
    summary["test_end"] = str(engine.TEST_END)

    account = simulate_one_contract_account(trades)
    if not account.empty:
        summary["starting_account"] = STARTING_ACCOUNT
        summary["ending_account"] = float(account.iloc[-1]["account_balance"])
        summary["account_return_pct"] = (
            summary["ending_account"] / STARTING_ACCOUNT - 1.0
        ) * 100.0
        summary["account_max_drawdown_pct"] = float(
            account["drawdown_pct"].min()
        )
        summary["unaffordable_trades"] = int((~account["can_enter"]).sum())

    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "v142_summary.csv", index=False
    )
    account.to_csv(RESULTS_DIR / "v142_account_1000.csv", index=False)

    by_ticker = (
        engine.build_by_ticker(trades) if not trades.empty else pd.DataFrame()
    )
    by_month = (
        engine.build_by_month(trades) if not trades.empty else pd.DataFrame()
    )
    by_ticker.to_csv(RESULTS_DIR / "v142_by_ticker.csv", index=False)
    by_month.to_csv(RESULTS_DIR / "v142_by_month.csv", index=False)

    print("V1.4.2 FROZEN OUT-OF-SAMPLE RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()
    print("Synthetic option prices are model-based, not historical quotes.")
    print(f"Files written to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
