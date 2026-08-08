"""V1.5 risk-managed development grid and untouched validation.

Development uses the already-consumed 2024-2025 V1.4.2 period.  A policy is
selected mechanically, then frozen before it is evaluated on 2022-2023 data.
Signals remain V1.4: RVOL >= 3, signal by 10:00 ET, two-ATR underlying stop,
and a 90-minute maximum hold.

Option prices are Black-Scholes estimates, not historical option quotes.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import develop_v13_expanded as engine


STARTING_ACCOUNT = 1000.0
MIN_DEVELOPMENT_TRADES = 6
MAX_ACCEPTABLE_DRAWDOWN_PCT = -25.0
MIN_PROFIT_FACTOR = 1.30
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 150

DEVELOPMENT_DOWNLOAD_START = "2023-12-01"
DEVELOPMENT_DOWNLOAD_END = "2025-12-01"
DEVELOPMENT_START = pd.Timestamp("2024-01-02", tz="America/New_York")
DEVELOPMENT_END = pd.Timestamp(
    "2025-11-26 16:00", tz="America/New_York"
)
DEVELOPMENT_CACHE = Path("backtest_data_v142_oos")

VALIDATION_DOWNLOAD_START = "2021-12-01"
VALIDATION_DOWNLOAD_END = "2024-01-01"
VALIDATION_START = pd.Timestamp("2022-01-03", tz="America/New_York")
VALIDATION_END = pd.Timestamp(
    "2023-12-29 16:00", tz="America/New_York"
)
VALIDATION_CACHE = Path("backtest_data_v15_validation")

RESULTS_DIR = Path("backtest_results_v15")


@dataclass(frozen=True)
class Policy:
    target_delta: float
    option_target_return: float
    max_contract_allocation: float
    max_estimated_stop_risk: float

    @property
    def name(self):
        return (
            f"d{self.target_delta:.2f}_t{self.option_target_return:.2f}_"
            f"a{self.max_contract_allocation:.2f}_"
            f"r{self.max_estimated_stop_risk:.2f}"
        )


POLICIES = [
    Policy(delta, target, allocation, risk)
    for delta in (0.25, 0.35, 0.50)
    for target in (0.30, 0.45, 0.60)
    for allocation in (0.25, 0.40)
    for risk in (0.03, 0.05)
]


def configure_frozen_entries():
    engine.RVOL_MIN = 3.0
    engine.SIGNAL_END = "10:00"
    engine.OPTION_STOP_RETURN = None
    engine.TRAILING_STOP_ACTIVATION_RETURN = None
    engine.TRAILING_STOP_DISTANCE = None


def configure_period(download_start, download_end, test_start, test_end, cache):
    engine.DOWNLOAD_START = download_start
    engine.DOWNLOAD_END = download_end
    engine.TEST_START = test_start
    engine.TEST_END = test_end
    engine.CACHE_DIR = cache


def load_period(download_start, download_end, test_start, test_end, cache):
    configure_period(download_start, download_end, test_start, test_end, cache)
    engine.ensure_directories()
    raw = engine.download_all_data()
    data = {
        symbol: engine.add_indicators(raw[symbol])
        for symbol in engine.SYMBOLS
    }
    market = engine.create_market_confirmation(data)
    return data, engine.build_signals(data, market)


def estimated_stop_risk_dollars(trade):
    """Estimate one-contract loss at the underlying stop at entry time."""
    stop_value = engine.bs_call_price(
        float(trade["stop_stock"]),
        float(trade["option_strike"]),
        engine.OPTION_DTE / 365,
        engine.RISK_FREE_RATE,
        float(trade["estimated_volatility"]),
    ) * (1 - engine.EXIT_FRICTION)
    entry_cost = float(trade["modeled_option_entry"]) * 100.0
    return max(0.0, entry_cost - stop_value * 100.0)


def run_policy(signals, data, policy, starting_account=STARTING_ACCOUNT):
    """Run a cash-only, one-contract, one-position paper portfolio."""
    engine.TARGET_DELTA = policy.target_delta
    engine.OPTION_TARGET_RETURN = policy.option_target_return

    balance = float(starting_account)
    peak = balance
    next_available_time = None
    accepted = []
    account_rows = []
    skip_rows = []

    for _, signal in signals.sort_values("signal_time").iterrows():
        if (
            next_available_time is not None
            and signal["signal_time"] < next_available_time
        ):
            continue

        trade = engine.simulate_trade(signal, data[signal["symbol"]])
        if trade is None:
            continue

        contract_cost = float(trade["modeled_option_entry"]) * 100.0
        estimated_risk = estimated_stop_risk_dollars(trade)
        allocation_limit = balance * policy.max_contract_allocation
        risk_limit = balance * policy.max_estimated_stop_risk

        reason = None
        if contract_cost > balance:
            reason = "INSUFFICIENT_CASH"
        elif contract_cost > allocation_limit:
            reason = "ALLOCATION_LIMIT"
        elif estimated_risk > risk_limit:
            reason = "STOP_RISK_LIMIT"

        if reason:
            skip_rows.append(
                {
                    "signal_time": signal["signal_time"],
                    "symbol": signal["symbol"],
                    "reason": reason,
                    "contract_cost": contract_cost,
                    "estimated_stop_risk": estimated_risk,
                    "account_balance": balance,
                }
            )
            continue

        pnl = contract_cost * float(trade["option_return_pct"]) / 100.0
        balance += pnl
        peak = max(peak, balance)
        drawdown_pct = (balance / peak - 1.0) * 100.0

        trade.update(
            {
                "policy": policy.name,
                "contract_cost": contract_cost,
                "estimated_stop_risk": estimated_risk,
                "trade_pnl": pnl,
                "account_balance": balance,
                "account_drawdown_pct": drawdown_pct,
            }
        )
        accepted.append(trade)
        account_rows.append(
            {
                "entry_time": trade["entry_time"],
                "symbol": trade["symbol"],
                "contract_cost": contract_cost,
                "estimated_stop_risk": estimated_risk,
                "trade_pnl": pnl,
                "account_balance": balance,
                "drawdown_pct": drawdown_pct,
            }
        )
        next_available_time = trade["exit_time"]

    return (
        pd.DataFrame(accepted),
        pd.DataFrame(account_rows),
        pd.DataFrame(skip_rows),
    )


def portfolio_summary(trades, account, skipped, policy):
    summary = asdict(policy)
    summary["policy"] = policy.name
    summary.update(engine.summarize(trades))
    ending = (
        float(account.iloc[-1]["account_balance"])
        if not account.empty
        else STARTING_ACCOUNT
    )
    summary["starting_account"] = STARTING_ACCOUNT
    summary["ending_account"] = ending
    summary["account_return_pct"] = (
        ending / STARTING_ACCOUNT - 1.0
    ) * 100.0
    summary["account_max_drawdown_pct"] = (
        float(account["drawdown_pct"].min()) if not account.empty else 0.0
    )
    summary["skipped_signals"] = len(skipped)
    return summary


def candidate_is_eligible(row):
    return (
        int(row.get("trades", 0)) >= MIN_DEVELOPMENT_TRADES
        and float(row.get("account_return_pct", -np.inf)) > 0
        and float(row.get("account_max_drawdown_pct", -np.inf))
        >= MAX_ACCEPTABLE_DRAWDOWN_PCT
        and float(row.get("profit_factor", 0)) >= MIN_PROFIT_FACTOR
    )


def select_candidate(grid):
    eligible = grid[grid.apply(candidate_is_eligible, axis=1)].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        [
            "account_return_pct",
            "account_max_drawdown_pct",
            "profit_factor",
            "trades",
        ],
        ascending=[False, False, False, False],
    )
    name = eligible.iloc[0]["policy"]
    return next(policy for policy in POLICIES if policy.name == name)


def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(
        values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def run_development_grid(signals, data):
    rows = []
    for number, policy in enumerate(POLICIES, start=1):
        print(f"Development policy {number}/{len(POLICIES)}: {policy.name}")
        trades, account, skipped = run_policy(signals, data, policy)
        rows.append(portfolio_summary(trades, account, skipped, policy))
    return pd.DataFrame(rows)


def write_validation_outputs(policy, signals, trades, account, skipped):
    signals.to_csv(RESULTS_DIR / "v15_validation_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v15_validation_trades.csv", index=False)
    account.to_csv(RESULTS_DIR / "v15_validation_account.csv", index=False)
    skipped.to_csv(RESULTS_DIR / "v15_validation_skips.csv", index=False)

    summary = portfolio_summary(trades, account, skipped, policy)
    ci_low, ci_high = bootstrap_mean_ci(trades.get("option_return_pct", []))
    summary["bootstrap_mean_95_ci_low_pct"] = ci_low
    summary["bootstrap_mean_95_ci_high_pct"] = ci_high
    summary["test_start"] = str(VALIDATION_START)
    summary["test_end"] = str(VALIDATION_END)
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "v15_validation_summary.csv", index=False
    )

    by_ticker = engine.build_by_ticker(trades) if not trades.empty else pd.DataFrame()
    by_month = engine.build_by_month(trades) if not trades.empty else pd.DataFrame()
    by_ticker.to_csv(RESULTS_DIR / "v15_validation_by_ticker.csv", index=False)
    by_month.to_csv(RESULTS_DIR / "v15_validation_by_month.csv", index=False)
    return summary


def main():
    configure_frozen_entries()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("V1.5 DEVELOPMENT: 2024-2025")
    development_data, development_signals = load_period(
        DEVELOPMENT_DOWNLOAD_START,
        DEVELOPMENT_DOWNLOAD_END,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
        DEVELOPMENT_CACHE,
    )
    grid = run_development_grid(development_signals, development_data)
    grid.to_csv(RESULTS_DIR / "v15_development_grid.csv", index=False)

    candidate = select_candidate(grid)
    if candidate is None:
        pd.DataFrame(
            [{"status": "NO_ELIGIBLE_POLICY"}]
        ).to_csv(RESULTS_DIR / "v15_candidate.csv", index=False)
        print("No development policy met the pre-registered risk criteria.")
        return

    pd.DataFrame(
        [{"status": "SELECTED", **asdict(candidate), "policy": candidate.name}]
    ).to_csv(RESULTS_DIR / "v15_candidate.csv", index=False)
    print(f"Frozen candidate: {candidate.name}")

    development_trades, development_account, development_skips = run_policy(
        development_signals, development_data, candidate
    )
    development_trades.to_csv(
        RESULTS_DIR / "v15_selected_development_trades.csv", index=False
    )
    development_account.to_csv(
        RESULTS_DIR / "v15_selected_development_account.csv", index=False
    )
    development_skips.to_csv(
        RESULTS_DIR / "v15_selected_development_skips.csv", index=False
    )

    print("V1.5 UNTOUCHED VALIDATION: 2022-2023")
    validation_data, validation_signals = load_period(
        VALIDATION_DOWNLOAD_START,
        VALIDATION_DOWNLOAD_END,
        VALIDATION_START,
        VALIDATION_END,
        VALIDATION_CACHE,
    )
    trades, account, skipped = run_policy(
        validation_signals, validation_data, candidate
    )
    summary = write_validation_outputs(
        candidate, validation_signals, trades, account, skipped
    )

    print("V1.5 VALIDATION RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Synthetic option prices are model-based, not historical quotes.")


if __name__ == "__main__":
    main()
