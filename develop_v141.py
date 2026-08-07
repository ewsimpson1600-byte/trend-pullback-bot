"""V1.4.1 exit-policy comparison with frozen V1.4 entry rules."""

from pathlib import Path

import pandas as pd

import develop_v13_expanded as engine


engine.RVOL_MIN = 3.0
engine.SIGNAL_END = "10:00"
engine.CACHE_DIR = Path("backtest_data_v13_expanded")

RESULTS_DIR = Path("backtest_results_v141")
engine.RESULTS_DIR = RESULTS_DIR


VARIANTS = (
    {
        "variant": "UNDERLYING_STOP_ONLY",
        "option_stop": None,
        "trailing_activation": None,
        "trailing_distance": None,
    },
    {
        "variant": "OPTION_STOP_25",
        "option_stop": -0.25,
        "trailing_activation": None,
        "trailing_distance": None,
    },
    {
        "variant": "OPTION_STOP_30",
        "option_stop": -0.30,
        "trailing_activation": None,
        "trailing_distance": None,
    },
    {
        "variant": "OPTION_STOP_35",
        "option_stop": -0.35,
        "trailing_activation": None,
        "trailing_distance": None,
    },
    {
        "variant": "TRAIL_AFTER_15_DISTANCE_10",
        "option_stop": None,
        "trailing_activation": 0.15,
        "trailing_distance": 0.10,
    },
)


def configure_exit(variant):
    engine.OPTION_STOP_RETURN = variant["option_stop"]
    engine.TRAILING_STOP_ACTIVATION_RETURN = variant["trailing_activation"]
    engine.TRAILING_STOP_DISTANCE = variant["trailing_distance"]


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
    signals.to_csv(RESULTS_DIR / "v141_signals.csv", index=False)

    summaries = []
    all_trades = []
    all_tickers = []
    all_months = []

    for variant in VARIANTS:
        configure_exit(variant)
        name = variant["variant"]
        print(f"Running {name}...")

        trades = engine.run_portfolio(signals, data)
        if not trades.empty:
            entries = pd.to_datetime(
                trades["entry_time"], utc=True
            ).dt.tz_convert("America/New_York")
            if (entries.dt.strftime("%H:%M") > "10:05").any():
                raise AssertionError(f"{name} produced an entry after 10:05 ET")

        summary = engine.summarize(trades)
        summary["variant"] = name
        summary["option_stop_return"] = variant["option_stop"]
        summary["trailing_activation_return"] = variant["trailing_activation"]
        summary["trailing_distance"] = variant["trailing_distance"]
        summaries.append(summary)

        if not trades.empty:
            variant_trades = trades.copy()
            variant_trades.insert(0, "variant", name)
            all_trades.append(variant_trades)

            by_ticker = engine.build_by_ticker(trades)
            by_ticker.insert(0, "variant", name)
            all_tickers.append(by_ticker)

            by_month = engine.build_by_month(trades)
            by_month.insert(0, "variant", name)
            all_months.append(by_month)

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(RESULTS_DIR / "v141_comparison.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(
        RESULTS_DIR / "v141_trades.csv", index=False
    )
    pd.concat(all_tickers, ignore_index=True).to_csv(
        RESULTS_DIR / "v141_by_ticker.csv", index=False
    )
    pd.concat(all_months, ignore_index=True).to_csv(
        RESULTS_DIR / "v141_by_month.csv", index=False
    )

    columns = [
        "variant",
        "trades",
        "win_rate_pct",
        "avg_option_return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "option_stop_hits",
        "trailing_stop_hits",
    ]
    print()
    print(comparison[columns].to_string(index=False))
    print()
    print("Synthetic option prices are model-based, not historical quotes.")
    print(f"Files written to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
