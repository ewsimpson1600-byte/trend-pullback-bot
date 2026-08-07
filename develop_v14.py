"""V1.4 candidate backtest.

V1.4 preserves the V1.3 expanded-universe signal and option model while
testing the pre-registered changes selected from the V1.3 diagnostic:

* breakout RVOL >= 3.0;
* entry no later than 10:05 America/New_York (signal no later than 10:00);
* modeled option-premium stop at -20%.

The underlying 2-ATR stop, +30% option target, 90-minute maximum hold,
five-DTE synthetic call, friction assumptions, and one-position portfolio rule
remain unchanged.
"""

from pathlib import Path

import pandas as pd

import develop_v13_expanded as engine


# V1.4 candidate changes.
engine.RVOL_MIN = 3.0
engine.SIGNAL_END = "10:00"
engine.OPTION_STOP_RETURN = -0.20

# Reuse the V1.3 expanded raw-data cache but never overwrite V1.3 results.
engine.CACHE_DIR = Path("backtest_data_v13_expanded")
RESULTS_DIR = Path("backtest_results_v14")
engine.RESULTS_DIR = RESULTS_DIR


def main():
    engine.ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("V1.4 CANDIDATE — EXPANDED UNIVERSE BACKTEST")
    print("=" * 75)
    print(f"Symbols: {len(engine.SYMBOLS)}")
    print(f"RVOL minimum: {engine.RVOL_MIN}")
    print("Latest entry: 10:05 America/New_York")
    print(f"Modeled option stop: {engine.OPTION_STOP_RETURN:.0%}")
    print(f"Modeled option target: {engine.OPTION_TARGET_RETURN:.0%}")
    print()

    raw_data = engine.download_all_data()
    data = {
        symbol: engine.add_indicators(raw_data[symbol])
        for symbol in engine.SYMBOLS
    }
    market = engine.create_market_confirmation(data)
    signals = engine.build_signals(data, market)

    signals.to_csv(
        RESULTS_DIR / "v14_signals.csv",
        index=False,
    )

    trades = engine.run_portfolio(signals, data)

    if not trades.empty:
        entry_times = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(
            "America/New_York"
        )
        if (entry_times.dt.strftime("%H:%M") > "10:05").any():
            raise AssertionError("V1.4 produced an entry after 10:05 ET")

    trades.to_csv(
        RESULTS_DIR / "v14_trades.csv",
        index=False,
    )

    summary = engine.summarize(trades)
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "v14_summary.csv",
        index=False,
    )

    by_ticker = (
        engine.build_by_ticker(trades)
        if not trades.empty
        else pd.DataFrame()
    )
    by_month = (
        engine.build_by_month(trades)
        if not trades.empty
        else pd.DataFrame()
    )
    by_ticker.to_csv(
        RESULTS_DIR / "v14_by_ticker.csv",
        index=False,
    )
    by_month.to_csv(
        RESULTS_DIR / "v14_by_month.csv",
        index=False,
    )

    print("=" * 75)
    print("V1.4 RESULTS")
    print("=" * 75)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()
    print("Synthetic option prices are model-based, not historical quotes.")
    print(f"Files written to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
