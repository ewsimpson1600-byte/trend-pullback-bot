"""V5.0 two-year screen for an NR7 volatility-contraction ETF setup.

The predeclared signal is a seven-session narrowest-range day which closes in
the top quarter of its range while the ETF is above its 200-day EMA. The trade
enters at the next open with a 1.5 ATR stop, 3 ATR target, and eight-session
maximum hold. This is a compression/expansion hypothesis evaluated only by the
frozen 2024-2025 V4.7 screen. It inherits the cash-account simulator and has no
brokerage or notification path.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V5.0"
VARIANT = "NR7_TOP_QUARTER_CLOSE"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("NR7_EXPANSION", 1.5, 3.0, 8)
LOOKBACK = 7
MIN_CLOSE_LOCATION = 0.75
RESULTS_DIR = Path("screen_results_v50")


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    rows = []
    for symbol, frame in data.items():
        x = frame.copy()
        daily_range = x["high"] - x["low"]
        rolling_min = daily_range.rolling(LOOKBACK).min()
        close_location = (x["close"] - x["low"]) / daily_range.replace(0, pd.NA)
        valid = (
            x["date"].between(start, end)
            & x[["atr14", "ema200"]].notna().all(axis=1)
            & (x["close"] > x["ema200"])
            & (daily_range <= rolling_min)
            & (close_location >= MIN_CLOSE_LOCATION)
        )
        strength = close_location + (x["close"] / x["ema200"] - 1)
        for idx in x.index[valid]:
            rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                         "signal_idx": int(idx), "signal_time": x.at[idx, "date"],
                         "atr": float(x.at[idx, "atr14"]),
                         "strength": float(strength.at[idx])})
    if not rows:
        return pd.DataFrame(columns=["family", "variant", "symbol", "signal_idx",
                                     "signal_time", "atr", "strength"])
    return pd.DataFrame(rows).sort_values(
        ["signal_time", "strength", "symbol"], ascending=[True, False, True]
    ).reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    signals = build_signals(data)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = {"version": VERSION, "variant": VARIANT, "screen_only": True,
               "screen_start": protocol.SCREEN_START.date().isoformat(),
               "screen_end": protocol.SCREEN_END.date().isoformat(),
               "universe_size": len(SYMBOLS), "nr_lookback": LOOKBACK,
               "minimum_close_location": MIN_CLOSE_LOCATION, **asdict(FAMILY),
               **v21.summarize(trades, skips, "TWO_YEAR_SCREEN")}
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v50_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v50_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v50_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v50_skips.csv", index=False)
    print("V5.0 TWO-YEAR SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
