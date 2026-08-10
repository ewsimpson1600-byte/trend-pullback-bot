"""V5.2 two-year screen for a three-lower-closes ETF pullback.

The fixed signal requires three consecutive lower closes while a liquid broad
ETF remains above its 200-day EMA. It enters at the next open, uses a 2 ATR
protective stop, a 1.5 ATR rebound target, and a five-session maximum hold.
This price-sequence premise is separate from prior RSI/Bollinger triggers. It
is only a frozen 2024-2025 screen and inherits the research-only cash simulator.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V5.2"
VARIANT = "THREE_LOWER_CLOSES_UPTREND"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("THREE_LOWER_CLOSES", 2.0, 1.5, 5)
CONSECUTIVE_LOWER_CLOSES = 3
RESULTS_DIR = Path("screen_results_v52")


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    rows = []
    for symbol, frame in data.items():
        x = frame.copy()
        lower_sequence = (
            (x["close"] < x["close"].shift(1))
            & (x["close"].shift(1) < x["close"].shift(2))
            & (x["close"].shift(2) < x["close"].shift(3))
        )
        valid = (
            x["date"].between(start, end)
            & x[["atr14", "ema200"]].notna().all(axis=1)
            & (x["close"] > x["ema200"])
            & lower_sequence
        )
        pullback = (x["close"].shift(3) / x["close"] - 1)
        for idx in x.index[valid]:
            rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                         "signal_idx": int(idx), "signal_time": x.at[idx, "date"],
                         "atr": float(x.at[idx, "atr14"]),
                         "strength": float(pullback.at[idx])})
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
               "universe_size": len(SYMBOLS),
               "consecutive_lower_closes": CONSECUTIVE_LOWER_CLOSES,
               **asdict(FAMILY), **v21.summarize(trades, skips, "TWO_YEAR_SCREEN")}
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v52_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v52_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v52_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v52_skips.csv", index=False)
    print("V5.2 TWO-YEAR SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
