"""V4.9 two-year screen for a broad-ETF downside-gap reclaim strategy.

The fixed hypothesis is that a liquid ETF in a long-term uptrend which gaps
down at least 1% but closes back above the prior close has demonstrated demand
that may persist for several sessions. Entry is the next open. The simulator
retains cash-only integer shares, conservative costs, overnight-gap exits,
one position, a 2% account-risk cap, and an 80% allocation cap. This is only a
2024-2025 screen under the frozen V4.7 protocol and cannot place orders.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V4.9"
VARIANT = "BULLISH_DOWNSIDE_GAP_RECLAIM"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("GAP_RECLAIM", 2.0, 2.0, 5)
MIN_GAP_DOWN = -0.01
RESULTS_DIR = Path("screen_results_v49")


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    rows = []
    for symbol, frame in data.items():
        x = frame.copy()
        prior_close = x["close"].shift(1)
        gap = x["open"] / prior_close - 1
        valid = (
            x["date"].between(start, end)
            & x[["atr14", "ema200"]].notna().all(axis=1)
            & (x["close"] > x["ema200"])
            & (gap <= MIN_GAP_DOWN)
            & (x["close"] > prior_close)
            & (x["close"] > x["open"])
        )
        strength = (x["close"] / prior_close - 1) - gap
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
               "universe_size": len(SYMBOLS), "minimum_gap_down_pct": MIN_GAP_DOWN * 100,
               **asdict(FAMILY), **v21.summarize(trades, skips, "TWO_YEAR_SCREEN")}
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v49_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v49_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v49_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v49_skips.csv", index=False)
    print("V4.9 TWO-YEAR SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
