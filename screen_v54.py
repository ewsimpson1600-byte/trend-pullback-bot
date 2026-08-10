"""V5.4 two-year screen for biweekly ETF relative-strength continuation.

This is a structurally different response to V5.2/V5.3's unstable dip-buying
expectancy. On alternating week-end sessions, it selects the strongest
risk-adjusted 63-session leader among SPY, QQQ, and IWM, but only when that ETF
has positive absolute momentum above a rising 200-day EMA. Entry is next open,
the protective stop is 2 ATR, and the position exits after five sessions. The
100 ATR target intentionally removes profit taking as a practical exit.

This file is only the frozen 2024-2025 first-stage screen. It cannot validate
the strategy and cannot place orders.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V5.4"
VARIANT = "BIWEEKLY_RISK_ADJUSTED_RELATIVE_STRENGTH"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("RELATIVE_STRENGTH_CONTINUATION", 2.0, 100.0, 5)
MOMENTUM_SESSIONS = 63
VOLATILITY_SESSIONS = 63
EMA_RISE_SESSIONS = 20
WEEK_STRIDE = 2
RESULTS_DIR = Path("screen_results_v54")


def add_relative_strength_fields(frame):
    x = frame.copy()
    daily_return = x["close"].pct_change()
    x["momentum63"] = x["close"] / x["close"].shift(MOMENTUM_SESSIONS) - 1
    x["volatility63"] = daily_return.rolling(VOLATILITY_SESSIONS).std() * np.sqrt(VOLATILITY_SESSIONS)
    x["risk_adjusted_momentum"] = x["momentum63"] / x["volatility63"].replace(0, np.nan)
    x["ema200_rising"] = x["ema200"] > x["ema200"].shift(EMA_RISE_SESSIONS)
    return x


def alternating_week_end_dates(frame, start, end):
    calendar = frame.loc[frame["date"].between(start, end), ["date"]].copy()
    if calendar.empty:
        return []
    calendar["week"] = calendar["date"].dt.to_period("W-FRI")
    week_ends = calendar.groupby("week", sort=True).tail(1)["date"].tolist()
    return week_ends[::WEEK_STRIDE]


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    enriched = {symbol: add_relative_strength_fields(frame) for symbol, frame in data.items()}
    rows = []
    for signal_time in alternating_week_end_dates(enriched["SPY"], start, end):
        eligible = []
        for symbol in SYMBOLS:
            x = enriched[symbol]
            matches = x.index[x["date"] == signal_time]
            if len(matches) != 1:
                continue
            idx = int(matches[0])
            row = x.loc[idx]
            if (
                pd.notna(row["atr14"])
                and pd.notna(row["risk_adjusted_momentum"])
                and bool(row["ema200_rising"])
                and float(row["momentum63"]) > 0
                and float(row["close"]) > float(row["ema200"])
            ):
                eligible.append((float(row["risk_adjusted_momentum"]), symbol, idx, row))
        if not eligible:
            continue
        score, symbol, idx, row = max(eligible, key=lambda item: (item[0], item[1]))
        rows.append({
            "family": FAMILY.name,
            "variant": VARIANT,
            "symbol": symbol,
            "signal_idx": idx,
            "signal_time": signal_time,
            "atr": float(row["atr14"]),
            "strength": score,
            "momentum63": float(row["momentum63"]),
        })
    return pd.DataFrame(rows, columns=[
        "family", "variant", "symbol", "signal_idx", "signal_time", "atr",
        "strength", "momentum63",
    ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    signals = build_signals(data)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = {
        "version": VERSION,
        "variant": VARIANT,
        "screen_only": True,
        "screen_start": protocol.SCREEN_START.date().isoformat(),
        "screen_end": protocol.SCREEN_END.date().isoformat(),
        "universe_size": len(SYMBOLS),
        "momentum_sessions": MOMENTUM_SESSIONS,
        "volatility_sessions": VOLATILITY_SESSIONS,
        "ema_rise_sessions": EMA_RISE_SESSIONS,
        "week_stride": WEEK_STRIDE,
        **asdict(FAMILY),
        **v21.summarize(trades, skips, "TWO_YEAR_SCREEN"),
    }
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v54_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v54_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v54_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v54_skips.csv", index=False)
    print("V5.4 TWO-YEAR RELATIVE-STRENGTH SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
