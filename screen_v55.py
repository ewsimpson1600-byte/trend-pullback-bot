"""V5.5 two-year screen for expanding sector-breadth continuation.

On alternating week-end sessions, at least six of nine legacy sector ETFs
must be above their 50-day EMA and the count must have increased from five
sessions earlier. The least-used eligible sector with positive 20-session
momentum is selected, breaking usage ties by momentum. This breadth-expansion
premise differs from V5.4 relative-strength ranking and earlier dip buying.

Entry is next open with a 2 ATR protective stop and a five-session time exit.
The 100 ATR target removes profit taking as a practical exit. This is only a
frozen 2024-2025 screen and has no order-submission path.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v24 as v24
import develop_v26 as v26
import research_protocol_v47 as protocol


VERSION = "V5.5"
VARIANT = "EXPANDING_SECTOR_BREADTH_CONTINUATION"
SYMBOLS = v26.SECTOR_SYMBOLS
FAMILY = v21.Family("SECTOR_BREADTH_CONTINUATION", 2.0, 100.0, 5)
TREND_EMA = 50
MOMENTUM_SESSIONS = 20
BREADTH_LOOKBACK_SESSIONS = 5
MIN_SECTORS_ABOVE_TREND = 6
WEEK_STRIDE = 2
RESULTS_DIR = Path("screen_results_v55")


def add_fields(frame):
    x = frame.copy()
    x["momentum20"] = x["close"] / x["close"].shift(MOMENTUM_SESSIONS) - 1
    x["above_trend"] = x["close"] > x["ema50"]
    return x


def breadth_frame(data):
    columns = []
    for symbol in SYMBOLS:
        x = data[symbol].set_index("date")
        columns.append(x["above_trend"].rename(symbol))
    result = pd.concat(columns, axis=1).dropna()
    result["breadth_count"] = result[list(SYMBOLS)].sum(axis=1)
    result["prior_breadth_count"] = result["breadth_count"].shift(BREADTH_LOOKBACK_SESSIONS)
    result["breadth_expanding"] = result["breadth_count"] > result["prior_breadth_count"]
    return result


def alternating_week_ends(frame, start, end):
    calendar = frame.loc[frame["date"].between(start, end), ["date"]].copy()
    if calendar.empty:
        return []
    calendar["week"] = calendar["date"].dt.to_period("W-FRI")
    week_ends = calendar.groupby("week", sort=True).tail(1)["date"].tolist()
    return week_ends[::WEEK_STRIDE]


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    enriched = {symbol: add_fields(data[symbol]) for symbol in SYMBOLS}
    breadth = breadth_frame(enriched)
    usage = {symbol: 0 for symbol in SYMBOLS}
    rows = []
    for signal_time in alternating_week_ends(enriched["XLB"], start, end):
        if signal_time not in breadth.index:
            continue
        state = breadth.loc[signal_time]
        if (
            pd.isna(state["prior_breadth_count"])
            or int(state["breadth_count"]) < MIN_SECTORS_ABOVE_TREND
            or not bool(state["breadth_expanding"])
        ):
            continue
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
                and pd.notna(row["momentum20"])
                and bool(row["above_trend"])
                and float(row["momentum20"]) > 0
            ):
                eligible.append((usage[symbol], -float(row["momentum20"]), symbol, idx, row))
        if not eligible:
            continue
        _, neg_momentum, symbol, idx, row = min(eligible)
        usage[symbol] += 1
        rows.append({
            "family": FAMILY.name,
            "variant": VARIANT,
            "symbol": symbol,
            "signal_idx": idx,
            "signal_time": signal_time,
            "atr": float(row["atr14"]),
            "strength": float(-usage[symbol]),
            "momentum20": -neg_momentum,
            "breadth_count": int(state["breadth_count"]),
            "prior_breadth_count": int(state["prior_breadth_count"]),
            "selection_count": usage[symbol],
        })
    return pd.DataFrame(rows, columns=[
        "family", "variant", "symbol", "signal_idx", "signal_time", "atr",
        "strength", "momentum20", "breadth_count", "prior_breadth_count",
        "selection_count",
    ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_data = v24.load_data()
    data = {symbol: all_data[symbol] for symbol in SYMBOLS}
    signals = build_signals(data)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = {
        "version": VERSION,
        "variant": VARIANT,
        "screen_only": True,
        "screen_start": protocol.SCREEN_START.date().isoformat(),
        "screen_end": protocol.SCREEN_END.date().isoformat(),
        "universe_size": len(SYMBOLS),
        "trend_ema": TREND_EMA,
        "momentum_sessions": MOMENTUM_SESSIONS,
        "breadth_lookback_sessions": BREADTH_LOOKBACK_SESSIONS,
        "min_sectors_above_trend": MIN_SECTORS_ABOVE_TREND,
        "week_stride": WEEK_STRIDE,
        **asdict(FAMILY),
        **v21.summarize(trades, skips, "TWO_YEAR_SCREEN"),
    }
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v55_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v55_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v55_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v55_skips.csv", index=False)
    print("V5.5 TWO-YEAR EXPANDING-BREADTH SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
