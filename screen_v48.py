"""V4.8 two-year screen for a diversified turn-of-month ETF strategy.

This is a calendar-effect hypothesis, not another threshold revision of an
earlier signal family. On the second-to-last trading session of each month it
selects the least-used affordable broad-market ETF that is above its 200-day
EMA. Entry is the next open (the final trading session), and the position is
held through four sessions unless a 3 ATR protective stop is reached. The
2024-2025 window and advancement gates come only from research_protocol_v47.
Historical screening cannot constitute fresh validation and this module has
no order-submission path.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V4.8"
VARIANT = "DIVERSIFIED_TURN_OF_MONTH"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("TURN_OF_MONTH", 3.0, 100.0, 4)
RESULTS_DIR = Path("screen_results_v48")


def month_signal_dates(frame, start, end):
    """Return the second-to-last observed session of each eligible month."""
    work = frame.loc[frame["date"].between(start, end), ["date"]].copy()
    if work.empty:
        return []
    work["month"] = work["date"].dt.to_period("M")
    dates = []
    for _, group in work.groupby("month", sort=True):
        if len(group) >= 2:
            dates.append(group.iloc[-2]["date"])
    return dates


def eligible_on_date(frame, date):
    matches = frame.index[frame["date"] == date]
    if len(matches) != 1:
        return None
    idx = int(matches[0])
    row = frame.loc[idx]
    if pd.isna(row["ema200"]) or pd.isna(row["atr14"]):
        return None
    if float(row["close"]) <= float(row["ema200"]):
        return None
    return idx, row


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    usage = {symbol: 0 for symbol in SYMBOLS}
    rows = []
    for date in month_signal_dates(data["SPY"], start, end):
        eligible = []
        for symbol in SYMBOLS:
            result = eligible_on_date(data[symbol], date)
            if result is not None:
                eligible.append((usage[symbol], symbol, result))
        if not eligible:
            continue
        _, symbol, (idx, row) = min(eligible, key=lambda item: (item[0], item[1]))
        usage[symbol] += 1
        rows.append({"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
                     "signal_idx": idx, "signal_time": date,
                     "atr": float(row["atr14"]), "strength": 1.0})
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx",
                                      "signal_time", "atr", "strength"])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    signals = build_signals(data)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = {"version": VERSION, "variant": VARIANT, "screen_only": True,
               "screen_start": protocol.SCREEN_START.date().isoformat(),
               "screen_end": protocol.SCREEN_END.date().isoformat(),
               "universe_size": len(SYMBOLS), **asdict(FAMILY),
               **v21.summarize(trades, skips, "TWO_YEAR_SCREEN")}
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v48_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v48_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v48_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v48_skips.csv", index=False)
    print("V4.8 TWO-YEAR SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
