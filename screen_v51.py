"""V5.1 two-year screen for a diversified weekend ETF calendar effect.

On the second-to-last observed session of each ISO week, the rule chooses the
least-used SPY/QQQ/IWM ETF above its 200-day EMA. It enters at the next open
(normally Friday) and exits after two sessions (normally Monday close), unless
a 1.5 ATR protective stop is hit. This is a fixed 2024-2025 screen under the
V4.7 funnel, inherits all cash-account controls, and cannot place orders.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V5.1"
VARIANT = "DIVERSIFIED_WEEKEND_EFFECT"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("WEEKEND_EFFECT", 1.5, 100.0, 2)
RESULTS_DIR = Path("screen_results_v51")


def weekly_signal_dates(frame, start, end):
    work = frame.loc[frame["date"].between(start, end), ["date"]].copy()
    if work.empty:
        return []
    iso = work["date"].dt.isocalendar()
    work["week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    dates = []
    for _, group in work.groupby("week", sort=False):
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
    return (idx, row) if float(row["close"]) > float(row["ema200"]) else None


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    usage = {symbol: 0 for symbol in SYMBOLS}
    rows = []
    for date in weekly_signal_dates(data["SPY"], start, end):
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
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v51_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v51_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v51_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v51_skips.csv", index=False)
    print("V5.1 TWO-YEAR SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
