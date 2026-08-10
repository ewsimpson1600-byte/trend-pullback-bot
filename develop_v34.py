"""V3.4 alternating SPY-core and sector-diversifier trend research.

V3.1 established a profitable SPY absolute-trend benchmark but was fully
concentrated; V3.3 diversified broadly but lacked consistency. V3.4 combines
their independent ideas without simultaneous positions: alternating monthly
slots are reserved for SPY and for the least-used eligible sector. Every asset
must have positive six-month momentum above a rising 200-day EMA. Development
is 2010-2025 before the still-sealed 2002-2009 stress holdout can open once.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v25 as v25
import develop_v32 as v32
import develop_v33 as v33


VERSION = "V3.4"
VARIANT = "ALTERNATING_SPY_CORE_SECTOR_TREND"
SECTOR_SYMBOLS = v33.SYMBOLS
SYMBOLS = ("SPY",) + SECTOR_SYMBOLS
FAMILY = v21.Family("ALTERNATING_TREND", 3.0, 100.0, 20)
RESULTS_DIR = Path("backtest_results_v34")


def load_data():
    broad = v32.load_data()
    sectors = v33.load_data()
    return {"SPY": broad["SPY"], **sectors}


def make_signal(symbol, idx, row, date, slot):
    return {"family": FAMILY.name, "variant": VARIANT, "symbol": symbol,
            "signal_idx": idx, "signal_time": date, "atr": float(row["atr14"]),
            "strength": 1.0, "slot": slot}


def build_signals(data, start, end):
    usage = {symbol: 0 for symbol in SECTOR_SYMBOLS}
    rows = []
    for slot_index, date in enumerate(v25.monthly_signal_dates(data["SPY"], start, end)):
        if slot_index % 2 == 0:
            result = v33.eligible_on_date(data["SPY"], date)
            if result is not None:
                idx, row = result
                rows.append(make_signal("SPY", idx, row, date, "SPY_CORE"))
            continue
        eligible = []
        for symbol in SECTOR_SYMBOLS:
            result = v33.eligible_on_date(data[symbol], date)
            if result is not None:
                eligible.append((usage[symbol], symbol, result))
        if eligible:
            _, symbol, (idx, row) = min(eligible, key=lambda item: (item[0], item[1]))
            usage[symbol] += 1
            rows.append(make_signal(symbol, idx, row, date, "SECTOR_DIVERSIFIER"))
    return pd.DataFrame(rows, columns=["family", "variant", "symbol", "signal_idx", "signal_time", "atr", "strength", "slot"])


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_{suffix}.csv")


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    signals = build_signals(data, v32.DEVELOPMENT_START, v32.DEVELOPMENT_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    dev = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY),
           **v32.summarize_period(trades, skips, "DEVELOPMENT", v32.DEVELOPMENT_START, v32.DEVELOPMENT_END)}
    write_period("v34_development", signals, trades, skips, dev)
    if not dev["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "NEW_DEVELOPMENT_GATES_FAILED", **dev}]).to_csv(RESULTS_DIR / "v34_candidate.csv", index=False)
        print(pd.DataFrame([dev]).to_string(index=False)); print("V3.4 failed development; historical holdout stayed sealed."); return
    pd.DataFrame([{"status": "LOCKED", **dev}]).to_csv(RESULTS_DIR / "v34_candidate.csv", index=False)
    signals = build_signals(data, v32.VALIDATION_START, v32.VALIDATION_END)
    trades, skips = v21.run_account(signals, data, FAMILY)
    val = {"version": VERSION, "variant": VARIANT, "universe_size": len(SYMBOLS), **asdict(FAMILY),
           **v32.summarize_period(trades, skips, "HISTORICAL_VALIDATION", v32.VALIDATION_START, v32.VALIDATION_END)}
    write_period("v34_historical_validation", signals, trades, skips, val)
    print("V3.4 LOCKED HISTORICAL VALIDATION RESULT"); print(pd.DataFrame([val]).to_string(index=False))


if __name__ == "__main__":
    main()
