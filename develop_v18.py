"""V1.8 development-only opening-range false-break reversal research."""

from pathlib import Path

import pandas as pd

import develop_v16 as v16
import develop_v17 as v17


RESULTS_DIR = Path("backtest_results_v18_development")
BREAKOUT_START = "10:00"
BREAKOUT_END = "11:15"
SIGNAL_END = "11:30"
MAX_FAILURE_BARS = 3
MIN_EXTENSION_ATR = 0.10
MAX_EXTENSION_ATR = 1.00
REENTRY_ATR = 0.05


def build_false_break_signals(data, market):
    rows = []
    for symbol in v16.engine.SYMBOLS:
        x = v17.add_30_minute_range(data[symbol]).merge(market, on="datetime", how="left")
        x[["bull", "bear"]] = x[["bull", "bear"]].fillna(False)
        used_dates = set()
        common = (
            (x["datetime"] >= v16.DEVELOPMENT_START) & (x["datetime"] <= v16.DEVELOPMENT_END)
            & (x["time"] >= BREAKOUT_START) & (x["time"] <= BREAKOUT_END)
            & (x["rvol"] >= v16.RVOL_MIN) & x["atr"].notna()
        )
        upside = common & (x["close"] >= x["or30_high"] + MIN_EXTENSION_ATR * x["atr"]) & (x["close"] <= x["or30_high"] + MAX_EXTENSION_ATR * x["atr"])
        downside = common & (x["close"] <= x["or30_low"] - MIN_EXTENSION_ATR * x["atr"]) & (x["close"] >= x["or30_low"] - MAX_EXTENSION_ATR * x["atr"])
        for failed_side, direction, regime, valid in (
            ("UPSIDE", "PUT", "REVERSAL_BEAR", upside),
            ("DOWNSIDE", "CALL", "REVERSAL_BULL", downside),
        ):
            for breakout_idx, breakout in x.loc[valid].sort_values("datetime").iterrows():
                day = breakout["date"]
                if day in used_dates:
                    continue
                last = min(int(breakout_idx) + MAX_FAILURE_BARS, len(x) - 1)
                for i in range(int(breakout_idx) + 1, last + 1):
                    bar = x.iloc[i]
                    if bar["date"] != day or bar["time"] > SIGNAL_END:
                        break
                    atr = float(bar["atr"])
                    if failed_side == "UPSIDE":
                        reentered = float(bar["close"]) <= float(bar["or30_high"]) - REENTRY_ATR * atr
                        confirmed = float(bar["close"]) < float(bar["ema9"]) and not bool(bar["bull"])
                    else:
                        reentered = float(bar["close"]) >= float(bar["or30_low"]) + REENTRY_ATR * atr
                        confirmed = float(bar["close"]) > float(bar["ema9"]) and not bool(bar["bear"])
                    if reentered and confirmed:
                        rows.append({
                            "symbol": symbol, "direction": direction, "regime": regime,
                            "failed_side": failed_side, "signal_idx": int(i), "signal_time": bar["datetime"],
                            "breakout_time": breakout["datetime"], "rvol": float(breakout["rvol"]),
                            "atr": atr,
                        })
                        used_dates.add(day)
                        break
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_time", "rvol"], ascending=[True, False]).reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v16.load_development_data()
    market = v16.create_market_regime(data)
    signals = build_false_break_signals(data, market)
    trades, skips = v16.run_account(signals, data)
    work = trades.copy()
    if not work.empty:
        work["month"] = pd.to_datetime(work["entry_time"], utc=True).dt.strftime("%Y-%m")
        work["year"] = pd.to_datetime(work["entry_time"], utc=True).dt.year
    summary = v16.summarize(trades, skips)
    signals.to_csv(RESULTS_DIR / "v18_development_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v18_development_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v18_development_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v18_development_summary.csv", index=False)
    if not work.empty:
        work.groupby("symbol")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v18_by_ticker.csv")
        work.groupby("month")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v18_by_month.csv")
        work.groupby("year")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v18_by_year.csv")
        work.groupby("direction")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v18_by_direction.csv")
    print("V1.8 DEVELOPMENT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("No untouched holdout was loaded.")


if __name__ == "__main__":
    main()
