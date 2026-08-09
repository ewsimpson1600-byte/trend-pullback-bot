"""V1.7 development-only 30-minute OR pullback/reclaim research.

V1.6 showed that unconfirmed opening-range breaks created enough trades but
had strongly negative expectancy, especially immediately after 09:45. V1.7
keeps the same option model, exits, costs, and account limits, but requires a
30-minute opening range and a bounded pullback/reclaim before entry.
"""

from pathlib import Path

import pandas as pd

import develop_v16 as v16


RESULTS_DIR = Path("backtest_results_v17_development")
BREAKOUT_START = "10:00"
BREAKOUT_END = "11:15"
SIGNAL_END = "11:30"
MAX_RETEST_BARS = 3
RETEST_OUTSIDE_ATR = 0.25
RETEST_INSIDE_ATR = 0.75


def add_30_minute_range(df):
    x = df.copy()
    opening = (
        x.loc[(x["time"] >= "09:30") & (x["time"] < "10:00")]
        .groupby("date")
        .agg(or30_high=("high", "max"), or30_low=("low", "min"))
    )
    return x.merge(opening, left_on="date", right_index=True, how="left")


def build_retest_signals(data, market):
    rows = []
    for symbol in v16.engine.SYMBOLS:
        df = add_30_minute_range(data[symbol])
        x = df.merge(market, on="datetime", how="left")
        x[["bull", "bear"]] = x[["bull", "bear"]].fillna(False)
        x["previous_close"] = x["close"].shift(1)
        common = (
            (x["datetime"] >= v16.DEVELOPMENT_START)
            & (x["datetime"] <= v16.DEVELOPMENT_END)
            & (x["time"] >= BREAKOUT_START)
            & (x["time"] <= BREAKOUT_END)
            & (x["rvol"] >= v16.RVOL_MIN)
            & x["atr"].notna()
        )
        definitions = (
            (
                "CALL", "BULL", "or30_high",
                common & (x["close"] > x["or30_high"]) & (x["previous_close"] <= x["or30_high"])
                & x["bull"] & (x["close"] > x["vwap"]) & (x["ema9"] > x["ema21"]),
            ),
            (
                "PUT", "BEAR", "or30_low",
                common & (x["close"] < x["or30_low"]) & (x["previous_close"] >= x["or30_low"])
                & x["bear"] & (x["close"] < x["vwap"]) & (x["ema9"] < x["ema21"]),
            ),
        )
        used_dates = set()
        for direction, regime, level_column, valid in definitions:
            for breakout_idx, breakout in x.loc[valid].sort_values("datetime").iterrows():
                day = breakout["date"]
                if day in used_dates:
                    continue
                level = float(breakout[level_column])
                atr = float(breakout["atr"])
                last = min(int(breakout_idx) + MAX_RETEST_BARS, len(x) - 1)
                for i in range(int(breakout_idx) + 1, last + 1):
                    bar = x.iloc[i]
                    if bar["date"] != day or bar["time"] > SIGNAL_END:
                        break
                    if direction == "CALL":
                        touched = level - RETEST_INSIDE_ATR * atr <= float(bar["low"]) <= level + RETEST_OUTSIDE_ATR * atr
                        reclaimed = float(bar["close"]) > level and float(bar["close"]) > float(bar["vwap"]) and bool(bar["bull"])
                    else:
                        touched = level - RETEST_OUTSIDE_ATR * atr <= float(bar["high"]) <= level + RETEST_INSIDE_ATR * atr
                        reclaimed = float(bar["close"]) < level and float(bar["close"]) < float(bar["vwap"]) and bool(bar["bear"])
                    if touched and reclaimed:
                        rows.append({
                            "symbol": symbol, "direction": direction, "signal_idx": int(i),
                            "signal_time": bar["datetime"], "breakout_time": breakout["datetime"],
                            "rvol": float(breakout["rvol"]), "atr": float(bar["atr"]), "regime": regime,
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
    signals = build_retest_signals(data, market)
    trades, skips = v16.run_account(signals, data)
    work = trades.copy()
    if not work.empty:
        work["month"] = pd.to_datetime(work["entry_time"], utc=True).dt.strftime("%Y-%m")
        work["year"] = pd.to_datetime(work["entry_time"], utc=True).dt.year
    summary = v16.summarize(trades, skips)
    signals.to_csv(RESULTS_DIR / "v17_development_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v17_development_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v17_development_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v17_development_summary.csv", index=False)
    if not work.empty:
        work.groupby("symbol")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v17_by_ticker.csv")
        work.groupby("month")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v17_by_month.csv")
        work.groupby("year")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v17_by_year.csv")
        work.groupby("direction")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v17_by_direction.csv")
    print("V1.7 DEVELOPMENT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("No untouched holdout was loaded.")


if __name__ == "__main__":
    main()
