"""V2.0 development-only high-volume daily momentum breakout research."""

from pathlib import Path

import pandas as pd

import develop_v16 as v16
import develop_v19 as v19


RESULTS_DIR = Path("backtest_results_v20_development")
BREAKOUT_LOOKBACK = 20
VOLUME_MULTIPLE = 1.20
MIN_CLOSE_LOCATION = 0.70


def build_momentum_signals(data):
    daily = {symbol: v19.daily_indicators(frame) for symbol, frame in data.items()}
    for d in daily.values():
        d["prior_high20"] = d["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
        d["prior_low20"] = d["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
        d["volume20"] = d["volume"].shift(1).rolling(BREAKOUT_LOOKBACK).mean()
        day_range = (d["high"] - d["low"]).replace(0, pd.NA)
        d["close_location"] = (d["close"] - d["low"]) / day_range
    market = daily["SPY"][["date", "close", "ema20", "ema50"]].rename(columns={"close": "spy_close", "ema20": "spy20", "ema50": "spy50"})
    qqq = daily["QQQ"][["date", "close", "ema20", "ema50"]].rename(columns={"close": "qqq_close", "ema20": "qqq20", "ema50": "qqq50"})
    market = market.merge(qqq, on="date", how="inner")
    market["bull"] = (market["spy_close"] > market["spy20"]) & (market["spy20"] > market["spy50"]) & (market["qqq_close"] > market["qqq20"]) & (market["qqq20"] > market["qqq50"])
    market["bear"] = (market["spy_close"] < market["spy20"]) & (market["spy20"] < market["spy50"]) & (market["qqq_close"] < market["qqq20"]) & (market["qqq20"] < market["qqq50"])
    rows = []
    for symbol in v16.engine.SYMBOLS:
        d = daily[symbol].merge(market[["date", "bull", "bear"]], on="date", how="left")
        dates = pd.to_datetime(d["date"])
        in_period = dates.between(v16.DEVELOPMENT_START.tz_localize(None), v16.DEVELOPMENT_END.tz_localize(None))
        liquid_move = d["volume"] >= VOLUME_MULTIPLE * d["volume20"]
        call = in_period & d["bull"] & liquid_move & (d["close"] > d["prior_high20"]) & (d["close_location"] >= MIN_CLOSE_LOCATION)
        put = in_period & d["bear"] & liquid_move & (d["close"] < d["prior_low20"]) & (d["close_location"] <= 1 - MIN_CLOSE_LOCATION)
        for direction, regime, valid in (("CALL", "BULL_MOMENTUM", call), ("PUT", "BEAR_MOMENTUM", put)):
            for _, row in d.loc[valid].iterrows():
                idx = int(row["signal_idx"])
                rows.append({"symbol": symbol, "direction": direction, "regime": regime, "signal_idx": idx, "signal_time": data[symbol].iloc[idx]["datetime"], "atr": float(row["atr14"]), "volume_multiple": float(row["volume"] / row["volume20"]), "close_location": float(row["close_location"])})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_time", "volume_multiple"], ascending=[True, False]).reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v16.load_development_data()
    signals = build_momentum_signals(data)
    trades, skips = v19.run_account(signals, data)
    summary = v16.summarize(trades, skips)
    signals.to_csv(RESULTS_DIR / "v20_development_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v20_development_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v20_development_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v20_development_summary.csv", index=False)
    print("V2.0 DEVELOPMENT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("No untouched holdout was loaded.")


if __name__ == "__main__":
    main()
