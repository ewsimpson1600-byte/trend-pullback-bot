"""V1.9 development-only multi-day trend-pullback option research."""

from pathlib import Path

import numpy as np
import pandas as pd

import develop_v16 as v16


RESULTS_DIR = Path("backtest_results_v19_development")
OPTION_DTE = 14
TARGET_ABS_DELTA = 0.25
OPTION_TARGET_RETURN = 0.35
OPTION_STOP_RETURN = -0.20
MAX_HOLD_SESSIONS = 3
STOP_ATR_MULTIPLE = 1.5
ENTRY_FRICTION = 0.03
EXIT_FRICTION = 0.03


def daily_indicators(df):
    d = (
        df.groupby("date", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), signal_idx=("datetime", lambda x: x.index[-1]))
        .reset_index()
    )
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    previous = d["close"].shift(1)
    true_range = pd.concat([(d["high"] - d["low"]), (d["high"] - previous).abs(), (d["low"] - previous).abs()], axis=1).max(axis=1)
    d["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def build_swing_signals(data):
    daily = {symbol: daily_indicators(frame) for symbol, frame in data.items()}
    market = daily["SPY"][["date", "close", "ema20", "ema50"]].rename(columns={"close": "spy_close", "ema20": "spy20", "ema50": "spy50"})
    qqq = daily["QQQ"][["date", "close", "ema20", "ema50"]].rename(columns={"close": "qqq_close", "ema20": "qqq20", "ema50": "qqq50"})
    market = market.merge(qqq, on="date", how="inner")
    market["bull"] = (market["spy_close"] > market["spy20"]) & (market["spy20"] > market["spy50"]) & (market["qqq_close"] > market["qqq20"]) & (market["qqq20"] > market["qqq50"])
    market["bear"] = (market["spy_close"] < market["spy20"]) & (market["spy20"] < market["spy50"]) & (market["qqq_close"] < market["qqq20"]) & (market["qqq20"] < market["qqq50"])
    rows = []
    for symbol in v16.engine.SYMBOLS:
        d = daily[symbol].merge(market[["date", "bull", "bear"]], on="date", how="left")
        d["previous_close"] = d["close"].shift(1)
        in_period = pd.to_datetime(d["date"]).between(v16.DEVELOPMENT_START.tz_localize(None), v16.DEVELOPMENT_END.tz_localize(None))
        call = in_period & d["bull"] & (d["ema20"] > d["ema50"]) & (d["close"] > d["ema50"]) & (d["low"] <= d["ema20"]) & (d["close"] >= d["ema20"]) & (d["close"] > d["open"])
        put = in_period & d["bear"] & (d["ema20"] < d["ema50"]) & (d["close"] < d["ema50"]) & (d["high"] >= d["ema20"]) & (d["close"] <= d["ema20"]) & (d["close"] < d["open"])
        for direction, regime, valid in (("CALL", "BULL_SWING", call), ("PUT", "BEAR_SWING", put)):
            for _, row in d.loc[valid].iterrows():
                idx = int(row["signal_idx"])
                rows.append({"symbol": symbol, "direction": direction, "regime": regime, "signal_idx": idx, "signal_time": data[symbol].iloc[idx]["datetime"], "atr": float(row["atr14"])})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def select_strike(spot, volatility, direction):
    years = OPTION_DTE / 365
    width = max(spot * 0.20, 5.0)
    best = None
    for strike in np.linspace(max(0.01, spot - width), spot + width, 321):
        delta = v16.option_delta(spot, strike, years, volatility, direction)
        candidate = (abs(abs(delta) - TARGET_ABS_DELTA), strike, delta)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return float(best[1]), float(best[2])


def simulate_swing(signal, df):
    entry_idx = int(signal["signal_idx"]) + 1
    if entry_idx >= len(df):
        return None
    entry_bar = df.iloc[entry_idx]
    if entry_bar["datetime"].date() <= signal["signal_time"].date():
        return None
    entry_stock = float(entry_bar["open"])
    volatility = v16.engine.estimate_volatility(df, entry_idx)
    if not np.isfinite(volatility):
        return None
    direction = signal["direction"]
    strike, delta = select_strike(entry_stock, volatility, direction)
    modeled_entry = v16.option_price(entry_stock, strike, OPTION_DTE / 365, volatility, direction) * (1 + ENTRY_FRICTION)
    target, stop = modeled_entry * (1 + OPTION_TARGET_RETURN), modeled_entry * (1 + OPTION_STOP_RETURN)
    underlying_stop = entry_stock - STOP_ATR_MULTIPLE * signal["atr"] if direction == "CALL" else entry_stock + STOP_ATR_MULTIPLE * signal["atr"]
    session_dates = list(pd.unique(df.iloc[entry_idx:]["date"]))[:MAX_HOLD_SESSIONS]
    final_idx = df.index[(df.index >= entry_idx) & df["date"].isin(session_dates)][-1]
    exit_time = exit_stock = modeled_exit = None
    exit_reason = "TIME"
    for i in range(entry_idx, final_idx + 1):
        bar = df.iloc[i]
        elapsed = (bar["datetime"] - entry_bar["datetime"]).total_seconds() / 86400
        years = max((OPTION_DTE - elapsed) / 365, 1 / (365 * 1440))
        adverse = float(bar["low"] if direction == "CALL" else bar["high"])
        favorable = float(bar["high"] if direction == "CALL" else bar["low"])
        adverse_option = v16.option_price(adverse, strike, years, volatility, direction) * (1 - EXIT_FRICTION)
        favorable_option = v16.option_price(favorable, strike, years, volatility, direction) * (1 - EXIT_FRICTION)
        underlying_hit = adverse <= underlying_stop if direction == "CALL" else adverse >= underlying_stop
        if underlying_hit or adverse_option <= stop:
            exit_reason, exit_time, exit_stock = ("UNDERLYING_STOP" if underlying_hit else "OPTION_STOP"), bar["datetime"], adverse
            modeled_exit = min(stop, adverse_option) if underlying_hit else stop
            break
        if favorable_option >= target:
            exit_reason, exit_time, exit_stock, modeled_exit = "OPTION_TARGET", bar["datetime"], favorable, target
            break
    if exit_time is None:
        bar = df.iloc[final_idx]
        exit_time, exit_stock = bar["datetime"], float(bar["close"])
        elapsed = (exit_time - entry_bar["datetime"]).total_seconds() / 86400
        modeled_exit = v16.option_price(exit_stock, strike, max((OPTION_DTE - elapsed) / 365, 1 / (365 * 1440)), volatility, direction) * (1 - EXIT_FRICTION)
    gross_cost, proceeds = modeled_entry * 100, modeled_exit * 100
    pnl = proceeds - gross_cost - 2 * v16.FEE_PER_SIDE
    return {**signal.to_dict(), "entry_time": entry_bar["datetime"], "exit_time": exit_time, "entry_stock": entry_stock, "exit_stock": exit_stock, "option_strike": strike, "option_delta": delta, "estimated_volatility": volatility, "modeled_option_entry": modeled_entry, "modeled_option_exit": modeled_exit, "contract_cost": gross_cost + v16.FEE_PER_SIDE, "trade_pnl": pnl, "option_return_pct": pnl / (gross_cost + v16.FEE_PER_SIDE) * 100, "exit_reason": exit_reason}


def run_account(signals, data):
    balance = peak = v16.STARTING_ACCOUNT
    next_available = None
    trades, skips = [], []
    for _, signal in signals.iterrows():
        if next_available is not None and signal["signal_time"] < next_available:
            continue
        trade = simulate_swing(signal, data[signal["symbol"]])
        if trade is None:
            continue
        cost = trade["contract_cost"]
        hard_risk = cost * abs(OPTION_STOP_RETURN) + v16.FEE_PER_SIDE
        reason = "INSUFFICIENT_CASH" if cost > balance else "ALLOCATION_LIMIT" if cost > balance * v16.MAX_CONTRACT_ALLOCATION else "RISK_LIMIT" if hard_risk > balance * v16.MAX_ACCOUNT_RISK else None
        if reason:
            skips.append({"signal_time": signal["signal_time"], "symbol": signal["symbol"], "reason": reason, "contract_cost": cost, "hard_risk": hard_risk, "account_balance": balance})
            continue
        balance += trade["trade_pnl"]
        peak = max(peak, balance)
        trade["account_balance"] = balance
        trade["account_drawdown_pct"] = (balance / peak - 1) * 100
        trades.append(trade)
        next_available = trade["exit_time"]
    return pd.DataFrame(trades), pd.DataFrame(skips)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v16.load_development_data()
    signals = build_swing_signals(data)
    trades, skips = run_account(signals, data)
    summary = v16.summarize(trades, skips)
    signals.to_csv(RESULTS_DIR / "v19_development_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v19_development_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v19_development_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v19_development_summary.csv", index=False)
    print("V1.9 DEVELOPMENT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("No untouched holdout was loaded.")


if __name__ == "__main__":
    main()
