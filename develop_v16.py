"""V1.6 development-only symmetric opening-range continuation research.

This version is a structural response to V1.5's inadequate sample size. It
uses the already-consumed 2024-2025 development period only. No untouched
holdout is loaded by this module.

Prices are conservative Black-Scholes estimates because historical option
quotes are unavailable. A 3% adverse fill adjustment is applied on entry and
exit, plus $0.65 per contract per side.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

import develop_v13_expanded as engine


STARTING_ACCOUNT = 1000.0
RESULTS_DIR = Path("backtest_results_v16_development")
DEVELOPMENT_CACHE = Path("backtest_data_v142_oos")
DEVELOPMENT_DOWNLOAD_START = "2023-12-01"
DEVELOPMENT_DOWNLOAD_END = "2025-12-01"
DEVELOPMENT_START = pd.Timestamp("2024-01-02", tz="America/New_York")
DEVELOPMENT_END = pd.Timestamp("2025-11-26 16:00", tz="America/New_York")

SIGNAL_START = "09:45"
SIGNAL_END = "11:30"
RVOL_MIN = 1.5
TARGET_ABS_DELTA = 0.30
OPTION_DTE = 5
OPTION_TARGET_RETURN = 0.30
OPTION_STOP_RETURN = -0.20
MAX_HOLD_MINUTES = 60
STOP_ATR_MULTIPLE = 1.5
ENTRY_FRICTION = 0.03
EXIT_FRICTION = 0.03
FEE_PER_SIDE = 0.65
MAX_CONTRACT_ALLOCATION = 0.25
MAX_ACCOUNT_RISK = 0.04
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 160


def configure_period():
    engine.DOWNLOAD_START = DEVELOPMENT_DOWNLOAD_START
    engine.DOWNLOAD_END = DEVELOPMENT_DOWNLOAD_END
    engine.TEST_START = DEVELOPMENT_START
    engine.TEST_END = DEVELOPMENT_END
    engine.CACHE_DIR = DEVELOPMENT_CACHE


def load_development_data():
    configure_period()
    engine.ensure_directories()
    raw = engine.download_all_data()
    data = {symbol: engine.add_indicators(raw[symbol]) for symbol in engine.SYMBOLS}
    return data


def create_market_regime(data):
    frames = []
    for symbol, prefix in (("SPY", "spy"), ("QQQ", "qqq")):
        x = data[symbol][["datetime", "close", "ema9", "ema21", "vwap"]].copy()
        x[f"{prefix}_bull"] = (x["close"] > x["vwap"]) & (x["ema9"] > x["ema21"])
        x[f"{prefix}_bear"] = (x["close"] < x["vwap"]) & (x["ema9"] < x["ema21"])
        frames.append(x[["datetime", f"{prefix}_bull", f"{prefix}_bear"]])
    market = frames[0].merge(frames[1], on="datetime", how="inner")
    market["bull"] = market["spy_bull"] & market["qqq_bull"]
    market["bear"] = market["spy_bear"] & market["qqq_bear"]
    return market[["datetime", "bull", "bear"]]


def build_directional_signals(data, market):
    rows = []
    for symbol in engine.SYMBOLS:
        x = data[symbol].merge(market, on="datetime", how="left")
        x[["bull", "bear"]] = x[["bull", "bear"]].fillna(False)
        x["previous_close"] = x["close"].shift(1)
        x["call_breakout"] = (
            (x["close"] > x["opening_range_high"])
            & (x["previous_close"] <= x["opening_range_high"])
        )
        x["put_breakout"] = (
            (x["close"] < x["opening_range_low"])
            & (x["previous_close"] >= x["opening_range_low"])
        )
        common = (
            (x["datetime"] >= DEVELOPMENT_START)
            & (x["datetime"] <= DEVELOPMENT_END)
            & (x["time"] >= SIGNAL_START)
            & (x["time"] <= SIGNAL_END)
            & (x["rvol"] >= RVOL_MIN)
            & x["atr"].notna()
        )
        call_valid = common & x["call_breakout"] & x["bull"] & (x["close"] > x["vwap"]) & (x["ema9"] > x["ema21"])
        put_valid = common & x["put_breakout"] & x["bear"] & (x["close"] < x["vwap"]) & (x["ema9"] < x["ema21"])
        for direction, valid in (("CALL", call_valid), ("PUT", put_valid)):
            selected = x.loc[valid].copy()
            if selected.empty:
                continue
            selected = selected.sort_values("datetime").drop_duplicates("date", keep="first")
            for idx, bar in selected.iterrows():
                rows.append({
                    "symbol": symbol,
                    "direction": direction,
                    "signal_idx": int(idx),
                    "signal_time": bar["datetime"],
                    "rvol": float(bar["rvol"]),
                    "atr": float(bar["atr"]),
                    "regime": "BULL" if direction == "CALL" else "BEAR",
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_time", "rvol"], ascending=[True, False]).reset_index(drop=True)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def option_price(spot, strike, years, volatility, direction):
    volatility = max(float(volatility), 0.05)
    years = max(float(years), 1 / (365 * 1440))
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (engine.RISK_FREE_RATE + 0.5 * volatility**2) * years) / (volatility * root)
    d2 = d1 - volatility * root
    discount = strike * math.exp(-engine.RISK_FREE_RATE * years)
    if direction == "CALL":
        return max(spot * normal_cdf(d1) - discount * normal_cdf(d2), 0.01)
    return max(discount * normal_cdf(-d2) - spot * normal_cdf(-d1), 0.01)


def option_delta(spot, strike, years, volatility, direction):
    volatility = max(float(volatility), 0.05)
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (engine.RISK_FREE_RATE + 0.5 * volatility**2) * years) / (volatility * root)
    call_delta = normal_cdf(d1)
    return call_delta if direction == "CALL" else call_delta - 1.0


def select_strike(spot, volatility, direction):
    years = OPTION_DTE / 365
    width = max(spot * 0.15, 5.0)
    best = None
    for strike in np.linspace(max(0.01, spot - width), spot + width, 241):
        delta = option_delta(spot, strike, years, volatility, direction)
        candidate = (abs(abs(delta) - TARGET_ABS_DELTA), strike, delta)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return float(best[1]), float(best[2])


def simulate_trade(signal, df):
    signal_idx = int(signal["signal_idx"])
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_bar = df.iloc[entry_idx]
    if entry_bar["datetime"].date() != signal["signal_time"].date():
        return None
    direction = signal["direction"]
    entry_stock = float(entry_bar["open"])
    volatility = engine.estimate_volatility(df, entry_idx)
    if not np.isfinite(volatility):
        return None
    strike, delta = select_strike(entry_stock, volatility, direction)
    theoretical = option_price(entry_stock, strike, OPTION_DTE / 365, volatility, direction)
    modeled_entry = theoretical * (1 + ENTRY_FRICTION)
    target = modeled_entry * (1 + OPTION_TARGET_RETURN)
    stop = modeled_entry * (1 + OPTION_STOP_RETURN)
    underlying_stop = entry_stock - STOP_ATR_MULTIPLE * float(signal["atr"]) if direction == "CALL" else entry_stock + STOP_ATR_MULTIPLE * float(signal["atr"])
    final_idx = min(entry_idx + MAX_HOLD_MINUTES // 5, len(df) - 1)
    exit_reason, exit_time, exit_stock, modeled_exit = "TIME", None, None, None
    for i in range(entry_idx, final_idx + 1):
        bar = df.iloc[i]
        if bar["datetime"].date() != entry_bar["datetime"].date():
            break
        elapsed = (bar["datetime"] - entry_bar["datetime"]).total_seconds() / 60
        years = max((OPTION_DTE - elapsed / 1440) / 365, 1 / (365 * 1440))
        adverse_spot = float(bar["low"] if direction == "CALL" else bar["high"])
        favorable_spot = float(bar["high"] if direction == "CALL" else bar["low"])
        adverse_option = option_price(adverse_spot, strike, years, volatility, direction) * (1 - EXIT_FRICTION)
        favorable_option = option_price(favorable_spot, strike, years, volatility, direction) * (1 - EXIT_FRICTION)
        underlying_hit = adverse_spot <= underlying_stop if direction == "CALL" else adverse_spot >= underlying_stop
        if underlying_hit or adverse_option <= stop:
            exit_reason = "UNDERLYING_STOP" if underlying_hit else "OPTION_STOP"
            exit_time, exit_stock = bar["datetime"], adverse_spot
            modeled_exit = min(stop, adverse_option) if underlying_hit else stop
            break
        if favorable_option >= target:
            exit_reason, exit_time, exit_stock, modeled_exit = "OPTION_TARGET", bar["datetime"], favorable_spot, target
            break
    if exit_time is None:
        bar = df.iloc[final_idx]
        exit_time, exit_stock = bar["datetime"], float(bar["close"])
        elapsed = (exit_time - entry_bar["datetime"]).total_seconds() / 60
        years = max((OPTION_DTE - elapsed / 1440) / 365, 1 / (365 * 1440))
        modeled_exit = option_price(exit_stock, strike, years, volatility, direction) * (1 - EXIT_FRICTION)
    gross_cost = modeled_entry * 100
    gross_proceeds = modeled_exit * 100
    pnl = gross_proceeds - gross_cost - 2 * FEE_PER_SIDE
    return {
        **signal.to_dict(), "entry_time": entry_bar["datetime"], "exit_time": exit_time,
        "entry_stock": entry_stock, "exit_stock": exit_stock, "underlying_stop": underlying_stop,
        "option_strike": strike, "option_delta": delta, "estimated_volatility": volatility,
        "modeled_option_entry": modeled_entry, "modeled_option_exit": modeled_exit,
        "contract_cost": gross_cost + FEE_PER_SIDE, "trade_pnl": pnl,
        "option_return_pct": pnl / (gross_cost + FEE_PER_SIDE) * 100,
        "exit_reason": exit_reason,
    }


def run_account(signals, data):
    balance, peak, next_available = STARTING_ACCOUNT, STARTING_ACCOUNT, None
    trades, skips = [], []
    for _, signal in signals.iterrows():
        if next_available is not None and signal["signal_time"] < next_available:
            continue
        trade = simulate_trade(signal, data[signal["symbol"]])
        if trade is None:
            continue
        cost = trade["contract_cost"]
        hard_risk = cost * abs(OPTION_STOP_RETURN) + FEE_PER_SIDE
        reason = None
        if cost > balance:
            reason = "INSUFFICIENT_CASH"
        elif cost > balance * MAX_CONTRACT_ALLOCATION:
            reason = "ALLOCATION_LIMIT"
        elif hard_risk > balance * MAX_ACCOUNT_RISK:
            reason = "RISK_LIMIT"
        if reason:
            skips.append({"signal_time": signal["signal_time"], "symbol": signal["symbol"], "direction": signal["direction"], "reason": reason, "contract_cost": cost, "hard_risk": hard_risk, "account_balance": balance})
            continue
        balance += trade["trade_pnl"]
        peak = max(peak, balance)
        trade["account_balance"] = balance
        trade["account_drawdown_pct"] = (balance / peak - 1) * 100
        trades.append(trade)
        next_available = trade["exit_time"]
    return pd.DataFrame(trades), pd.DataFrame(skips)


def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return float("inf") if losses == 0 and gains > 0 else (gains / losses if losses else 0.0)


def concentration_pct(trades, field):
    if trades.empty or trades["trade_pnl"].sum() <= 0:
        return np.nan
    profits = trades.loc[trades["trade_pnl"] > 0].groupby(field)["trade_pnl"].sum()
    return float(profits.max() / profits.sum() * 100) if len(profits) else np.nan


def summarize(trades, skips):
    if trades.empty:
        return {"trades": 0, "ending_account": STARTING_ACCOUNT, "development_pass": False}
    returns = trades["option_return_pct"].astype(float)
    ci_low, ci_high = bootstrap_mean_ci(returns)
    work = trades.copy()
    work["year"] = pd.to_datetime(work["entry_time"], utc=True).dt.year
    work["month"] = pd.to_datetime(work["entry_time"], utc=True).dt.strftime("%Y-%m")
    year_pnl = work.groupby("year")["trade_pnl"].sum()
    direction_pnl = work.groupby("direction")["trade_pnl"].sum()
    summary = {
        "trades": len(work), "wins": int((returns > 0).sum()),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "avg_option_return_pct": float(returns.mean()), "profit_factor": profit_factor(returns),
        "bootstrap_mean_95_ci_low_pct": ci_low, "bootstrap_mean_95_ci_high_pct": ci_high,
        "ending_account": float(work.iloc[-1]["account_balance"]),
        "account_return_pct": float((work.iloc[-1]["account_balance"] / STARTING_ACCOUNT - 1) * 100),
        "account_max_drawdown_pct": float(work["account_drawdown_pct"].min()),
        "max_ticker_profit_contribution_pct": concentration_pct(work, "symbol"),
        "max_month_profit_contribution_pct": concentration_pct(work, "month"),
        "positive_years": int((year_pnl > 0).sum()), "years_tested": int(len(year_pnl)),
        "positive_directions": int((direction_pnl > 0).sum()), "directions_tested": int(len(direction_pnl)),
        "skipped_signals": len(skips),
    }
    summary["development_pass"] = bool(
        summary["trades"] >= 50 and summary["bootstrap_mean_95_ci_low_pct"] > 0
        and summary["profit_factor"] >= 1.5 and summary["account_max_drawdown_pct"] >= -25
        and summary["max_ticker_profit_contribution_pct"] <= 35
        and summary["max_month_profit_contribution_pct"] <= 35
        and summary["positive_years"] == summary["years_tested"]
        and summary["positive_directions"] == summary["directions_tested"]
    )
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_development_data()
    market = create_market_regime(data)
    signals = build_directional_signals(data, market)
    trades, skips = run_account(signals, data)
    work = trades.copy()
    if not work.empty:
        work["month"] = pd.to_datetime(work["entry_time"], utc=True).dt.strftime("%Y-%m")
        work["year"] = pd.to_datetime(work["entry_time"], utc=True).dt.year
    summary = summarize(trades, skips)
    signals.to_csv(RESULTS_DIR / "v16_development_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v16_development_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v16_development_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v16_development_summary.csv", index=False)
    if not work.empty:
        work.groupby("symbol")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v16_by_ticker.csv")
        work.groupby("month")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v16_by_month.csv")
        work.groupby("year")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v16_by_year.csv")
        work.groupby("direction")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / "v16_by_direction.csv")
    print("V1.6 DEVELOPMENT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("No untouched holdout was loaded.")


if __name__ == "__main__":
    main()
