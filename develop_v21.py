"""V2.1 daily ETF share research for a $1,000 cash account.

Three long-only signal families are compared on 2010-2017 development data.
The best mechanically eligible family is locked before one evaluation on
2018-2025 validation data. Trades use integer shares, next-session entries,
overnight-gap-aware exits, conservative slippage, per-share costs, and no
margin or leverage. This module cannot submit orders.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import requests


SYMBOLS = ("SPY", "QQQ", "IWM")
DOWNLOAD_START = "2009-01-01"
DOWNLOAD_END = "2026-01-01"
DEVELOPMENT_START = pd.Timestamp("2010-01-04")
DEVELOPMENT_END = pd.Timestamp("2017-12-29")
VALIDATION_START = pd.Timestamp("2018-01-02")
VALIDATION_END = pd.Timestamp("2025-12-31")

STARTING_ACCOUNT = 1000.0
MAX_ALLOCATION = 0.80
MAX_RISK = 0.02
SLIPPAGE_RATE = 0.0002
PER_SHARE_COST = 0.01
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 210

MIN_DEVELOPMENT_TRADES = 40
MIN_DEVELOPMENT_PROFIT_FACTOR = 1.15
MAX_DEVELOPMENT_DRAWDOWN = -25.0

API_URL = "https://api.twelvedata.com/time_series"
CACHE_DIR = Path("backtest_data_v21_daily")
RESULTS_DIR = Path("backtest_results_v21")


@dataclass(frozen=True)
class Family:
    name: str
    stop_atr: float
    target_atr: float
    max_hold_sessions: int


FAMILIES = (
    Family("TREND_PULLBACK", 2.0, 3.0, 10),
    Family("MEAN_REVERSION", 2.0, 1.5, 5),
    Family("VOLUME_BREAKOUT", 2.0, 4.0, 15),
)


def require_api_key():
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing")
    return key


def fetch_daily(symbol, api_key):
    response = requests.get(
        API_URL,
        params={
            "symbol": symbol,
            "interval": "1day",
            "start_date": DOWNLOAD_START,
            "end_date": DOWNLOAD_END,
            "timezone": "America/New_York",
            "apikey": api_key,
            "format": "JSON",
            "order": "ASC",
            "outputsize": 5000,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error for {symbol}: {payload}")
    values = payload.get("values") or []
    if not values:
        raise RuntimeError(f"No daily data returned for {symbol}")
    frame = pd.DataFrame(values)
    frame["date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame[["date", "open", "high", "low", "close", "volume"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = None
    data = {}
    for symbol in SYMBOLS:
        path = CACHE_DIR / f"{symbol}_1day.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
        else:
            key = key or require_api_key()
            frame = fetch_daily(symbol, key)
            frame.to_csv(path, index=False)
            time.sleep(8)
        data[symbol] = add_indicators(frame)
    return data


def add_indicators(frame):
    x = frame.copy().sort_values("date").reset_index(drop=True)
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=200, adjust=False).mean()
    x["sma5"] = x["close"].rolling(5).mean()
    x["sma20"] = x["close"].rolling(20).mean()
    x["std20"] = x["close"].rolling(20).std()
    x["bb_lower"] = x["sma20"] - 2.0 * x["std20"]
    x["prior_high20"] = x["high"].shift(1).rolling(20).max()
    x["volume20"] = x["volume"].shift(1).rolling(20).mean()
    previous = x["close"].shift(1)
    true_range = pd.concat(
        [x["high"] - x["low"], (x["high"] - previous).abs(), (x["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    x["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 2, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi2"] = (100 - 100 / (1 + rs)).fillna(100)
    return x


def build_signals(data, family, start, end):
    rows = []
    for symbol, x in data.items():
        period = x["date"].between(start, end)
        ready = period & x[["atr14", "ema200", "prior_high20", "volume20"]].notna().all(axis=1)
        if family.name == "TREND_PULLBACK":
            valid = (
                ready
                & (x["close"] > x["ema200"])
                & (x["ema20"] > x["ema50"])
                & (x["low"] <= x["ema20"])
                & (x["close"] >= x["ema20"])
                & (x["close"] > x["open"])
                & x["rsi2"].between(35, 75)
            )
            strength = (x["ema20"] / x["ema50"] - 1) - (x["close"] / x["ema20"] - 1).abs()
        elif family.name == "MEAN_REVERSION":
            valid = (
                ready
                & (x["close"] > x["ema200"])
                & (x["close"] < x["bb_lower"])
                & (x["rsi2"] <= 10)
            )
            strength = (10 - x["rsi2"]) + (x["bb_lower"] / x["close"] - 1) * 100
        elif family.name == "VOLUME_BREAKOUT":
            volume_ratio = x["volume"] / x["volume20"]
            valid = (
                ready
                & (x["close"] > x["ema200"])
                & (x["ema50"] > x["ema200"])
                & (x["close"] > x["prior_high20"])
                & (volume_ratio >= 1.20)
            )
            strength = (x["close"] / x["prior_high20"] - 1) * 100 + volume_ratio
        else:
            raise ValueError(f"Unknown family: {family.name}")
        for idx in x.index[valid]:
            rows.append(
                {
                    "family": family.name,
                    "symbol": symbol,
                    "signal_idx": int(idx),
                    "signal_time": x.at[idx, "date"],
                    "atr": float(x.at[idx, "atr14"]),
                    "strength": float(strength.at[idx]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["signal_time", "strength", "symbol"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def price_with_slippage(price, side):
    multiplier = 1 + SLIPPAGE_RATE if side == "BUY" else 1 - SLIPPAGE_RATE
    return float(price) * multiplier


def simulate_trade(signal, frame, family):
    entry_idx = int(signal["signal_idx"]) + 1
    if entry_idx >= len(frame):
        return None
    entry_bar = frame.iloc[entry_idx]
    entry = price_with_slippage(entry_bar["open"], "BUY")
    atr = float(signal["atr"])
    stop = entry - family.stop_atr * atr
    target = entry + family.target_atr * atr
    final_idx = min(entry_idx + family.max_hold_sessions - 1, len(frame) - 1)
    exit_price = None
    exit_reason = "TIME"
    exit_idx = final_idx
    for idx in range(entry_idx, final_idx + 1):
        bar = frame.iloc[idx]
        if float(bar["open"]) <= stop:
            exit_price, exit_reason, exit_idx = price_with_slippage(bar["open"], "SELL"), "STOP_GAP", idx
            break
        if float(bar["open"]) >= target:
            exit_price, exit_reason, exit_idx = price_with_slippage(bar["open"], "SELL"), "TARGET_GAP", idx
            break
        if float(bar["low"]) <= stop:
            exit_price, exit_reason, exit_idx = price_with_slippage(stop, "SELL"), "STOP", idx
            break
        if float(bar["high"]) >= target:
            exit_price, exit_reason, exit_idx = price_with_slippage(target, "SELL"), "TARGET", idx
            break
        if family.name == "MEAN_REVERSION" and idx > entry_idx and float(bar["close"]) >= float(bar["sma5"]):
            exit_price, exit_reason, exit_idx = price_with_slippage(bar["close"], "SELL"), "MEAN_EXIT", idx
            break
    if exit_price is None:
        exit_price = price_with_slippage(frame.iloc[final_idx]["close"], "SELL")
    return {
        **signal.to_dict(),
        "entry_time": entry_bar["date"],
        "exit_time": frame.iloc[exit_idx]["date"],
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_price": stop,
        "target_price": target,
        "risk_per_share": entry - stop,
        "exit_reason": exit_reason,
        "hold_sessions": exit_idx - entry_idx + 1,
    }


def position_size(balance, entry_price, risk_per_share):
    cash_shares = int((balance * MAX_ALLOCATION) // (entry_price + PER_SHARE_COST))
    risk_shares = int((balance * MAX_RISK) // max(risk_per_share + 2 * PER_SHARE_COST, 0.01))
    return max(0, min(cash_shares, risk_shares))


def run_account(signals, data, family, starting_account=STARTING_ACCOUNT):
    balance = float(starting_account)
    peak = balance
    next_available = None
    trades, skips = [], []
    for signal_time, candidates in signals.groupby("signal_time", sort=True):
        if next_available is not None and signal_time <= next_available:
            continue
        accepted = False
        for _, signal in candidates.sort_values(["strength", "symbol"], ascending=[False, True]).iterrows():
            trade = simulate_trade(signal, data[signal["symbol"]], family)
            if trade is None:
                continue
            shares = position_size(balance, trade["entry_price"], trade["risk_per_share"])
            if shares < 1:
                skips.append(
                    {
                        "family": family.name,
                        "signal_time": signal_time,
                        "symbol": signal["symbol"],
                        "reason": "NO_AFFORDABLE_RISK_SIZED_SHARE",
                        "account_balance": balance,
                        "entry_price": trade["entry_price"],
                        "risk_per_share": trade["risk_per_share"],
                    }
                )
                continue
            entry_cost = shares * (trade["entry_price"] + PER_SHARE_COST)
            pnl = shares * (trade["exit_price"] - trade["entry_price"] - 2 * PER_SHARE_COST)
            if entry_cost > balance:
                raise AssertionError("Cash-only sizing allowed an unaffordable trade")
            balance += pnl
            peak = max(peak, balance)
            trade.update(
                {
                    "shares": shares,
                    "entry_cost": entry_cost,
                    "trade_pnl": pnl,
                    "trade_return_pct": pnl / entry_cost * 100,
                    "account_balance": balance,
                    "account_drawdown_pct": (balance / peak - 1) * 100,
                }
            )
            trades.append(trade)
            next_available = trade["exit_time"]
            accepted = True
            break
        if accepted:
            continue
    return pd.DataFrame(trades), pd.DataFrame(skips)


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float("inf") if losses == 0 and gains > 0 else (float(gains / losses) if losses else 0.0)


def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def profit_concentration(trades, field):
    winners = trades.loc[trades["trade_pnl"] > 0].copy()
    if winners.empty:
        return np.nan
    grouped = winners.groupby(field)["trade_pnl"].sum()
    return float(grouped.max() / grouped.sum() * 100)


def summarize(trades, skips, period):
    if trades.empty:
        return {"period": period, "trades": 0, "ending_account": STARTING_ACCOUNT, "pass": False}
    work = trades.copy()
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    returns = work["trade_return_pct"].astype(float)
    ci_low, ci_high = bootstrap_mean_ci(returns)
    year_pnl = work.groupby("year")["trade_pnl"].sum()
    result = {
        "period": period,
        "trades": len(work),
        "wins": int((returns > 0).sum()),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "avg_trade_return_pct": float(returns.mean()),
        "median_trade_return_pct": float(returns.median()),
        "profit_factor": profit_factor(work["trade_pnl"]),
        "bootstrap_mean_95_ci_low_pct": ci_low,
        "bootstrap_mean_95_ci_high_pct": ci_high,
        "ending_account": float(work.iloc[-1]["account_balance"]),
        "account_return_pct": float((work.iloc[-1]["account_balance"] / STARTING_ACCOUNT - 1) * 100),
        "account_max_drawdown_pct": float(work["account_drawdown_pct"].min()),
        "max_ticker_profit_contribution_pct": profit_concentration(work, "symbol"),
        "max_month_profit_contribution_pct": profit_concentration(work, "month"),
        "positive_years": int((year_pnl > 0).sum()),
        "years_tested": int(len(year_pnl)),
        "skipped_signals": len(skips),
    }
    if period == "DEVELOPMENT":
        result["pass"] = bool(
            result["trades"] >= MIN_DEVELOPMENT_TRADES
            and result["account_return_pct"] > 0
            and result["profit_factor"] >= MIN_DEVELOPMENT_PROFIT_FACTOR
            and result["account_max_drawdown_pct"] >= MAX_DEVELOPMENT_DRAWDOWN
            and result["positive_years"] >= max(1, result["years_tested"] - 2)
        )
    else:
        result["pass"] = bool(
            result["trades"] >= 50
            and result["bootstrap_mean_95_ci_low_pct"] > 0
            and result["profit_factor"] >= 1.50
            and result["account_max_drawdown_pct"] >= -25
            and result["max_ticker_profit_contribution_pct"] <= 60
            and result["max_month_profit_contribution_pct"] <= 35
            and result["positive_years"] >= max(1, result["years_tested"] - 1)
        )
    return result


def candidate_score(summary):
    drawdown = max(abs(float(summary["account_max_drawdown_pct"])), 1.0)
    return float(summary["account_return_pct"]) * float(summary["profit_factor"]) / drawdown


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    work.groupby("symbol")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_ticker.csv")
    work.groupby("month")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_month.csv")
    work.groupby("year")["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(RESULTS_DIR / f"{prefix}_by_year.csv")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    grid = []
    development_outputs = {}
    for family in FAMILIES:
        signals = build_signals(data, family, DEVELOPMENT_START, DEVELOPMENT_END)
        trades, skips = run_account(signals, data, family)
        summary = {"family": family.name, **asdict(family), **summarize(trades, skips, "DEVELOPMENT")}
        summary["score"] = candidate_score(summary) if summary["trades"] else float("-inf")
        grid.append(summary)
        development_outputs[family.name] = (signals, trades, skips)
    grid_frame = pd.DataFrame(grid).sort_values("score", ascending=False)
    grid_frame.to_csv(RESULTS_DIR / "v21_development_grid.csv", index=False)
    eligible = grid_frame.loc[grid_frame["pass"]]
    if eligible.empty:
        pd.DataFrame([{"status": "REJECTED", "reason": "NO_DEVELOPMENT_FAMILY_PASSED"}]).to_csv(
            RESULTS_DIR / "v21_candidate.csv", index=False
        )
        print(grid_frame.to_string(index=False))
        print("No family passed development; validation was not opened.")
        return
    selected_name = str(eligible.iloc[0]["family"])
    family = next(item for item in FAMILIES if item.name == selected_name)
    pd.DataFrame([{"status": "SELECTED", **asdict(family)}]).to_csv(RESULTS_DIR / "v21_candidate.csv", index=False)
    dev_signals, dev_trades, dev_skips = development_outputs[selected_name]
    dev_signals.to_csv(RESULTS_DIR / "v21_selected_development_signals.csv", index=False)
    dev_trades.to_csv(RESULTS_DIR / "v21_selected_development_trades.csv", index=False)
    dev_skips.to_csv(RESULTS_DIR / "v21_selected_development_skips.csv", index=False)
    save_breakdowns("v21_selected_development", dev_trades)
    validation_signals = build_signals(data, family, VALIDATION_START, VALIDATION_END)
    validation_trades, validation_skips = run_account(validation_signals, data, family)
    validation_summary = {"family": family.name, **asdict(family), **summarize(validation_trades, validation_skips, "VALIDATION")}
    pd.DataFrame([validation_summary]).to_csv(RESULTS_DIR / "v21_validation_summary.csv", index=False)
    validation_signals.to_csv(RESULTS_DIR / "v21_validation_signals.csv", index=False)
    validation_trades.to_csv(RESULTS_DIR / "v21_validation_trades.csv", index=False)
    validation_skips.to_csv(RESULTS_DIR / "v21_validation_skips.csv", index=False)
    save_breakdowns("v21_validation", validation_trades)
    print("V2.1 SELECTED FAMILY")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
