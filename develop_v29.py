"""V2.9 early-failure-exit ETF mean-reversion research.

V2.6 had positive expectancy and low drawdown, but rare full stops caused
three losing development years. V2.9 preserves its breadth-filtered signals,
entry geometry, target, catastrophe stop, sizing, and costs. It makes one
predeclared exit change: after all normal stop/target/mean-exit checks, close
the position at the second session's close when price is not above entry.
Development is 2010-2017; 2018-2025 remains sealed unless every frozen gate
passes.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22
import develop_v24 as v24
import develop_v26 as v26


VERSION = "V2.9"
VARIANT = "EARLY_FAILURE_EXIT_MEAN_REVERSION"
FAMILY = v26.FAMILY
EARLY_EXIT_SESSION = 2
RESULTS_DIR = Path("backtest_results_v29")


def build_signals(data, start, end):
    signals = v26.build_signals(data, start, end)
    if signals.empty:
        return signals
    signals = signals.copy()
    signals["variant"] = VARIANT
    return signals


def simulate_trade(signal, frame, family):
    entry_idx = int(signal["signal_idx"]) + 1
    if entry_idx >= len(frame):
        return None
    entry_bar = frame.iloc[entry_idx]
    entry = v21.price_with_slippage(entry_bar["open"], "BUY")
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
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(bar["open"], "SELL"), "STOP_GAP", idx
            break
        if float(bar["open"]) >= target:
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(bar["open"], "SELL"), "TARGET_GAP", idx
            break
        if float(bar["low"]) <= stop:
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(stop, "SELL"), "STOP", idx
            break
        if float(bar["high"]) >= target:
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(target, "SELL"), "TARGET", idx
            break
        if idx > entry_idx and float(bar["close"]) >= float(bar["sma5"]):
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(bar["close"], "SELL"), "MEAN_EXIT", idx
            break
        if idx == entry_idx + EARLY_EXIT_SESSION - 1 and float(bar["close"]) <= entry:
            exit_price, exit_reason, exit_idx = v21.price_with_slippage(bar["close"], "SELL"), "EARLY_FAILURE", idx
            break
    if exit_price is None:
        exit_price = v21.price_with_slippage(frame.iloc[final_idx]["close"], "SELL")
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


def run_account(signals, data, family, starting_account=v21.STARTING_ACCOUNT):
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
            shares = v21.position_size(balance, trade["entry_price"], trade["risk_per_share"])
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
            entry_cost = shares * (trade["entry_price"] + v21.PER_SHARE_COST)
            pnl = shares * (trade["exit_price"] - trade["entry_price"] - 2 * v21.PER_SHARE_COST)
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


def save_breakdowns(prefix, trades):
    if trades.empty:
        return
    work = trades.copy()
    work["month"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m")
    work["year"] = pd.to_datetime(work["entry_time"]).dt.year
    for field, suffix in (("symbol", "ticker"), ("month", "month"), ("year", "year")):
        work.groupby(field)["trade_pnl"].agg(["count", "sum", "mean"]).to_csv(
            RESULTS_DIR / f"{prefix}_by_{suffix}.csv"
        )


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v24.load_data()
    dev_signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_trades, dev_skips = run_account(dev_signals, data, FAMILY)
    dev_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "early_exit_session": EARLY_EXIT_SESSION,
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v29_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v29_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.9 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v29_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "early_exit_session": EARLY_EXIT_SESSION,
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v29_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.9 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
