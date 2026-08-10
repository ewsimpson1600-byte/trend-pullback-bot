"""V3.8 blocked regime-switching cross-asset screening.

V3.6 momentum and V3.7 mean reversion failed in different regimes.  V3.8
therefore changes architecture rather than tuning either engine: unchanged
V2.1 oversold rebounds are allowed only while SPY is above a rising 200-day
EMA, while unchanged V3.6 defensive momentum signals are allowed only while
SPY is outside that regime.  The two engines share one cash account and may
hold only one position total.  Historical results are screening only.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v36 as v36
import develop_v37 as v37


VERSION = "V3.8"
VARIANT = "BLOCKED_REGIME_SWITCHING_MEAN_REVERSION_DEFENSE"
RISK_ASSETS = ("SPY", "EFA", "EEM", "VNQ", "DBC")
DEFENSIVE_ASSETS = ("IEF", "TLT", "GLD")
MEAN_FAMILY = v37.FAMILY
DEFENSIVE_FAMILY = v36.FAMILY
START = v36.START
END = v36.END
FOLDS = v36.FOLDS
RESULTS_DIR = Path("backtest_results_v38")


def risk_on_map(data):
    spy = data["SPY"].copy()
    spy["prior_ema"] = spy["ema200"].shift(v36.v32.EMA_SLOPE_SESSIONS)
    spy["risk_on"] = (spy["close"] > spy["ema200"]) & (spy["ema200"] > spy["prior_ema"])
    return spy.set_index("date")["risk_on"]


def build_signals(data, start, end):
    regime = risk_on_map(data)
    mean = v21.build_signals({symbol: data[symbol] for symbol in RISK_ASSETS},
                             MEAN_FAMILY, start, end)
    if not mean.empty:
        mean = mean.loc[mean["signal_time"].map(regime).fillna(False)].copy()
        mean["engine"] = "MEAN_REVERSION"

    defense = v36.build_signals(data, start, end)
    if not defense.empty:
        defense = defense.loc[
            defense["symbol"].isin(DEFENSIVE_ASSETS)
            & ~defense["signal_time"].map(regime).fillna(False)
        ].copy()
        defense["engine"] = "DEFENSIVE_MOMENTUM"

    frames = [frame for frame in (mean, defense) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["family", "variant", "symbol", "signal_idx",
                                     "signal_time", "atr", "strength", "engine"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["signal_time", "strength", "symbol"], ascending=[True, False, True]
    ).reset_index(drop=True)


def run_account(signals, data):
    balance = float(v21.STARTING_ACCOUNT)
    peak = balance
    next_available = None
    trades, skips = [], []
    for signal_time, candidates in signals.groupby("signal_time", sort=True):
        if next_available is not None and signal_time <= next_available:
            continue
        for _, signal in candidates.sort_values(["strength", "symbol"],
                                                 ascending=[False, True]).iterrows():
            family = MEAN_FAMILY if signal["engine"] == "MEAN_REVERSION" else DEFENSIVE_FAMILY
            trade = v21.simulate_trade(signal, data[signal["symbol"]], family)
            if trade is None:
                continue
            shares = v21.position_size(balance, trade["entry_price"], trade["risk_per_share"])
            if shares < 1:
                skips.append({"signal_time": signal_time, "symbol": signal["symbol"],
                              "engine": signal["engine"], "reason": "NO_AFFORDABLE_RISK_SIZED_SHARE",
                              "account_balance": balance, "entry_price": trade["entry_price"],
                              "risk_per_share": trade["risk_per_share"]})
                continue
            entry_cost = shares * (trade["entry_price"] + v21.PER_SHARE_COST)
            if entry_cost > balance:
                raise AssertionError("Cash-only sizing allowed an unaffordable trade")
            pnl = shares * (trade["exit_price"] - trade["entry_price"] - 2 * v21.PER_SHARE_COST)
            balance += pnl
            peak = max(peak, balance)
            trade.update({"shares": shares, "entry_cost": entry_cost, "trade_pnl": pnl,
                          "trade_return_pct": pnl / entry_cost * 100,
                          "account_balance": balance,
                          "account_drawdown_pct": (balance / peak - 1) * 100})
            trades.append(trade)
            next_available = trade["exit_time"]
            break
    return pd.DataFrame(trades), pd.DataFrame(skips)


def run_period(data, start, end):
    signals = build_signals(data, start, end)
    trades, skips = run_account(signals, data)
    return signals, trades, skips, v21.summarize(trades, skips, "VALIDATION")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v36.load_data()
    signals, trades, skips, summary = run_period(data, START, END)
    positive_years, years = v37.fixed_year_stats(trades, START, END)
    summary.update({"version": VERSION, "variant": VARIANT, "screening_only": True,
                    "positive_years": positive_years, "years_tested": years,
                    "mean_family": asdict(MEAN_FAMILY), "defensive_family": asdict(DEFENSIVE_FAMILY)})
    signals.to_csv(RESULTS_DIR / "v38_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v38_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v38_skips.csv", index=False)

    fold_rows = []
    for number, (raw_start, raw_end) in enumerate(FOLDS, 1):
        _, fold_trades, fold_skips, fold = run_period(
            data, pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        )
        fold_rows.append({"fold": number, "start": raw_start, "end": raw_end, **fold})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS_DIR / "v38_folds.csv", index=False)
    positive_folds = int((fold_frame["account_return_pct"] > 0).sum())
    summary["positive_folds"] = positive_folds
    summary["folds_tested"] = len(FOLDS)
    summary["screening_pass"] = bool(
        summary["trades"] >= 100
        and summary["bootstrap_mean_95_ci_low_pct"] > 0
        and summary["profit_factor"] >= 1.50
        and summary["account_max_drawdown_pct"] >= -25
        and summary["max_ticker_profit_contribution_pct"] <= 60
        and summary["max_month_profit_contribution_pct"] <= 35
        and positive_years >= 15
        and positive_folds >= 4
        and fold_frame["account_max_drawdown_pct"].min() >= -25
    )
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v38_screening_summary.csv", index=False)
    pd.DataFrame([{
        "status": "SCREENING_PASS" if summary["screening_pass"] else "REJECTED",
        "reason": "FORWARD_VALIDATION_REQUIRED" if summary["screening_pass"]
                  else "BLOCKED_SCREENING_GATES_FAILED",
    }]).to_csv(RESULTS_DIR / "v38_candidate.csv", index=False)
    print("V3.8 BLOCKED SCREENING RESULT")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
