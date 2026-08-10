"""V5.6 two-year screen for long-only ETF dispersion convergence.

On alternating week-end sessions, this strategy compares five-session returns
for SPY, QQQ, and IWM that remain above their 200-day EMA. When at least two are
eligible and the leader-laggard spread exceeds their average ATR percentage,
it buys the laggard at the next open. The premise is peer-relative convergence,
not an absolute dip threshold, relative-strength continuation, or breadth.

Trades use a 2 ATR protective stop, 1.5 ATR rebound target, and five-session
maximum hold. This is only the frozen 2024-2025 screen and cannot place orders.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import develop_v21 as v21
import research_protocol_v47 as protocol


VERSION = "V5.6"
VARIANT = "ATR_SCALED_CROSS_SECTIONAL_DISPERSION_CONVERGENCE"
SYMBOLS = v21.SYMBOLS
FAMILY = v21.Family("DISPERSION_CONVERGENCE", 2.0, 1.5, 5)
RETURN_SESSIONS = 5
WEEK_STRIDE = 2
MIN_ELIGIBLE_ETFS = 2
RESULTS_DIR = Path("screen_results_v56")


def add_fields(frame):
    x = frame.copy()
    x["return5"] = x["close"] / x["close"].shift(RETURN_SESSIONS) - 1
    x["atr_pct"] = x["atr14"] / x["close"]
    return x


def alternating_week_ends(frame, start, end):
    calendar = frame.loc[frame["date"].between(start, end), ["date"]].copy()
    if calendar.empty:
        return []
    calendar["week"] = calendar["date"].dt.to_period("W-FRI")
    week_ends = calendar.groupby("week", sort=True).tail(1)["date"].tolist()
    return week_ends[::WEEK_STRIDE]


def candidate_on_date(data, signal_time):
    eligible = []
    for symbol in SYMBOLS:
        x = data[symbol]
        matches = x.index[x["date"] == signal_time]
        if len(matches) != 1:
            continue
        idx = int(matches[0])
        row = x.loc[idx]
        if (
            pd.notna(row["return5"])
            and pd.notna(row["atr_pct"])
            and pd.notna(row["atr14"])
            and float(row["close"]) > float(row["ema200"])
        ):
            eligible.append((float(row["return5"]), symbol, idx, row))
    if len(eligible) < MIN_ELIGIBLE_ETFS:
        return None
    spread = max(item[0] for item in eligible) - min(item[0] for item in eligible)
    threshold = float(np.mean([float(item[3]["atr_pct"]) for item in eligible]))
    if not np.isfinite(threshold) or threshold <= 0 or spread <= threshold:
        return None
    laggard_return, symbol, idx, row = min(eligible, key=lambda item: (item[0], item[1]))
    return {
        "symbol": symbol,
        "signal_idx": idx,
        "atr": float(row["atr14"]),
        "laggard_return5": laggard_return,
        "dispersion_spread": spread,
        "dispersion_threshold": threshold,
        "eligible_etfs": len(eligible),
    }


def build_signals(data, start=protocol.SCREEN_START, end=protocol.SCREEN_END):
    enriched = {symbol: add_fields(frame) for symbol, frame in data.items()}
    rows = []
    for signal_time in alternating_week_ends(enriched["SPY"], start, end):
        candidate = candidate_on_date(enriched, signal_time)
        if candidate is None:
            continue
        rows.append({
            "family": FAMILY.name,
            "variant": VARIANT,
            "signal_time": signal_time,
            "strength": candidate["dispersion_spread"] / candidate["dispersion_threshold"],
            **candidate,
        })
    return pd.DataFrame(rows, columns=[
        "family", "variant", "symbol", "signal_idx", "signal_time", "atr",
        "strength", "laggard_return5", "dispersion_spread",
        "dispersion_threshold", "eligible_etfs",
    ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    signals = build_signals(data)
    trades, skips = v21.run_account(signals, data, FAMILY)
    summary = {
        "version": VERSION,
        "variant": VARIANT,
        "screen_only": True,
        "screen_start": protocol.SCREEN_START.date().isoformat(),
        "screen_end": protocol.SCREEN_END.date().isoformat(),
        "universe_size": len(SYMBOLS),
        "return_sessions": RETURN_SESSIONS,
        "week_stride": WEEK_STRIDE,
        "min_eligible_etfs": MIN_ELIGIBLE_ETFS,
        "dispersion_threshold": "MEAN_ATR_PERCENT",
        **asdict(FAMILY),
        **v21.summarize(trades, skips, "TWO_YEAR_SCREEN"),
    }
    summary["screen_pass"] = protocol.two_year_screen_pass(summary)
    summary["next_stage"] = protocol.stage_after_screen(summary)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "v56_summary.csv", index=False)
    signals.to_csv(RESULTS_DIR / "v56_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / "v56_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / "v56_skips.csv", index=False)
    print("V5.6 TWO-YEAR DISPERSION-CONVERGENCE SCREEN")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
