"""V2.3 confirmed-reversal ETF mean-reversion research.

V2.3 preserves the frozen V2.2 regime, account, execution, and exit rules.
It makes one predeclared structural change: an oversold setup must be followed
immediately by a bullish session that closes above the setup day's high.
The trade enters at the next session's open. Development remains 2010-2017;
2018-2025 stays sealed unless every frozen development gate passes.
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

import develop_v21 as v21
import develop_v22 as v22


VERSION = "V2.3"
VARIANT = "CONFIRMED_REVERSAL_MEAN_REVERSION"
FAMILY = v22.FAMILY
RESULTS_DIR = Path("backtest_results_v23")


def is_bullish_confirmation(frame, setup_idx):
    """True only when the session after setup confirms a bullish reversal."""
    confirmation_idx = int(setup_idx) + 1
    if confirmation_idx >= len(frame):
        return False
    setup = frame.iloc[int(setup_idx)]
    confirmation = frame.iloc[confirmation_idx]
    return bool(
        confirmation["close"] > setup["high"]
        and confirmation["close"] > confirmation["open"]
    )


def build_signals(data, start, end):
    """Confirm V2.2 setups, then signal for entry on the following session."""
    setups = v22.build_signals(data, start, end)
    if setups.empty:
        return setups
    rows = []
    for _, setup in setups.iterrows():
        frame = data[setup["symbol"]]
        setup_idx = int(setup["signal_idx"])
        confirmation_idx = setup_idx + 1
        if not is_bullish_confirmation(frame, setup_idx):
            continue
        confirmation = frame.iloc[confirmation_idx]
        if confirmation["date"] > end:
            continue
        row = setup.to_dict()
        row.update(
            {
                "setup_time": setup["signal_time"],
                "signal_idx": confirmation_idx,
                "signal_time": confirmation["date"],
                "confirmation_return_pct": (
                    confirmation["close"] / frame.iloc[setup_idx]["close"] - 1
                ) * 100,
                "variant": VARIANT,
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(setups.columns) + ["setup_time", "confirmation_return_pct"])
    return (
        pd.DataFrame(rows)
        .sort_values(["signal_time", "strength", "symbol"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def write_period(prefix, signals, trades, skips, summary):
    signals.to_csv(RESULTS_DIR / f"{prefix}_signals.csv", index=False)
    trades.to_csv(RESULTS_DIR / f"{prefix}_trades.csv", index=False)
    skips.to_csv(RESULTS_DIR / f"{prefix}_skips.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    v22.save_breakdowns(prefix, trades)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = v21.load_data()
    dev_signals = build_signals(data, v21.DEVELOPMENT_START, v21.DEVELOPMENT_END)
    dev_trades, dev_skips = v21.run_account(dev_signals, data, FAMILY)
    dev_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "confirmation_rule": "next_close_above_setup_high_and_next_open",
        **asdict(FAMILY),
        **v22.fixed_period_summary(dev_trades, dev_skips, "DEVELOPMENT"),
    }
    write_period("v23_development", dev_signals, dev_trades, dev_skips, dev_summary)
    if not dev_summary["pass"]:
        pd.DataFrame([{"status": "REJECTED", "reason": "FROZEN_DEVELOPMENT_GATES_FAILED", **dev_summary}]).to_csv(
            RESULTS_DIR / "v23_candidate.csv", index=False
        )
        print(pd.DataFrame([dev_summary]).to_string(index=False))
        print("V2.3 failed frozen development gates; validation was not opened.")
        return

    pd.DataFrame([{"status": "LOCKED", **dev_summary}]).to_csv(RESULTS_DIR / "v23_candidate.csv", index=False)
    validation_signals = build_signals(data, v21.VALIDATION_START, v21.VALIDATION_END)
    validation_trades, validation_skips = v21.run_account(validation_signals, data, FAMILY)
    validation_summary = {
        "version": VERSION,
        "variant": VARIANT,
        "confirmation_rule": "next_close_above_setup_high_and_next_open",
        **asdict(FAMILY),
        **v22.fixed_period_summary(validation_trades, validation_skips, "VALIDATION"),
    }
    write_period("v23_validation", validation_signals, validation_trades, validation_skips, validation_summary)
    print("V2.3 LOCKED VALIDATION RESULT")
    print(pd.DataFrame([validation_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
