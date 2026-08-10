"""Shared two-stage research funnel for strategies created after V4.6.

The two-year window is a fast, explicitly non-validating screen. A strategy
that passes may receive longer historical robustness testing, but only new
2026-forward observations can provide fresh validation because all historical
holdouts used by earlier versions have been opened.
"""

import pandas as pd


SCREEN_START = pd.Timestamp("2024-01-02")
SCREEN_END = pd.Timestamp("2025-12-31")
MIN_SCREEN_TRADES = 20
MIN_SCREEN_PROFIT_FACTOR = 1.25
MAX_SCREEN_DRAWDOWN_PCT = -15.0
REQUIRED_POSITIVE_YEARS = 2

# Frozen after V5.6. These are the independent strategy families that saw the
# same 2024-2025 screen. Only V5.2 passed, then failed unchanged 2010-2023
# robustness as V5.3. Additional strategies may not be selected on this window.
HISTORICAL_SCREEN_LEDGER = (
    ("V4.8", "TURN_OF_MONTH", False),
    ("V4.9", "DOWNSIDE_GAP_RECLAIM", False),
    ("V5.0", "NR7_VOLATILITY_CONTRACTION", False),
    ("V5.1", "DIVERSIFIED_WEEKEND_EFFECT", False),
    ("V5.2", "THREE_LOWER_CLOSES_UPTREND", True),
    ("V5.4", "BIWEEKLY_RISK_ADJUSTED_RELATIVE_STRENGTH", False),
    ("V5.5", "EXPANDING_SECTOR_BREADTH_CONTINUATION", False),
    ("V5.6", "ATR_SCALED_CROSS_SECTIONAL_DISPERSION_CONVERGENCE", False),
)
HISTORICAL_SCREEN_BUDGET_EXHAUSTED = True
LAST_HISTORICAL_SCREEN_VERSION = "V5.6"


def two_year_screen_pass(summary):
    """Return True only when every frozen fast-screen gate passes."""
    return bool(
        summary.get("trades", 0) >= MIN_SCREEN_TRADES
        and summary.get("account_return_pct", 0) > 0
        and summary.get("profit_factor", 0) >= MIN_SCREEN_PROFIT_FACTOR
        and summary.get("account_max_drawdown_pct", -100) >= MAX_SCREEN_DRAWDOWN_PCT
        and summary.get("positive_years", 0) >= REQUIRED_POSITIVE_YEARS
    )


def stage_after_screen(summary):
    """Route a strategy without implying historical screening is validation."""
    return "LONGER_HISTORICAL_ROBUSTNESS" if two_year_screen_pass(summary) else "REJECTED_AFTER_TWO_YEAR_SCREEN"


def historical_screen_authorized(version):
    """Allow reproducible reruns, but never a new selection on the used window."""
    return any(item[0] == version for item in HISTORICAL_SCREEN_LEDGER)


def stage_for_new_strategy(version):
    """Route unseen strategies away from the exhausted historical screen."""
    if HISTORICAL_SCREEN_BUDGET_EXHAUSTED and not historical_screen_authorized(version):
        return "FORWARD_ONLY_PREDECLARATION"
    return "TWO_YEAR_SCREEN"
