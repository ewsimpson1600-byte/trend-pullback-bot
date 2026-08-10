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
