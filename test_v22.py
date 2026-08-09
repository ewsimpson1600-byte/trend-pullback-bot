import unittest

import pandas as pd

import develop_v22 as v22


class RegimeFilterTest(unittest.TestCase):
    def test_only_above_rising_ema200_is_risk_on(self):
        dates = pd.date_range("2020-01-01", periods=25, freq="B")
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": [101.0] * 25,
                "ema200": [100.0] * 20 + [100.1, 100.2, 100.3, 100.4, 100.5],
            }
        )
        allowed = v22.risk_on_dates({"SPY": frame})
        self.assertNotIn(dates[19].normalize(), allowed)
        self.assertIn(dates[24].normalize(), allowed)

    def test_falling_long_term_trend_is_rejected(self):
        dates = pd.date_range("2020-01-01", periods=25, freq="B")
        frame = pd.DataFrame(
            {"date": dates, "close": [110.0] * 25, "ema200": list(reversed(range(100, 125)))}
        )
        self.assertEqual(v22.risk_on_dates({"SPY": frame}), set())


class FrozenRulesTest(unittest.TestCase):
    def test_v22_keeps_v21_mean_reversion_trade_geometry(self):
        self.assertEqual(v22.FAMILY.name, "MEAN_REVERSION")
        self.assertEqual(v22.FAMILY.stop_atr, 2.0)
        self.assertEqual(v22.FAMILY.target_atr, 1.5)
        self.assertEqual(v22.FAMILY.max_hold_sessions, 5)

    def test_calendar_year_denominator_is_fixed(self):
        rows = []
        for year in range(2010, 2015):
            rows.append(
                {
                    "trade_return_pct": 1.0,
                    "trade_pnl": 1.0,
                    "account_balance": 1000 + len(rows) + 1,
                    "account_drawdown_pct": 0.0,
                    "entry_time": pd.Timestamp(year=year, month=6, day=1),
                    "symbol": "SPY",
                }
            )
        summary = v22.fixed_period_summary(pd.DataFrame(rows), pd.DataFrame(), "DEVELOPMENT")
        self.assertEqual(summary["years_tested"], 8)
        self.assertEqual(summary["positive_years"], 5)
        self.assertFalse(summary["pass"])


if __name__ == "__main__":
    unittest.main()
