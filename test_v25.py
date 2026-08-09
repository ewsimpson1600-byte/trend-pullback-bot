import unittest

import pandas as pd

import develop_v25 as v25


class MonthlyRotationTest(unittest.TestCase):
    def test_monthly_signal_dates_use_last_available_session(self):
        dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28"])
        spy = pd.DataFrame({"date": dates})
        result = v25.monthly_signal_dates(spy, dates.min(), dates.max())
        self.assertEqual(result, [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")])

    def test_family_preserves_cash_risk_sizing_geometry(self):
        self.assertEqual(v25.FAMILY.name, "MONTHLY_ROTATION")
        self.assertEqual(v25.FAMILY.stop_atr, 3.0)
        self.assertEqual(v25.FAMILY.max_hold_sessions, 20)
        self.assertEqual(v25.v21.MAX_RISK, 0.02)
        self.assertEqual(v25.v21.MAX_ALLOCATION, 0.80)

    def test_universe_is_inherited_without_result_based_pruning(self):
        self.assertEqual(v25.SYMBOLS, v25.v24.SYMBOLS)
        self.assertEqual(len(v25.SYMBOLS), 12)

    def test_result_directory_is_version_specific(self):
        self.assertEqual(v25.RESULTS_DIR.name, "backtest_results_v25")


if __name__ == "__main__":
    unittest.main()
