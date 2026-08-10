import unittest

import pandas as pd

import robustness_v53 as v53


class V53FrozenRobustnessTest(unittest.TestCase):
    def test_candidate_rules_are_unchanged_from_v52(self):
        self.assertIs(v53.FAMILY, v53.v52.FAMILY)
        self.assertEqual(v53.VARIANT, v53.v52.VARIANT)
        self.assertEqual(v53.SYMBOLS, ("SPY", "QQQ", "IWM"))
        self.assertEqual(v53.v52.CONSECUTIVE_LOWER_CLOSES, 3)
        self.assertEqual(v53.FAMILY.stop_atr, 2.0)
        self.assertEqual(v53.FAMILY.target_atr, 1.5)
        self.assertEqual(v53.FAMILY.max_hold_sessions, 5)

    def test_historical_window_excludes_v52_screen(self):
        self.assertEqual(v53.START, pd.Timestamp("2010-01-04"))
        self.assertEqual(v53.END, pd.Timestamp("2023-12-29"))
        self.assertLess(v53.END, v53.v52.protocol.SCREEN_START)

    def test_gates_and_folds_are_frozen(self):
        self.assertEqual(v53.MIN_TRADES, 100)
        self.assertEqual(v53.MIN_PROFIT_FACTOR, 1.50)
        self.assertEqual(v53.MAX_DRAWDOWN_PCT, -25.0)
        self.assertEqual(v53.MAX_TICKER_CONTRIBUTION_PCT, 60.0)
        self.assertEqual(v53.MAX_MONTH_CONTRIBUTION_PCT, 35.0)
        self.assertEqual(v53.MIN_POSITIVE_YEARS, 10)
        self.assertEqual(v53.MIN_POSITIVE_FOLDS, 3)
        self.assertEqual(len(v53.FOLDS), 4)


if __name__ == "__main__":
    unittest.main()
