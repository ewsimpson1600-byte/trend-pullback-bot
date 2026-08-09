import unittest

import pandas as pd

import develop_v23 as v23


class ConfirmationTest(unittest.TestCase):
    def test_requires_close_above_setup_high_and_confirmation_open(self):
        frame = pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "close": 99.0},
                {"open": 99.5, "high": 102.0, "close": 101.5},
            ]
        )
        self.assertTrue(v23.is_bullish_confirmation(frame, 0))

    def test_rejects_bounce_that_does_not_reclaim_setup_high(self):
        frame = pd.DataFrame(
            [
                {"open": 100.0, "high": 103.0, "close": 99.0},
                {"open": 99.5, "high": 102.0, "close": 101.5},
            ]
        )
        self.assertFalse(v23.is_bullish_confirmation(frame, 0))

    def test_rejects_bearish_confirmation_candle(self):
        frame = pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "close": 99.0},
                {"open": 103.0, "high": 104.0, "close": 102.0},
            ]
        )
        self.assertFalse(v23.is_bullish_confirmation(frame, 0))


class FrozenRulesTest(unittest.TestCase):
    def test_v23_keeps_v22_trade_geometry(self):
        self.assertEqual(v23.FAMILY, v23.v22.FAMILY)
        self.assertEqual(v23.FAMILY.stop_atr, 2.0)
        self.assertEqual(v23.FAMILY.target_atr, 1.5)
        self.assertEqual(v23.FAMILY.max_hold_sessions, 5)

    def test_breakdowns_use_v23_result_directory(self):
        self.assertEqual(v23.RESULTS_DIR.name, "backtest_results_v23")


if __name__ == "__main__":
    unittest.main()
