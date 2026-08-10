import unittest

import pandas as pd

import screen_v56 as v56


class V56DispersionConvergenceTest(unittest.TestCase):
    def test_frozen_protocol_and_account_controls(self):
        self.assertEqual(v56.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v56.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v56.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v56.v21.MAX_RISK, 0.02)
        self.assertEqual(v56.v21.MAX_ALLOCATION, 0.80)

    def test_strategy_specification_is_fixed(self):
        self.assertEqual(v56.SYMBOLS, ("SPY", "QQQ", "IWM"))
        self.assertEqual(v56.RETURN_SESSIONS, 5)
        self.assertEqual(v56.WEEK_STRIDE, 2)
        self.assertEqual(v56.MIN_ELIGIBLE_ETFS, 2)
        self.assertEqual(v56.FAMILY.stop_atr, 2.0)
        self.assertEqual(v56.FAMILY.target_atr, 1.5)
        self.assertEqual(v56.FAMILY.max_hold_sessions, 5)

    def test_atr_scaled_dispersion_selects_laggard(self):
        date = pd.Timestamp("2024-01-12")
        returns = {"SPY": 0.04, "QQQ": 0.01, "IWM": -0.02}
        data = {}
        for symbol, value in returns.items():
            data[symbol] = pd.DataFrame([{
                "date": date, "return5": value, "atr_pct": 0.01,
                "atr14": 2.0, "close": 100.0, "ema200": 90.0,
            }])
        candidate = v56.candidate_on_date(data, date)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["symbol"], "IWM")
        self.assertAlmostEqual(candidate["dispersion_spread"], 0.06)
        self.assertAlmostEqual(candidate["dispersion_threshold"], 0.01)

    def test_small_dispersion_is_rejected(self):
        date = pd.Timestamp("2024-01-12")
        data = {}
        for symbol, value in {"SPY": 0.01, "QQQ": 0.008, "IWM": 0.006}.items():
            data[symbol] = pd.DataFrame([{
                "date": date, "return5": value, "atr_pct": 0.01,
                "atr14": 2.0, "close": 100.0, "ema200": 90.0,
            }])
        self.assertIsNone(v56.candidate_on_date(data, date))


if __name__ == "__main__":
    unittest.main()
