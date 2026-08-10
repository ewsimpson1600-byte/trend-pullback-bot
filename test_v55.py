import unittest

import pandas as pd

import screen_v55 as v55


class V55SectorBreadthTest(unittest.TestCase):
    def test_frozen_protocol_and_account_controls(self):
        self.assertEqual(v55.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v55.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v55.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v55.v21.MAX_RISK, 0.02)
        self.assertEqual(v55.v21.MAX_ALLOCATION, 0.80)

    def test_breadth_specification_is_fixed(self):
        self.assertEqual(len(v55.SYMBOLS), 9)
        self.assertEqual(v55.TREND_EMA, 50)
        self.assertEqual(v55.MOMENTUM_SESSIONS, 20)
        self.assertEqual(v55.BREADTH_LOOKBACK_SESSIONS, 5)
        self.assertEqual(v55.MIN_SECTORS_ABOVE_TREND, 6)
        self.assertEqual(v55.WEEK_STRIDE, 2)
        self.assertEqual(v55.FAMILY.stop_atr, 2.0)
        self.assertEqual(v55.FAMILY.target_atr, 100.0)
        self.assertEqual(v55.FAMILY.max_hold_sessions, 5)

    def test_breadth_requires_strict_expansion(self):
        dates = pd.date_range("2024-01-02", periods=7, freq="B")
        data = {}
        for number, symbol in enumerate(v55.SYMBOLS):
            above = [number < 5] * 5 + [number < 6] * 2
            data[symbol] = pd.DataFrame({"date": dates, "above_trend": above})
        breadth = v55.breadth_frame(data)
        self.assertEqual(int(breadth.iloc[-1]["breadth_count"]), 6)
        self.assertEqual(int(breadth.iloc[-1]["prior_breadth_count"]), 5)
        self.assertTrue(bool(breadth.iloc[-1]["breadth_expanding"]))


if __name__ == "__main__":
    unittest.main()
