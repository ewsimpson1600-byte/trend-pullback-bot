import unittest

import pandas as pd

import screen_v54 as v54


class V54RelativeStrengthTest(unittest.TestCase):
    def test_frozen_protocol_and_account_controls(self):
        self.assertEqual(v54.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v54.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v54.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v54.v21.MAX_RISK, 0.02)
        self.assertEqual(v54.v21.MAX_ALLOCATION, 0.80)

    def test_strategy_specification_is_fixed(self):
        self.assertEqual(v54.SYMBOLS, ("SPY", "QQQ", "IWM"))
        self.assertEqual(v54.MOMENTUM_SESSIONS, 63)
        self.assertEqual(v54.VOLATILITY_SESSIONS, 63)
        self.assertEqual(v54.EMA_RISE_SESSIONS, 20)
        self.assertEqual(v54.WEEK_STRIDE, 2)
        self.assertEqual(v54.FAMILY.stop_atr, 2.0)
        self.assertEqual(v54.FAMILY.target_atr, 100.0)
        self.assertEqual(v54.FAMILY.max_hold_sessions, 5)

    def test_alternating_week_end_calendar(self):
        dates = pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
        ])
        frame = pd.DataFrame({"date": dates})
        selected = v54.alternating_week_end_dates(
            frame, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-31")
        )
        self.assertEqual(selected, [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-19")])


if __name__ == "__main__":
    unittest.main()
