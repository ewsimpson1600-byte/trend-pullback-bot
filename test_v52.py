import unittest

import pandas as pd

import screen_v52 as v52


class V52ThreeLowerClosesTest(unittest.TestCase):
    def test_frozen_protocol_and_account(self):
        self.assertEqual(v52.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v52.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v52.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v52.v21.MAX_RISK, 0.02)
        self.assertEqual(v52.v21.MAX_ALLOCATION, 0.80)

    def test_price_sequence_specification_is_fixed(self):
        self.assertEqual(v52.CONSECUTIVE_LOWER_CLOSES, 3)
        self.assertEqual(v52.FAMILY.stop_atr, 2.0)
        self.assertEqual(v52.FAMILY.target_atr, 1.5)
        self.assertEqual(v52.FAMILY.max_hold_sessions, 5)
        self.assertEqual(v52.SYMBOLS, ("SPY", "QQQ", "IWM"))


if __name__ == "__main__":
    unittest.main()
