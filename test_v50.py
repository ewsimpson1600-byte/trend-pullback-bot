import unittest

import pandas as pd

import screen_v50 as v50


class V50NR7Test(unittest.TestCase):
    def test_frozen_protocol_and_account(self):
        self.assertEqual(v50.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v50.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v50.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v50.v21.MAX_RISK, 0.02)
        self.assertEqual(v50.v21.MAX_ALLOCATION, 0.80)

    def test_nr7_specification_is_fixed(self):
        self.assertEqual(v50.LOOKBACK, 7)
        self.assertEqual(v50.MIN_CLOSE_LOCATION, 0.75)
        self.assertEqual(v50.FAMILY.stop_atr, 1.5)
        self.assertEqual(v50.FAMILY.target_atr, 3.0)
        self.assertEqual(v50.FAMILY.max_hold_sessions, 8)
        self.assertEqual(v50.SYMBOLS, ("SPY", "QQQ", "IWM"))


if __name__ == "__main__":
    unittest.main()
