import unittest

import pandas as pd

import screen_v49 as v49


class V49GapReclaimTest(unittest.TestCase):
    def test_frozen_protocol_and_account(self):
        self.assertEqual(v49.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v49.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v49.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v49.v21.MAX_RISK, 0.02)
        self.assertEqual(v49.v21.MAX_ALLOCATION, 0.80)

    def test_gap_reclaim_specification_is_fixed(self):
        self.assertEqual(v49.MIN_GAP_DOWN, -0.01)
        self.assertEqual(v49.FAMILY.stop_atr, 2.0)
        self.assertEqual(v49.FAMILY.target_atr, 2.0)
        self.assertEqual(v49.FAMILY.max_hold_sessions, 5)
        self.assertEqual(v49.SYMBOLS, ("SPY", "QQQ", "IWM"))


if __name__ == "__main__":
    unittest.main()
