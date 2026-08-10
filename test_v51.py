import unittest

import pandas as pd

import screen_v51 as v51


class V51WeekendTest(unittest.TestCase):
    def test_frozen_protocol_and_account(self):
        self.assertEqual(v51.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v51.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v51.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v51.v21.MAX_RISK, 0.02)
        self.assertEqual(v51.v21.MAX_ALLOCATION, 0.80)

    def test_weekend_specification_is_fixed(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-12")
        frame = pd.DataFrame({"date": dates})
        selected = v51.weekly_signal_dates(frame, dates.min(), dates.max())
        self.assertEqual(selected, [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-11")])
        self.assertEqual(v51.FAMILY.stop_atr, 1.5)
        self.assertEqual(v51.FAMILY.max_hold_sessions, 2)
        self.assertEqual(v51.SYMBOLS, ("SPY", "QQQ", "IWM"))


if __name__ == "__main__":
    unittest.main()
