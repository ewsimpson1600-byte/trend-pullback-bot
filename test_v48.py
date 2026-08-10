import unittest

import pandas as pd

import screen_v48 as v48


class V48TurnOfMonthTest(unittest.TestCase):
    def test_frozen_screen_and_account_controls(self):
        self.assertEqual(v48.protocol.SCREEN_START, pd.Timestamp("2024-01-02"))
        self.assertEqual(v48.protocol.SCREEN_END, pd.Timestamp("2025-12-31"))
        self.assertEqual(v48.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v48.v21.MAX_RISK, 0.02)
        self.assertEqual(v48.v21.MAX_ALLOCATION, 0.80)

    def test_fixed_calendar_rule(self):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        frame = pd.DataFrame({"date": dates})
        selected = v48.month_signal_dates(frame, dates.min(), dates.max())
        self.assertEqual(selected, [pd.Timestamp("2024-01-30"), pd.Timestamp("2024-02-28")])
        self.assertEqual(v48.FAMILY.stop_atr, 3.0)
        self.assertEqual(v48.FAMILY.max_hold_sessions, 4)

    def test_diversified_universe_is_frozen(self):
        self.assertEqual(v48.SYMBOLS, ("SPY", "QQQ", "IWM"))
        self.assertEqual(v48.VARIANT, "DIVERSIFIED_TURN_OF_MONTH")


if __name__ == "__main__":
    unittest.main()
