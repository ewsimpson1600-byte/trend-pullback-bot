import unittest

import develop_v31 as v31


class MonthlySpyTrendTest(unittest.TestCase):
    def test_single_broad_market_universe(self):
        self.assertEqual(v31.VARIANT, "MONTHLY_SPY_ABSOLUTE_TREND")

    def test_monthly_geometry_is_frozen(self):
        self.assertEqual(v31.FAMILY.stop_atr, 3.0)
        self.assertEqual(v31.FAMILY.target_atr, 100.0)
        self.assertEqual(v31.FAMILY.max_hold_sessions, 20)
        self.assertEqual(v31.MOMENTUM_SESSIONS, 126)


if __name__ == "__main__":
    unittest.main()
