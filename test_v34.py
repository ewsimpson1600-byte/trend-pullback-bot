import unittest

import develop_v34 as v34


class AlternatingTrendTest(unittest.TestCase):
    def test_universe_contains_spy_and_all_sectors(self):
        self.assertEqual(v34.SYMBOLS[0], "SPY")
        self.assertEqual(set(v34.SECTOR_SYMBOLS), set(v34.v33.SYMBOLS))
        self.assertEqual(len(v34.SYMBOLS), 10)

    def test_single_position_account_controls_are_frozen(self):
        self.assertEqual(v34.v21.MAX_RISK, 0.02)
        self.assertEqual(v34.v21.MAX_ALLOCATION, 0.80)
        self.assertEqual(v34.FAMILY.max_hold_sessions, 20)

    def test_original_holdout_is_not_reused(self):
        self.assertEqual(v34.v32.VALIDATION_START.year, 2002)
        self.assertEqual(v34.v32.VALIDATION_END.year, 2009)


if __name__ == "__main__":
    unittest.main()
