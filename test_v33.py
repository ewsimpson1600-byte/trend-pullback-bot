import unittest

import develop_v33 as v33


class DiversifiedSectorTrendTest(unittest.TestCase):
    def test_all_legacy_sectors_are_fixed(self):
        self.assertEqual(len(v33.SYMBOLS), 9)
        self.assertEqual(len(set(v33.SYMBOLS)), 9)

    def test_no_profit_rank_is_part_of_family(self):
        self.assertEqual(v33.VARIANT, "DIVERSIFIED_SECTOR_ABSOLUTE_TREND")

    def test_new_holdout_remains_2002_2009(self):
        self.assertEqual(v33.v32.VALIDATION_START.year, 2002)
        self.assertEqual(v33.v32.VALIDATION_END.year, 2009)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v33.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v33.v21.MAX_RISK, 0.02)
        self.assertEqual(v33.v21.MAX_ALLOCATION, 0.80)
        self.assertEqual(v33.FAMILY.max_hold_sessions, 20)


if __name__ == "__main__":
    unittest.main()
