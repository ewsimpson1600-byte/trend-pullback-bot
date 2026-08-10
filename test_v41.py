import unittest

import develop_v41 as v41


class LowVolatilityCoreSectorTest(unittest.TestCase):
    def test_trend_family_and_universe_are_unchanged(self):
        self.assertIs(v41.FAMILY, v41.v34.FAMILY)
        self.assertEqual(v41.SYMBOLS, v41.v34.SYMBOLS)

    def test_five_fixed_blocks_are_reused(self):
        self.assertEqual(v41.FOLDS, v41.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v41.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v41.v21.MAX_RISK, 0.02)
        self.assertEqual(v41.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
