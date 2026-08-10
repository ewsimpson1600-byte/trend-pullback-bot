import unittest

import develop_v44 as v44


class CrossAssetTrendPullbackTest(unittest.TestCase):
    def test_rule_is_exactly_v21_trend_pullback(self):
        self.assertEqual(v44.FAMILY.name, "TREND_PULLBACK")
        self.assertEqual((v44.FAMILY.stop_atr, v44.FAMILY.target_atr,
                          v44.FAMILY.max_hold_sessions), (2.0, 3.0, 10))

    def test_universe_and_blocks_are_unchanged(self):
        self.assertEqual(v44.SYMBOLS, v44.v36.SYMBOLS)
        self.assertEqual(v44.FOLDS, v44.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v44.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v44.v21.MAX_RISK, 0.02)
        self.assertEqual(v44.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
