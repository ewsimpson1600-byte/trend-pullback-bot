import unittest

import develop_v36 as v36


class BlockedScreeningTest(unittest.TestCase):
    def test_result_is_explicitly_screening_only(self):
        self.assertEqual(v36.VARIANT, "BLOCKED_CROSS_ASSET_DUAL_MOMENTUM")

    def test_five_nonoverlapping_folds_are_fixed(self):
        self.assertEqual(len(v36.FOLDS), 5)
        for left, right in zip(v36.FOLDS, v36.FOLDS[1:]):
            self.assertLess(left[1], right[0])

    def test_cross_asset_universe_is_fixed(self):
        self.assertEqual(len(v36.SYMBOLS), 8)
        self.assertIn("SPY", v36.SYMBOLS); self.assertIn("TLT", v36.SYMBOLS); self.assertIn("GLD", v36.SYMBOLS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v36.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v36.v21.MAX_RISK, 0.02)
        self.assertEqual(v36.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
