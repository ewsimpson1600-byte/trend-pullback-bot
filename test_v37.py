import unittest

import develop_v37 as v37


class BlockedMeanReversionScreeningTest(unittest.TestCase):
    def test_rule_is_the_unchanged_v21_mean_reversion_family(self):
        self.assertEqual(v37.FAMILY.name, "MEAN_REVERSION")
        self.assertEqual((v37.FAMILY.stop_atr, v37.FAMILY.target_atr, v37.FAMILY.max_hold_sessions),
                         (2.0, 1.5, 5))

    def test_universe_and_folds_are_inherited_without_tuning(self):
        self.assertEqual(v37.SYMBOLS, v37.v36.SYMBOLS)
        self.assertEqual(v37.FOLDS, v37.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v37.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v37.v21.MAX_RISK, 0.02)
        self.assertEqual(v37.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
