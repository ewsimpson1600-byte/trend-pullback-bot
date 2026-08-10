import unittest

import develop_v42 as v42


class EqualOpportunityCrossAssetTest(unittest.TestCase):
    def test_universe_and_family_are_unchanged(self):
        self.assertEqual(v42.SYMBOLS, v42.v36.SYMBOLS)
        self.assertIs(v42.FAMILY, v42.v36.FAMILY)

    def test_five_fixed_blocks_are_reused(self):
        self.assertEqual(v42.FOLDS, v42.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v42.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v42.v21.MAX_RISK, 0.02)
        self.assertEqual(v42.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
