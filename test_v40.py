import unittest

import develop_v40 as v40


class UnchangedV34RobustnessTest(unittest.TestCase):
    def test_source_rule_is_exactly_v34(self):
        self.assertIs(v40.FAMILY, v40.v34.FAMILY)
        self.assertEqual(v40.SYMBOLS, v40.v34.SYMBOLS)
        self.assertEqual(v40.v34.VERSION, "V3.4")

    def test_five_fixed_blocks_are_reused(self):
        self.assertEqual(v40.FOLDS, v40.v36.FOLDS)
        self.assertEqual(len(v40.FOLDS), 5)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v40.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v40.v21.MAX_RISK, 0.02)
        self.assertEqual(v40.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
