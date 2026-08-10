import unittest

import develop_v38 as v38


class RegimeSwitchingScreeningTest(unittest.TestCase):
    def test_components_are_inherited_unchanged(self):
        self.assertEqual(v38.MEAN_FAMILY, v38.v37.FAMILY)
        self.assertEqual(v38.DEFENSIVE_FAMILY, v38.v36.FAMILY)

    def test_asset_roles_are_fixed_and_disjoint(self):
        self.assertFalse(set(v38.RISK_ASSETS) & set(v38.DEFENSIVE_ASSETS))
        self.assertEqual(set(v38.RISK_ASSETS) | set(v38.DEFENSIVE_ASSETS),
                         set(v38.v36.SYMBOLS))

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v38.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v38.v21.MAX_RISK, 0.02)
        self.assertEqual(v38.v21.MAX_ALLOCATION, 0.80)

    def test_folds_are_inherited_without_tuning(self):
        self.assertEqual(v38.FOLDS, v38.v36.FOLDS)


if __name__ == "__main__":
    unittest.main()
