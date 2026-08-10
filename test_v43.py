import unittest

import develop_v43 as v43


class CanonicalGlobalDualMomentumTest(unittest.TestCase):
    def test_asset_roles_and_lookback_are_fixed(self):
        self.assertEqual(v43.EQUITIES, ("SPY", "EFA"))
        self.assertEqual(v43.DEFENSIVE, "IEF")
        self.assertEqual(v43.MOMENTUM_SESSIONS, 252)

    def test_five_fixed_blocks_are_reused(self):
        self.assertEqual(v43.FOLDS, v43.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v43.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v43.v21.MAX_RISK, 0.02)
        self.assertEqual(v43.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
