import unittest

import develop_v45 as v45


class CrossAssetVolumeBreakoutTest(unittest.TestCase):
    def test_rule_is_exactly_v21_volume_breakout(self):
        self.assertEqual(v45.FAMILY.name, "VOLUME_BREAKOUT")
        self.assertEqual((v45.FAMILY.stop_atr, v45.FAMILY.target_atr,
                          v45.FAMILY.max_hold_sessions), (2.0, 4.0, 15))

    def test_universe_and_blocks_are_unchanged(self):
        self.assertEqual(v45.SYMBOLS, v45.v36.SYMBOLS)
        self.assertEqual(v45.FOLDS, v45.v36.FOLDS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v45.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v45.v21.MAX_RISK, 0.02)
        self.assertEqual(v45.v21.MAX_ALLOCATION, 0.80)


if __name__ == "__main__":
    unittest.main()
