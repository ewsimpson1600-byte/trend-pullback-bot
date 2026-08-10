import unittest

import develop_v39 as v39


class VolatilityExpansionScreeningTest(unittest.TestCase):
    def test_precommitted_compression_and_breakout_rules(self):
        self.assertEqual(v39.BREAKOUT_SESSIONS, 20)
        self.assertEqual(v39.COMPRESSION_LOOKBACK, 252)
        self.assertEqual(v39.COMPRESSION_QUANTILE, 0.20)

    def test_fixed_exit_and_account_controls(self):
        self.assertEqual((v39.FAMILY.stop_atr, v39.FAMILY.target_atr,
                          v39.FAMILY.max_hold_sessions), (2.0, 4.0, 20))
        self.assertEqual(v39.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v39.v21.MAX_RISK, 0.02)
        self.assertEqual(v39.v21.MAX_ALLOCATION, 0.80)

    def test_universe_and_folds_are_not_retuned(self):
        self.assertEqual(v39.SYMBOLS, v39.v36.SYMBOLS)
        self.assertEqual(v39.FOLDS, v39.v36.FOLDS)


if __name__ == "__main__":
    unittest.main()
