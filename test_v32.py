import unittest

import develop_v32 as v32


class HistoricalFrameworkTest(unittest.TestCase):
    def test_new_holdout_precedes_original_research_window(self):
        self.assertLess(v32.VALIDATION_END, v32.DEVELOPMENT_START)
        self.assertEqual((v32.VALIDATION_START.year, v32.VALIDATION_END.year), (2002, 2009))

    def test_three_broad_indices_are_fixed(self):
        self.assertEqual(v32.SYMBOLS, ("SPY", "QQQ", "IWM"))

    def test_risk_controls_remain_frozen(self):
        self.assertEqual(v32.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v32.v21.MAX_RISK, 0.02)
        self.assertEqual(v32.v21.MAX_ALLOCATION, 0.80)
        self.assertEqual(v32.FAMILY.stop_atr, 3.0)
        self.assertEqual(v32.FAMILY.max_hold_sessions, 20)

    def test_download_is_split_to_preserve_early_history(self):
        self.assertEqual(len(v32.DOWNLOAD_WINDOWS), 2)
        self.assertEqual(v32.DOWNLOAD_WINDOWS[0][0], "2000-01-01")


if __name__ == "__main__":
    unittest.main()
