import unittest

import pandas as pd

import develop_v26 as v26


class SectorBreadthTest(unittest.TestCase):
    def make_data(self, above_count):
        date = pd.Timestamp("2020-01-02")
        data = {}
        for index, symbol in enumerate(v26.SECTOR_SYMBOLS):
            close = 101.0 if index < above_count else 99.0
            data[symbol] = pd.DataFrame({"date": [date], "close": [close], "ema200": [100.0]})
        return data

    def test_strict_majority_is_risk_on(self):
        breadth = v26.breadth_frame(self.make_data(5))
        self.assertTrue(bool(breadth.iloc[0]["risk_on"]))

    def test_four_of_nine_is_rejected(self):
        breadth = v26.breadth_frame(self.make_data(4))
        self.assertFalse(bool(breadth.iloc[0]["risk_on"]))

    def test_sector_set_is_fixed_without_profit_based_pruning(self):
        self.assertEqual(len(v26.SECTOR_SYMBOLS), 9)
        self.assertEqual(v26.MIN_SECTORS_ABOVE_EMA200, 5)

    def test_v26_preserves_v22_trade_geometry(self):
        self.assertEqual(v26.FAMILY, v26.v22.FAMILY)
        self.assertEqual(v26.FAMILY.stop_atr, 2.0)
        self.assertEqual(v26.FAMILY.target_atr, 1.5)
        self.assertEqual(v26.FAMILY.max_hold_sessions, 5)

    def test_result_directory_is_version_specific(self):
        self.assertEqual(v26.RESULTS_DIR.name, "backtest_results_v26")


if __name__ == "__main__":
    unittest.main()
