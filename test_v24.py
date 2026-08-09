import unittest

import pandas as pd

import develop_v24 as v24


class DiversifiedUniverseTest(unittest.TestCase):
    def test_universe_is_fixed_and_unique(self):
        self.assertEqual(len(v24.SYMBOLS), 12)
        self.assertEqual(len(set(v24.SYMBOLS)), len(v24.SYMBOLS))

    def test_all_nine_legacy_sector_spdrs_are_included(self):
        sectors = {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}
        self.assertTrue(sectors.issubset(v24.SYMBOLS))

    def test_v24_preserves_v22_signal_and_trade_rules(self):
        self.assertEqual(v24.FAMILY, v24.v22.FAMILY)
        self.assertEqual(v24.FAMILY.stop_atr, 2.0)
        self.assertEqual(v24.FAMILY.target_atr, 1.5)
        self.assertEqual(v24.FAMILY.max_hold_sessions, 5)
        self.assertEqual(v24.v22.REGIME_SLOPE_SESSIONS, 20)

    def test_build_signals_keeps_sector_candidates(self):
        original = v24.v22.build_signals
        try:
            v24.v22.build_signals = lambda data, start, end: pd.DataFrame(
                [{"symbol": "XLK", "signal_time": pd.Timestamp("2015-01-02")}]
            )
            signals = v24.build_signals({}, pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31"))
        finally:
            v24.v22.build_signals = original
        self.assertEqual(signals.iloc[0]["symbol"], "XLK")
        self.assertEqual(signals.iloc[0]["variant"], v24.VARIANT)

    def test_result_directory_is_version_specific(self):
        self.assertEqual(v24.RESULTS_DIR.name, "backtest_results_v24")


if __name__ == "__main__":
    unittest.main()
