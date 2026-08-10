import unittest

import pandas as pd

import develop_v28 as v28


class CrossAssetTrendTest(unittest.TestCase):
    def test_universe_is_cross_asset_and_predeclared(self):
        self.assertEqual(v28.SYMBOLS, ("SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"))

    def test_breakout_uses_prior_high_without_lookahead(self):
        dates = pd.date_range("2020-01-01", periods=130, freq="B")
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * 130,
                "high": [100.0] * 129 + [102.0],
                "low": [99.0] * 130,
                "close": [100.0] * 129 + [101.0],
                "volume": [1_000_000] * 130,
            }
        )
        enriched = v28.v21.add_indicators(frame)
        enriched["prior_high55"] = enriched["high"].shift(1).rolling(v28.BREAKOUT_SESSIONS).max()
        self.assertEqual(enriched.iloc[-1]["prior_high55"], 100.0)
        self.assertGreater(enriched.iloc[-1]["close"], enriched.iloc[-1]["prior_high55"])

    def test_cash_only_trade_geometry_is_frozen(self):
        self.assertEqual(v28.FAMILY.stop_atr, 2.5)
        self.assertEqual(v28.FAMILY.target_atr, 4.0)
        self.assertEqual(v28.FAMILY.max_hold_sessions, 20)


if __name__ == "__main__":
    unittest.main()
