import unittest
from unittest.mock import patch

import pandas as pd

import develop_v13_expanded as engine


class ModeledOptionStopTest(unittest.TestCase):
    def test_option_stop_exits_before_wider_underlying_stop(self):
        timestamps = pd.date_range(
            "2026-01-05 09:55",
            periods=2,
            freq="5min",
            tz="America/New_York",
        )
        frame = pd.DataFrame(
            {
                "datetime": timestamps,
                "open": [100.0, 100.0],
                "high": [100.2, 100.2],
                "low": [99.8, 95.0],
                "close": [100.0, 96.0],
            }
        )
        signal = pd.Series(
            {
                "signal_idx": 0,
                "signal_time": timestamps[0],
                "atr": 10.0,
                "symbol": "TEST",
                "universe_group": "EXPANDED",
                "breakout_time": timestamps[0],
                "rvol": 4.0,
            }
        )

        original_stop = engine.OPTION_STOP_RETURN
        engine.OPTION_STOP_RETURN = -0.20
        try:
            with patch.object(engine, "estimate_volatility", return_value=0.40):
                trade = engine.simulate_trade(signal, frame)
        finally:
            engine.OPTION_STOP_RETURN = original_stop

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "OPTION_STOP")
        self.assertAlmostEqual(trade["option_return_pct"], -20.0, places=7)
        self.assertGreater(trade["exit_stock"], trade["stop_stock"])


if __name__ == "__main__":
    unittest.main()
