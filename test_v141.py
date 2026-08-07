import unittest
from unittest.mock import patch

import pandas as pd

import develop_v13_expanded as engine


class TrailingStopTest(unittest.TestCase):
    def test_trailing_stop_activates_and_locks_a_gain(self):
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
                "high": [100.2, 103.0],
                "low": [99.8, 99.5],
                "close": [100.0, 101.0],
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

        original = (
            engine.OPTION_STOP_RETURN,
            engine.TRAILING_STOP_ACTIVATION_RETURN,
            engine.TRAILING_STOP_DISTANCE,
        )
        engine.OPTION_STOP_RETURN = None
        engine.TRAILING_STOP_ACTIVATION_RETURN = 0.15
        engine.TRAILING_STOP_DISTANCE = 0.10
        try:
            with patch.object(engine, "estimate_volatility", return_value=0.40):
                trade = engine.simulate_trade(signal, frame)
        finally:
            (
                engine.OPTION_STOP_RETURN,
                engine.TRAILING_STOP_ACTIVATION_RETURN,
                engine.TRAILING_STOP_DISTANCE,
            ) = original

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "TRAILING_STOP")
        self.assertGreater(trade["option_return_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
