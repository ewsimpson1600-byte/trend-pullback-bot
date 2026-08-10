import unittest

import pandas as pd

import develop_v29 as v29


class EarlyFailureExitTest(unittest.TestCase):
    def frame(self, second_close):
        return pd.DataFrame(
            [
                {"date": pd.Timestamp("2020-01-01"), "open": 100, "high": 101, "low": 99, "close": 100, "sma5": 102},
                {"date": pd.Timestamp("2020-01-02"), "open": 100, "high": 101, "low": 99, "close": 100, "sma5": 102},
                {"date": pd.Timestamp("2020-01-03"), "open": 100, "high": 101, "low": 98, "close": second_close, "sma5": 102},
                {"date": pd.Timestamp("2020-01-06"), "open": 100, "high": 103, "low": 99, "close": 102, "sma5": 101},
            ]
        )

    def signal(self):
        return pd.Series({"signal_idx": 0, "atr": 2.0, "symbol": "SPY", "signal_time": pd.Timestamp("2020-01-01")})

    def test_failed_rebound_exits_on_second_close(self):
        trade = v29.simulate_trade(self.signal(), self.frame(99.5), v29.FAMILY)
        self.assertEqual(trade["exit_reason"], "EARLY_FAILURE")
        self.assertEqual(trade["hold_sessions"], 2)

    def test_profitable_rebound_keeps_normal_exit_path(self):
        trade = v29.simulate_trade(self.signal(), self.frame(100.5), v29.FAMILY)
        self.assertNotEqual(trade["exit_reason"], "EARLY_FAILURE")

    def test_v26_signal_and_risk_geometry_remain_frozen(self):
        self.assertEqual(v29.FAMILY, v29.v26.FAMILY)
        self.assertEqual(v29.EARLY_EXIT_SESSION, 2)


if __name__ == "__main__":
    unittest.main()
