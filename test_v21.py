import unittest

import numpy as np
import pandas as pd

import develop_v21 as v21


class ShareSizingTest(unittest.TestCase):
    def test_whole_share_sizing_obeys_cash_and_risk(self):
        shares = v21.position_size(1000, 200, 5)
        self.assertEqual(shares, 3)
        self.assertLessEqual(shares * (200 + v21.PER_SHARE_COST), 1000 * v21.MAX_ALLOCATION)
        self.assertLessEqual(shares * (5 + 2 * v21.PER_SHARE_COST), 1000 * v21.MAX_RISK)

    def test_unaffordable_risk_returns_zero(self):
        self.assertEqual(v21.position_size(1000, 600, 30), 0)


class ExecutionTest(unittest.TestCase):
    def test_gap_below_stop_uses_open_not_stop(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": [100, 100, 90, 90],
                "high": [101, 101, 92, 91],
                "low": [99, 99, 89, 89],
                "close": [100, 100, 91, 90],
                "sma5": [100, 100, 100, 100],
            }
        )
        signal = pd.Series({"family": "TREND_PULLBACK", "symbol": "SPY", "signal_idx": 0, "signal_time": dates[0], "atr": 2, "strength": 1})
        trade = v21.simulate_trade(signal, frame, v21.FAMILIES[0])
        self.assertEqual(trade["exit_reason"], "STOP_GAP")
        self.assertLess(trade["exit_price"], trade["stop_price"])


class GateTest(unittest.TestCase):
    def test_validation_rejects_sample_below_fifty(self):
        rows = []
        for i in range(49):
            rows.append(
                {
                    "trade_return_pct": 1.0,
                    "trade_pnl": 1.0,
                    "account_balance": 1001 + i,
                    "account_drawdown_pct": 0.0,
                    "entry_time": pd.Timestamp("2018-01-02") + pd.Timedelta(days=i),
                    "symbol": v21.SYMBOLS[i % len(v21.SYMBOLS)],
                }
            )
        self.assertFalse(v21.summarize(pd.DataFrame(rows), pd.DataFrame(), "VALIDATION")["pass"])

    def test_bootstrap_is_reproducible(self):
        values = np.array([-1.0, 2.0, 3.0])
        self.assertEqual(v21.bootstrap_mean_ci(values), v21.bootstrap_mean_ci(values))


if __name__ == "__main__":
    unittest.main()
