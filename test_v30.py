import unittest

import pandas as pd

import develop_v30 as v30


class DualSleeveTest(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(v30.MEAN_REVERSION_WEIGHT + v30.TREND_WEIGHT, 1.0)
        self.assertAlmostEqual(v30.MEAN_REVERSION_WEIGHT, 2 / 3)

    def test_combined_balance_is_total_portfolio_pnl(self):
        mean = pd.DataFrame([{"entry_time": "2020-01-01", "exit_time": "2020-01-03", "symbol": "SPY", "trade_pnl": 10.0}])
        trend = pd.DataFrame([{"entry_time": "2020-01-02", "exit_time": "2020-01-04", "symbol": "GLD", "trade_pnl": -3.0}])
        combined = v30.combine_sleeves(mean, trend)
        self.assertEqual(combined.iloc[-1]["account_balance"], 1007.0)
        self.assertEqual(set(combined["sleeve"]), {"MEAN_REVERSION", "TREND"})

    def test_aggregate_planned_risk_stays_two_percent(self):
        total = v30.MEAN_REVERSION_WEIGHT * v30.v21.MAX_RISK + v30.TREND_WEIGHT * v30.v21.MAX_RISK
        self.assertAlmostEqual(total, v30.v21.MAX_RISK)


if __name__ == "__main__":
    unittest.main()
