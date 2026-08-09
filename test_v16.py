import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import develop_v16 as v16


class OptionMathTest(unittest.TestCase):
    def test_call_and_put_prices_are_positive(self):
        self.assertGreater(v16.option_price(100, 100, 5 / 365, 0.30, "CALL"), 0)
        self.assertGreater(v16.option_price(100, 100, 5 / 365, 0.30, "PUT"), 0)

    def test_selected_put_delta_uses_absolute_target(self):
        _, delta = v16.select_strike(100, 0.30, "PUT")
        self.assertLess(delta, 0)
        self.assertAlmostEqual(abs(delta), v16.TARGET_ABS_DELTA, delta=0.02)


class DevelopmentGateTest(unittest.TestCase):
    def test_rejects_sample_below_fifty(self):
        rows = []
        for i in range(49):
            rows.append({
                "option_return_pct": 10.0, "trade_pnl": 1.0, "account_balance": 1001 + i,
                "account_drawdown_pct": 0.0, "entry_time": pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=i),
                "symbol": f"S{i % 5}", "direction": "CALL" if i % 2 else "PUT",
            })
        summary = v16.summarize(pd.DataFrame(rows), pd.DataFrame())
        self.assertFalse(summary["development_pass"])

    def test_bootstrap_is_reproducible(self):
        first = v16.bootstrap_mean_ci(np.array([-5, 10, 20, 30]))
        second = v16.bootstrap_mean_ci(np.array([-5, 10, 20, 30]))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
