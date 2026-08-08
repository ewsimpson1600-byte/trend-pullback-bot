import unittest
from unittest.mock import patch

import pandas as pd

import develop_v15 as v15


def signal(symbol, minute):
    return {
        "symbol": symbol,
        "signal_time": pd.Timestamp(
            f"2024-01-02 10:{minute:02d}", tz="America/New_York"
        ),
    }


def trade(symbol, entry_minute, exit_minute, cost, return_pct):
    entry = pd.Timestamp(
        f"2024-01-02 10:{entry_minute:02d}", tz="America/New_York"
    )
    return {
        "symbol": symbol,
        "entry_time": entry,
        "exit_time": pd.Timestamp(
            f"2024-01-02 10:{exit_minute:02d}", tz="America/New_York"
        ),
        "modeled_option_entry": cost / 100,
        "option_return_pct": return_pct,
        "underlying_return_pct": 1.0,
        "stop_stock": 99.0,
        "option_strike": 100.0,
        "estimated_volatility": 0.3,
        "exit_reason": "TIME",
    }


class RiskManagedPortfolioTest(unittest.TestCase):
    def setUp(self):
        self.policy = v15.Policy(0.25, 0.30, 0.25, 0.05)

    @patch.object(v15, "estimated_stop_risk_dollars", return_value=20.0)
    @patch.object(v15.engine, "simulate_trade")
    def test_skipped_unaffordable_trade_does_not_block_next_signal(
        self, simulate, _risk
    ):
        simulate.side_effect = [
            trade("EXPENSIVE", 0, 30, 300.0, 30.0),
            trade("AFFORDABLE", 5, 20, 200.0, 30.0),
        ]
        signals = pd.DataFrame([signal("EXPENSIVE", 0), signal("AFFORDABLE", 5)])
        trades, account, skipped = v15.run_policy(
            signals, {"EXPENSIVE": object(), "AFFORDABLE": object()}, self.policy
        )
        self.assertEqual(trades["symbol"].tolist(), ["AFFORDABLE"])
        self.assertEqual(skipped.iloc[0]["reason"], "ALLOCATION_LIMIT")
        self.assertAlmostEqual(account.iloc[-1]["account_balance"], 1060.0)

    @patch.object(v15, "estimated_stop_risk_dollars", return_value=60.0)
    @patch.object(v15.engine, "simulate_trade")
    def test_rejects_trade_above_estimated_risk_limit(self, simulate, _risk):
        simulate.return_value = trade("RISKY", 0, 20, 200.0, 30.0)
        trades, account, skipped = v15.run_policy(
            pd.DataFrame([signal("RISKY", 0)]), {"RISKY": object()}, self.policy
        )
        self.assertTrue(trades.empty)
        self.assertTrue(account.empty)
        self.assertEqual(skipped.iloc[0]["reason"], "STOP_RISK_LIMIT")


class CandidateSelectionTest(unittest.TestCase):
    def test_rejects_grid_without_eligible_policy(self):
        grid = pd.DataFrame(
            [
                {
                    "policy": v15.POLICIES[0].name,
                    "trades": 5,
                    "account_return_pct": 20.0,
                    "account_max_drawdown_pct": -10.0,
                    "profit_factor": 2.0,
                }
            ]
        )
        self.assertIsNone(v15.select_candidate(grid))

    def test_selects_highest_return_eligible_policy(self):
        first, second = v15.POLICIES[:2]
        grid = pd.DataFrame(
            [
                {
                    "policy": first.name,
                    "trades": 6,
                    "account_return_pct": 5.0,
                    "account_max_drawdown_pct": -10.0,
                    "profit_factor": 1.5,
                },
                {
                    "policy": second.name,
                    "trades": 7,
                    "account_return_pct": 8.0,
                    "account_max_drawdown_pct": -20.0,
                    "profit_factor": 1.4,
                },
            ]
        )
        self.assertEqual(v15.select_candidate(grid), second)


if __name__ == "__main__":
    unittest.main()
