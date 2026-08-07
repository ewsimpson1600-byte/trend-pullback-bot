import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from paper_trade_tracker import (
    activate_pending_trade,
    apply_exit_to_state,
    build_exit_alert,
    default_paper_state,
    evaluate_exit,
)


NY = ZoneInfo("America/New_York")


class PaperTradeTrackerTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 7, 10, 0, tzinfo=NY)
        self.pending = {
            "signal_id": "QQQ|example",
            "underlying": "QQQ",
            "option_symbol": "QQQ260814C00605000",
            "expiration": "2026-08-14",
            "strike": 605.0,
            "signal_time": self.now.isoformat(),
            "entry_due_time": self.now.isoformat(),
            "stop_underlying": 602.8,
        }

    def test_target_exit_updates_dollars_and_balance(self):
        opened, skipped = activate_pending_trade(
            self.pending, 4.00, self.now, 1000.0
        )
        self.assertIsNone(skipped)
        exit_trade = evaluate_exit(
            opened,
            option_bid=5.20,
            underlying_low=604.0,
            checked_at=self.now + timedelta(minutes=30),
        )
        self.assertEqual(exit_trade["exit_reason"], "OPTION_TARGET")
        self.assertAlmostEqual(exit_trade["option_return_pct"], 30.0)
        self.assertAlmostEqual(exit_trade["dollar_pnl"], 120.0)

        state = default_paper_state()
        state["open_paper_trade"] = opened
        completed = apply_exit_to_state(state, exit_trade)
        self.assertAlmostEqual(completed["account_balance_after"], 1120.0)

        alert = build_exit_alert(exit_trade, 1120.0)
        self.assertIn("+30.00%", alert)
        self.assertIn("+$120.00", alert)
        self.assertIn("$1120.00", alert)
        self.assertIn("NO OPTION WAS SOLD", alert)

    def test_stop_has_priority_over_target(self):
        opened, _ = activate_pending_trade(
            self.pending, 4.00, self.now, 1000.0
        )
        exit_trade = evaluate_exit(
            opened,
            option_bid=5.20,
            underlying_low=602.0,
            checked_at=self.now + timedelta(minutes=5),
        )
        self.assertEqual(exit_trade["exit_reason"], "UNDERLYING_STOP")

    def test_time_exit(self):
        opened, _ = activate_pending_trade(
            self.pending, 4.00, self.now, 1000.0
        )
        exit_trade = evaluate_exit(
            opened,
            option_bid=3.60,
            underlying_low=604.0,
            checked_at=self.now + timedelta(minutes=90),
        )
        self.assertEqual(exit_trade["exit_reason"], "TIME_EXIT")
        self.assertAlmostEqual(exit_trade["dollar_pnl"], -40.0)

    def test_unaffordable_contract_is_skipped(self):
        opened, skipped = activate_pending_trade(
            self.pending, 12.00, self.now, 1000.0
        )
        self.assertIsNone(opened)
        self.assertEqual(skipped["reason"], "INSUFFICIENT_PAPER_CASH")


if __name__ == "__main__":
    unittest.main()
