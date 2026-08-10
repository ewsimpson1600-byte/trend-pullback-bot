import unittest
from pathlib import Path

import forward_validate_v46 as v46


class ForwardOnlyValidationTest(unittest.TestCase):
    def test_forward_boundary_and_maturity_are_frozen(self):
        self.assertEqual(str(v46.FORWARD_START.date()), "2026-01-02")
        self.assertEqual(v46.MIN_FORWARD_TRADES, 50)
        self.assertEqual(v46.MIN_FORWARD_YEARS, 3)

    def test_candidate_rule_is_unchanged_v34(self):
        self.assertIs(v46.FAMILY, v46.v34.FAMILY)
        self.assertEqual(v46.SYMBOLS, v46.v34.SYMBOLS)

    def test_account_controls_remain_frozen(self):
        self.assertEqual(v46.v21.STARTING_ACCOUNT, 1000.0)
        self.assertEqual(v46.v21.MAX_RISK, 0.02)
        self.assertEqual(v46.v21.MAX_ALLOCATION, 0.80)

    def test_forward_monitor_runs_weekly(self):
        workflow = Path(".github/workflows/forward-v46.yml").read_text()
        self.assertIn('cron: "30 22 * * 5"', workflow)
        self.assertIn("group: v46-forward-paper", workflow)


if __name__ == "__main__":
    unittest.main()
