import unittest

import research_protocol_v47 as protocol


class TwoYearResearchFunnelTest(unittest.TestCase):
    def passing_summary(self):
        return {
            "trades": 20,
            "account_return_pct": 1.0,
            "profit_factor": 1.25,
            "account_max_drawdown_pct": -15.0,
            "positive_years": 2,
        }

    def test_window_is_exactly_2024_and_2025(self):
        self.assertEqual(str(protocol.SCREEN_START.date()), "2024-01-02")
        self.assertEqual(str(protocol.SCREEN_END.date()), "2025-12-31")

    def test_all_frozen_gates_pass_at_boundary(self):
        self.assertTrue(protocol.two_year_screen_pass(self.passing_summary()))

    def test_each_failed_gate_rejects(self):
        failures = {
            "trades": 19,
            "account_return_pct": 0.0,
            "profit_factor": 1.24,
            "account_max_drawdown_pct": -15.01,
            "positive_years": 1,
        }
        for field, value in failures.items():
            with self.subTest(field=field):
                summary = self.passing_summary()
                summary[field] = value
                self.assertFalse(protocol.two_year_screen_pass(summary))

    def test_pass_routes_to_robustness_not_validation(self):
        self.assertEqual(protocol.stage_after_screen(self.passing_summary()), "LONGER_HISTORICAL_ROBUSTNESS")


if __name__ == "__main__":
    unittest.main()
