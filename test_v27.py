import unittest

import develop_v27 as v27


class VolatilityBufferTest(unittest.TestCase):
    def test_only_trade_geometry_change_is_wider_stop(self):
        self.assertEqual(v27.FAMILY.name, v27.v22.FAMILY.name)
        self.assertEqual(v27.FAMILY.stop_atr, 3.0)
        self.assertEqual(v27.FAMILY.target_atr, v27.v22.FAMILY.target_atr)
        self.assertEqual(v27.FAMILY.max_hold_sessions, v27.v22.FAMILY.max_hold_sessions)

    def test_wider_stop_reduces_shares_and_preserves_risk_cap(self):
        old_shares = v27.v21.position_size(1000, 100, 4)
        new_shares = v27.v21.position_size(1000, 100, 6)
        self.assertLess(new_shares, old_shares)
        self.assertLessEqual(new_shares * (6 + 2 * v27.v21.PER_SHARE_COST), 1000 * v27.v21.MAX_RISK)

    def test_account_constraints_remain_frozen(self):
        self.assertEqual(v27.v21.MAX_RISK, 0.02)
        self.assertEqual(v27.v21.MAX_ALLOCATION, 0.80)
        self.assertEqual(v27.v21.STARTING_ACCOUNT, 1000.0)

    def test_result_directory_is_version_specific(self):
        self.assertEqual(v27.RESULTS_DIR.name, "backtest_results_v27")


if __name__ == "__main__":
    unittest.main()
