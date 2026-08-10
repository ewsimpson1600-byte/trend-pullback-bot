import unittest

import develop_v35 as v35


class TreasuryDiversifierTest(unittest.TestCase):
    def test_two_fixed_treasury_etfs(self):
        self.assertEqual(v35.TREASURY_SYMBOLS, ("IEF", "TLT"))

    def test_three_sleeves_share_one_account(self):
        self.assertEqual(v35.FAMILY.max_hold_sessions, 20)
        self.assertEqual(v35.v21.MAX_RISK, 0.02)
        self.assertEqual(v35.v21.MAX_ALLOCATION, 0.80)

    def test_holdout_is_pre_original_period(self):
        self.assertEqual(v35.VALIDATION_START.year, 2003)
        self.assertEqual(v35.VALIDATION_END.year, 2009)


if __name__ == "__main__":
    unittest.main()
