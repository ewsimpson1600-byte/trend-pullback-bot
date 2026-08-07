import unittest

import pandas as pd

import develop_v142 as v142


class OneContractAccountTest(unittest.TestCase):
    def test_skips_contract_that_exceeds_available_cash(self):
        trades = pd.DataFrame(
            [
                {
                    "entry_time": "2024-01-02 10:00:00-05:00",
                    "symbol": "TEST",
                    "modeled_option_entry": 12.00,
                    "option_return_pct": 30.0,
                }
            ]
        )
        account = v142.simulate_one_contract_account(trades)
        self.assertFalse(bool(account.iloc[0]["can_enter"]))
        self.assertEqual(account.iloc[0]["account_balance"], 1000.0)

    def test_applies_one_contract_profit_and_loss(self):
        trades = pd.DataFrame(
            [
                {
                    "entry_time": "2024-01-02 10:00:00-05:00",
                    "symbol": "A",
                    "modeled_option_entry": 5.00,
                    "option_return_pct": 20.0,
                },
                {
                    "entry_time": "2024-01-03 10:00:00-05:00",
                    "symbol": "B",
                    "modeled_option_entry": 4.00,
                    "option_return_pct": -25.0,
                },
            ]
        )
        account = v142.simulate_one_contract_account(trades)
        self.assertEqual(account.iloc[-1]["account_balance"], 1000.0)


if __name__ == "__main__":
    unittest.main()
