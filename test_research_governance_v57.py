import unittest

import research_protocol_v47 as protocol


class HistoricalScreenMultiplicityTest(unittest.TestCase):
    def test_screen_ledger_is_complete_and_frozen(self):
        versions = [item[0] for item in protocol.HISTORICAL_SCREEN_LEDGER]
        self.assertEqual(
            versions,
            ["V4.8", "V4.9", "V5.0", "V5.1", "V5.2", "V5.4", "V5.5", "V5.6"],
        )
        self.assertEqual(sum(bool(item[2]) for item in protocol.HISTORICAL_SCREEN_LEDGER), 1)
        self.assertEqual(protocol.LAST_HISTORICAL_SCREEN_VERSION, "V5.6")
        self.assertTrue(protocol.HISTORICAL_SCREEN_BUDGET_EXHAUSTED)

    def test_existing_versions_can_be_reproduced(self):
        for version, _, _ in protocol.HISTORICAL_SCREEN_LEDGER:
            self.assertTrue(protocol.historical_screen_authorized(version))
            self.assertEqual(protocol.stage_for_new_strategy(version), "TWO_YEAR_SCREEN")

    def test_unseen_version_is_forward_only(self):
        self.assertFalse(protocol.historical_screen_authorized("V5.7"))
        self.assertEqual(protocol.stage_for_new_strategy("V5.7"), "FORWARD_ONLY_PREDECLARATION")


if __name__ == "__main__":
    unittest.main()
