import unittest

from discord_research_alert import build_message


class DiscordResearchAlertTest(unittest.TestCase):
    def test_validated_message_contains_results_and_safety_label(self):
        message = build_message(
            {
                "status": "validated",
                "version": "V16",
                "summary": "Passed the locked out-of-sample validation gates.",
                "metrics": {
                    "Trades": "63",
                    "Profit factor": "1.71",
                    "Max drawdown": "-18.4%",
                },
                "url": "https://github.com/example/run/1",
            }
        )

        self.assertIn("STRATEGY RESEARCH COMPLETE", message)
        self.assertIn("V16", message)
        self.assertIn("Profit factor: 1.71", message)
        self.assertIn("no live order was submitted", message)

    def test_blocker_message_requests_attention(self):
        message = build_message(
            {
                "status": "blocker",
                "version": "V16",
                "summary": "Historical quote access requires user action.",
            }
        )

        self.assertIn("NEEDS YOUR ATTENTION", message)


if __name__ == "__main__":
    unittest.main()
