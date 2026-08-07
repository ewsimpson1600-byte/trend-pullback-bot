import unittest

from discord_practice_alert import build_practice_message


class PracticeAlertTest(unittest.TestCase):
    def test_message_is_unambiguously_practice_only(self):
        message = build_practice_message()
        self.assertIn("PRACTICE ALERT", message)
        self.assertIn("NOT A REAL TRADE", message)
        self.assertIn("NO ORDER SUBMITTED", message)
        self.assertIn("QQQ", message)


if __name__ == "__main__":
    unittest.main()
