import unittest
from unittest.mock import Mock, patch

import tradier_v13_bot as scanner


class DiscordNotificationTest(unittest.TestCase):
    def test_send_discord_posts_without_mentions(self):
        response = Mock()

        with (
            patch.object(scanner, "DISCORD_WEBHOOK_URL", "https://example.test/hook"),
            patch.object(scanner.requests, "post", return_value=response) as post,
        ):
            delivered = scanner.send_discord("test alert")

        self.assertTrue(delivered)
        post.assert_called_once_with(
            "https://example.test/hook",
            json={
                "content": "test alert",
                "allowed_mentions": {"parse": []},
            },
            timeout=20,
        )
        response.raise_for_status.assert_called_once_with()

    def test_missing_webhook_is_an_error(self):
        with patch.object(scanner, "DISCORD_WEBHOOK_URL", None):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                scanner.send_discord("test alert")


if __name__ == "__main__":
    unittest.main()
