"""Send a strategy-research completion or blocker alert to Discord."""

import json
import os
from pathlib import Path

import requests


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
NOTIFICATION_PATH = Path(
    os.getenv("RESEARCH_NOTIFICATION_PATH", ".github/research-notification.json")
)
VALID_STATUSES = {"validated", "blocker"}


def load_notification(path=NOTIFICATION_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        notification = json.load(handle)

    status = notification.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported research notification status: {status!r}")
    if not notification.get("version"):
        raise ValueError("Research notification requires a version")
    if not notification.get("summary"):
        raise ValueError("Research notification requires a summary")
    return notification


def build_message(notification):
    status = notification["status"]
    version = notification["version"]
    summary = notification["summary"]
    metrics = notification.get("metrics", {})
    url = notification.get("url")

    if status == "validated":
        lines = [
            "✅ **STRATEGY RESEARCH COMPLETE**",
            "",
            f"**Version:** {version}",
            summary,
        ]
    else:
        lines = [
            "⚠️ **STRATEGY RESEARCH NEEDS YOUR ATTENTION**",
            "",
            f"**Version:** {version}",
            summary,
        ]

    if metrics:
        lines.extend(["", "**Key results**"])
        lines.extend(f"- {name}: {value}" for name, value in metrics.items())

    if url:
        lines.extend(["", f"**Details:** {url}"])

    lines.extend(["", "Paper research only — no live order was submitted."])
    message = "\n".join(lines)
    if len(message) > 2000:
        raise ValueError("Discord research notification exceeds 2,000 characters")
    return message


def main():
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL is missing")

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json={
            "content": build_message(load_notification()),
            "allowed_mentions": {"parse": []},
        },
        timeout=20,
    )
    response.raise_for_status()
    print("Research Discord alert delivered successfully.")


if __name__ == "__main__":
    main()
