"""Send one clearly labeled practice trade alert to Discord."""

import os

import requests


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def build_practice_message():
    return (
        "🧪 **PRACTICE ALERT — NOT A REAL TRADE**\n\n"
        "🔴 **PAPER OPTION EXIT**\n\n"
        "**Ticker:** QQQ\n"
        "**Contract:** QQQ $605 Call — approximately 5 DTE\n"
        "**Exit reason:** +30% option target reached\n\n"
        "Entry premium: $4.85 ($485.00)\n"
        "Exit premium: $6.31 ($631.00)\n"
        "Option return: **+30.1%**\n"
        "One-contract P/L: **+$146.00**\n"
        "Holding time: 35 minutes\n"
        "Updated paper balance: **$1,146.00**\n\n"
        "Quotes: Tradier sandbox (delayed)\n\n"
        "⚠️ **PRACTICE ONLY — NO ORDER SUBMITTED OR OPTION SOLD**"
    )


def main():
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL is missing")

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json={
            "content": build_practice_message(),
            "allowed_mentions": {"parse": []},
        },
        timeout=20,
    )
    response.raise_for_status()
    print("Practice Discord alert delivered successfully.")


if __name__ == "__main__":
    main()
