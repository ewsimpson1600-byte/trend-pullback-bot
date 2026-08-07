"""Send one clearly labeled practice trade alert to Discord."""

import os

import requests


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def build_practice_message():
    return (
        "🧪 **PRACTICE ALERT — NOT A REAL TRADE**\n\n"
        "🟢 **V1.4 CANDIDATE SIGNAL**\n\n"
        "**Ticker:** QQQ\n"
        "**Breakout:** 09:45 ET\n"
        "**Retest/Reclaim:** 09:50 ET\n"
        "**Breakout RVOL:** 4.25x\n\n"
        "OR High: $604.95\n"
        "Retest Close: $605.10\n"
        "EMA9: $604.82\n"
        "EMA21: $604.41\n"
        "VWAP: $604.60\n"
        "ATR: $1.15\n\n"
        "**Example sandbox contract candidate:**\n"
        "QQQ call\n"
        "Expiration: approximately 5 DTE\n"
        "Strike: $605.00\n"
        "Delta: 0.50\n"
        "Bid/Ask: $4.70 / $4.85\n"
        "Spread: 3.1%\n\n"
        "Strategy entry would occur on the next 5-minute bar.\n"
        "Approx 2 ATR underlying stop reference: $602.80\n"
        "Option target: +30%\n"
        "Maximum hold: 90 minutes\n\n"
        "⚠️ **PRACTICE ONLY — NO ORDER SUBMITTED**"
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
