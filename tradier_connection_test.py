import os
import sys
from datetime import date, timedelta

import requests


BASE_URL = "https://sandbox.tradier.com/v1"

TOKEN = os.getenv("TRADIER_SANDBOX_TOKEN")
ACCOUNT_ID = os.getenv("TRADIER_SANDBOX_ACCOUNT")


def headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
    }


def request_get(path, params=None):
    response = requests.get(
        f"{BASE_URL}{path}",
        headers=headers(),
        params=params,
        timeout=30,
    )

    if not response.ok:
        print(f"\nERROR {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def request_post(path, data):
    response = requests.post(
        f"{BASE_URL}{path}",
        headers={
            **headers(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=data,
        timeout=30,
    )

    if not response.ok:
        print(f"\nERROR {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def get_profile():
    print("\n" + "=" * 70)
    print("1. TESTING SANDBOX AUTHENTICATION")
    print("=" * 70)

    data = request_get("/user/profile")

    print("✓ Authentication successful")

    profile = data.get("profile", data)

    return profile


def verify_account(profile):
    print("\n" + "=" * 70)
    print("2. VERIFYING PAPER ACCOUNT")
    print("=" * 70)

    if not ACCOUNT_ID:
        print("ERROR: TRADIER_SANDBOX_ACCOUNT is missing.")
        sys.exit(1)

    print(f"Configured sandbox account: ...{ACCOUNT_ID[-4:]}")

    account_numbers = []

    account_data = profile.get("account")

    if isinstance(account_data, list):
        for account in account_data:
            number = account.get("account_number")

            if number:
                account_numbers.append(number)

    elif isinstance(account_data, dict):
        number = account_data.get("account_number")

        if number:
            account_numbers.append(number)

    if account_numbers:
        if ACCOUNT_ID in account_numbers:
            print("✓ Sandbox account matches Tradier profile")
        else:
            print("WARNING:")
            print(
                "Configured account was not found in the profile response."
            )
            print(
                "The balance test below will determine whether the account is valid."
            )

    else:
        print(
            "Profile did not expose an account list in the expected format."
        )
        print(
            "Continuing with the configured sandbox account."
        )


def get_balance():
    print("\n" + "=" * 70)
    print("3. CHECKING PAPER BALANCE")
    print("=" * 70)

    data = request_get(
        f"/accounts/{ACCOUNT_ID}/balances"
    )

    balances = data.get("balances", data)

    total_cash = balances.get("total_cash")
    total_equity = balances.get("total_equity")
    option_bp = balances.get("option_buying_power")
    stock_bp = balances.get("stock_buying_power")

    print("✓ Balance endpoint accessible")

    if total_cash is not None:
        print(f"Total cash: ${float(total_cash):,.2f}")

    if total_equity is not None:
        print(f"Total equity: ${float(total_equity):,.2f}")

    if option_bp is not None:
        print(f"Option buying power: ${float(option_bp):,.2f}")

    if stock_bp is not None:
        print(f"Stock buying power: ${float(stock_bp):,.2f}")


def find_expiration(symbol="SPY"):
    print("\n" + "=" * 70)
    print("4. FINDING OPTION EXPIRATION")
    print("=" * 70)

    data = request_get(
        "/markets/options/expirations",
        params={
            "symbol": symbol,
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )

    expirations = (
        data.get("expirations", {})
        .get("date", [])
    )

    if isinstance(expirations, str):
        expirations = [expirations]

    if not expirations:
        print("No option expirations returned.")
        sys.exit(1)

    today = date.today()

    candidates = []

    for exp in expirations:
        exp_date = date.fromisoformat(exp)

        days = (exp_date - today).days

        if 3 <= days <= 10:
            candidates.append(
                (abs(days - 5), exp_date, exp)
            )

    if candidates:
        candidates.sort()
        expiration = candidates[0][2]
    else:
        expiration = expirations[0]

    print(f"✓ Found expiration: {expiration}")

    return expiration


def get_option_chain(symbol, expiration):
    print("\n" + "=" * 70)
    print("5. PULLING OPTION CHAIN")
    print("=" * 70)

    data = request_get(
        "/markets/options/chains",
        params={
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true",
        },
    )

    options = (
        data.get("options", {})
        .get("option", [])
    )

    if isinstance(options, dict):
        options = [options]

    if not options:
        print("No option contracts returned.")
        sys.exit(1)

    print(f"✓ Received {len(options)} option contracts")

    return options


def choose_call(options):
    print("\n" + "=" * 70)
    print("6. SELECTING A TEST CALL")
    print("=" * 70)

    calls = [
        option
        for option in options
        if option.get("option_type") == "call"
    ]

    if not calls:
        print("No calls found.")
        sys.exit(1)

    delta_candidates = []

    for option in calls:
        greeks = option.get("greeks") or {}

        delta = greeks.get("delta")

        if delta is None:
            continue

        try:
            delta = float(delta)
        except (TypeError, ValueError):
            continue

        delta_candidates.append(
            (
                abs(delta - 0.50),
                option,
            )
        )

    if delta_candidates:
        delta_candidates.sort(
            key=lambda x: x[0]
        )

        chosen = delta_candidates[0][1]

    else:
        calls.sort(
            key=lambda x: float(
                x.get("strike", 0)
            )
        )

        chosen = calls[len(calls) // 2]

    print("Selected test contract:")
    print(f"Symbol: {chosen.get('symbol')}")
    print(f"Strike: {chosen.get('strike')}")
    print(f"Expiration: {chosen.get('expiration_date')}")

    greeks = chosen.get("greeks") or {}

    if greeks.get("delta") is not None:
        print(f"Delta: {greeks.get('delta')}")

    if chosen.get("bid") is not None:
        print(f"Bid: {chosen.get('bid')}")

    if chosen.get("ask") is not None:
        print(f"Ask: {chosen.get('ask')}")

    return chosen


def preview_option_order(symbol, option):
    print("\n" + "=" * 70)
    print("7. PREVIEWING PAPER OPTION ORDER")
    print("=" * 70)

    option_symbol = option.get("symbol")

    if not option_symbol:
        print("Option OCC symbol missing.")
        sys.exit(1)

    payload = {
        "class": "option",
        "symbol": symbol,
        "option_symbol": option_symbol,
        "side": "buy_to_open",
        "quantity": 1,
        "type": "market",
        "duration": "day",

        # IMPORTANT:
        # This means Tradier validates the order,
        # but DOES NOT submit it.
        "preview": "true",
    }

    result = request_post(
        f"/accounts/{ACCOUNT_ID}/orders",
        payload,
    )

    order = result.get("order", result)

    print("✓ Preview request completed")
    print()
    print("NO ORDER WAS PLACED.")
    print()

    print(f"Status: {order.get('status')}")
    print(f"Result: {order.get('result')}")

    if order.get("commission") is not None:
        print(f"Commission: ${float(order['commission']):,.2f}")

    if order.get("fees") is not None:
        print(f"Fees: ${float(order['fees']):,.2f}")

    if order.get("cost") is not None:
        print(f"Estimated cost: ${abs(float(order['cost'])):,.2f}")

    if order.get("order_cost") is not None:
        print(
            f"Estimated order cost: "
            f"${abs(float(order['order_cost'])):,.2f}"
        )

    return order


def main():
    print()
    print("=" * 70)
    print("TRADIER SANDBOX CONNECTION TEST")
    print("=" * 70)

    print()
    print("Environment:")
    print("SANDBOX / PAPER TRADING")
    print()
    print("THIS SCRIPT WILL NOT PLACE A TRADE.")

    if not TOKEN:
        print()
        print("ERROR: TRADIER_SANDBOX_TOKEN is missing.")
        sys.exit(1)

    if not ACCOUNT_ID:
        print()
        print("ERROR: TRADIER_SANDBOX_ACCOUNT is missing.")
        sys.exit(1)

    profile = get_profile()

    verify_account(profile)

    get_balance()

    symbol = "SPY"

    expiration = find_expiration(symbol)

    options = get_option_chain(
        symbol,
        expiration,
    )

    option = choose_call(options)

    preview_option_order(
        symbol,
        option,
    )

    print()
    print("=" * 70)
    print("CONNECTION TEST COMPLETE")
    print("=" * 70)
    print()
    print("✓ Sandbox authentication works")
    print("✓ Paper account accessible")
    print("✓ Balance accessible")
    print("✓ Options expirations accessible")
    print("✓ Options chain accessible")
    print("✓ Option order preview accessible")
    print()
    print("✓ NO PAPER ORDER WAS SUBMITTED")
    print("✓ NO REAL MONEY WAS USED")
    print()


if __name__ == "__main__":
    main()
