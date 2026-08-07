import os
import time
from datetime import date

import requests


# ============================================================
# SAFETY CONFIG
# ============================================================

BASE_URL = "https://sandbox.tradier.com/v1"

TOKEN = os.getenv("TRADIER_SANDBOX_TOKEN")
ACCOUNT_ID = os.getenv("TRADIER_SANDBOX_ACCOUNT")

UNDERLYING = "SPY"
QUANTITY = 1

POLL_SECONDS = 5
MAX_POLL_ATTEMPTS = 24


# ============================================================
# HARD SAFETY CHECKS
# ============================================================

def safety_checks():
    print("=" * 70)
    print("TRADIER SANDBOX ROUND-TRIP OPTION TEST")
    print("=" * 70)

    if BASE_URL != "https://sandbox.tradier.com/v1":
        raise RuntimeError(
            "SAFETY STOP: Tradier production URL is not allowed."
        )

    if not TOKEN:
        raise RuntimeError(
            "TRADIER_SANDBOX_TOKEN is missing."
        )

    if not ACCOUNT_ID:
        raise RuntimeError(
            "TRADIER_SANDBOX_ACCOUNT is missing."
        )

    if QUANTITY != 1:
        raise RuntimeError(
            "SAFETY STOP: quantity must remain exactly 1."
        )

    print("✓ Sandbox URL confirmed")
    print("✓ Sandbox token loaded")
    print(f"✓ Sandbox account: ...{ACCOUNT_ID[-4:]}")
    print("✓ Maximum quantity: 1 contract")
    print()


# ============================================================
# HTTP HELPERS
# ============================================================

def headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
    }


def get(path, params=None):
    response = requests.get(
        f"{BASE_URL}{path}",
        headers=headers(),
        params=params,
        timeout=30,
    )

    if not response.ok:
        print(f"\nGET ERROR {response.status_code}")
        print(response.text)
        raise RuntimeError(
            "Tradier GET request failed."
        )

    return response.json()


def post(path, data):
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
        print(f"\nPOST ERROR {response.status_code}")
        print(response.text)
        raise RuntimeError(
            "Tradier POST request failed."
        )

    return response.json()


# ============================================================
# SAFE NORMALIZERS
# ============================================================

def normalize_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    # Tradier may sometimes return strings such as "null"
    # or other scalar values when no records exist.
    return []


def extract_nested_list(payload, outer_key, inner_key):
    """
    Safely handles shapes like:

    {
        "positions": {
            "position": [...]
        }
    }

    or:
    {
        "positions": "null"
    }

    or:
    {
        "positions": null
    }
    """

    outer = payload.get(outer_key)

    if outer is None:
        return []

    if not isinstance(outer, dict):
        return []

    inner = outer.get(inner_key)

    return normalize_list(inner)


# ============================================================
# PAPER ACCOUNT CHECK
# ============================================================

def check_balance():
    print("=" * 70)
    print("1. VERIFY PAPER ACCOUNT")
    print("=" * 70)

    data = get(
        f"/accounts/{ACCOUNT_ID}/balances"
    )

    balances = data.get(
        "balances",
        data,
    )

    if not isinstance(balances, dict):
        raise RuntimeError(
            "Unexpected balances response from Tradier."
        )

    print("✓ Paper account accessible")

    for field in [
        "total_cash",
        "total_equity",
        "option_buying_power",
        "stock_buying_power",
    ]:
        value = balances.get(field)

        if value is not None:
            try:
                print(
                    f"{field}: ${float(value):,.2f}"
                )
            except Exception:
                print(f"{field}: {value}")

    print()


# ============================================================
# CHECK EXISTING ORDERS
# ============================================================

def get_orders():
    data = get(
        f"/accounts/{ACCOUNT_ID}/orders"
    )

    return extract_nested_list(
        data,
        "orders",
        "order",
    )


def check_existing_open_orders():
    print("=" * 70)
    print("2. CHECK EXISTING OPEN ORDERS")
    print("=" * 70)

    orders = get_orders()

    blocking_states = {
        "pending",
        "open",
        "submitted",
        "partially_filled",
        "accepted_for_bidding",
        "calculating",
    }

    blocking = []

    for order in orders:
        status = str(
            order.get("status", "")
        ).lower()

        symbol = str(
            order.get("symbol", "")
        ).upper()

        if (
            status in blocking_states
            and symbol == UNDERLYING
        ):
            blocking.append(order)

    if blocking:
        print(
            "SAFETY STOP: Existing SPY paper order(s) detected."
        )

        for order in blocking:
            print(
                f"Order ID: {order.get('id')} "
                f"| Status: {order.get('status')} "
                f"| Side: {order.get('side')}"
            )

        print()
        print(
            "No new paper buy will be submitted."
        )

        return False

    print("✓ No blocking SPY paper orders found")
    print()

    return True


# ============================================================
# CHECK EXISTING POSITIONS
# ============================================================

def get_positions():
    data = get(
        f"/accounts/{ACCOUNT_ID}/positions"
    )

    return extract_nested_list(
        data,
        "positions",
        "position",
    )


def check_existing_spy_option_position():
    print("=" * 70)
    print("3. CHECK EXISTING SPY POSITION")
    print("=" * 70)

    positions = get_positions()

    blocking = []

    for position in positions:
        symbol = str(
            position.get("symbol", "")
        ).upper()

        quantity = position.get(
            "quantity"
        )

        try:
            quantity = float(quantity)
        except Exception:
            quantity = 0

        if (
            quantity != 0
            and symbol.startswith("SPY")
        ):
            blocking.append(
                position
            )

    if blocking:
        print(
            "SAFETY STOP: Existing SPY paper position detected."
        )

        for position in blocking:
            print(
                f"Symbol: {position.get('symbol')} "
                f"| Quantity: {position.get('quantity')} "
                f"| Cost basis: {position.get('cost_basis')}"
            )

        print()
        print(
            "No new paper buy will be submitted."
        )

        return False

    print("✓ No existing SPY paper position found")
    print()

    return True


# ============================================================
# FIND EXPIRATION
# ============================================================

def find_expiration():
    print("=" * 70)
    print("4. FIND ~5 DTE SPY EXPIRATION")
    print("=" * 70)

    data = get(
        "/markets/options/expirations",
        params={
            "symbol": UNDERLYING,
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )

    expirations_container = data.get(
        "expirations",
        {}
    )

    if not isinstance(expirations_container, dict):
        raise RuntimeError(
            "Unexpected expiration response."
        )

    expirations = expirations_container.get(
        "date",
        []
    )

    if isinstance(expirations, str):
        expirations = [expirations]

    if not expirations:
        raise RuntimeError(
            "No SPY expirations returned."
        )

    today = date.today()

    candidates = []

    for expiration in expirations:
        exp_date = date.fromisoformat(
            expiration
        )

        dte = (
            exp_date - today
        ).days

        if 3 <= dte <= 10:
            candidates.append(
                (
                    abs(dte - 5),
                    dte,
                    expiration,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No expiration found between 3 and 10 DTE."
        )

    candidates.sort()

    _, dte, expiration = (
        candidates[0]
    )

    print(
        f"✓ Selected expiration: "
        f"{expiration} ({dte} calendar DTE)"
    )

    print()

    return expiration


# ============================================================
# OPTION CHAIN
# ============================================================

def get_chain(expiration):
    print("=" * 70)
    print("5. LOAD OPTION CHAIN")
    print("=" * 70)

    data = get(
        "/markets/options/chains",
        params={
            "symbol": UNDERLYING,
            "expiration": expiration,
            "greeks": "true",
        },
    )

    options_container = data.get(
        "options",
        {}
    )

    if not isinstance(options_container, dict):
        raise RuntimeError(
            "Unexpected options-chain response."
        )

    options = normalize_list(
        options_container.get(
            "option"
        )
    )

    if not options:
        raise RuntimeError(
            "Tradier returned no option contracts."
        )

    print(
        f"✓ Received {len(options)} contracts"
    )

    print()

    return options


# ============================================================
# CHOOSE CALL
# ============================================================

def choose_test_call(options):
    print("=" * 70)
    print("6. SELECT ~0.50 DELTA CALL")
    print("=" * 70)

    candidates = []

    for option in options:
        if (
            option.get("option_type")
            != "call"
        ):
            continue

        greeks = (
            option.get("greeks")
            or {}
        )

        if not isinstance(greeks, dict):
            continue

        delta = greeks.get(
            "delta"
        )

        if delta is None:
            continue

        try:
            delta = float(delta)
        except Exception:
            continue

        try:
            bid = float(
                option.get("bid", 0)
            )

            ask = float(
                option.get("ask", 0)
            )
        except Exception:
            continue

        if ask <= 0:
            continue

        candidates.append(
            (
                abs(
                    delta - 0.50
                ),
                option,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No suitable SPY call found."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    option = (
        candidates[0][1]
    )

    greeks = (
        option.get("greeks")
        or {}
    )

    print(
        f"OCC symbol: {option.get('symbol')}"
    )

    print(
        f"Strike: {option.get('strike')}"
    )

    print(
        f"Expiration: "
        f"{option.get('expiration_date')}"
    )

    print(
        f"Delta: {greeks.get('delta')}"
    )

    print(
        f"Bid: ${float(option.get('bid', 0)):.2f}"
    )

    print(
        f"Ask: ${float(option.get('ask', 0)):.2f}"
    )

    print()

    return option


# ============================================================
# PREVIEW BUY
# ============================================================

def preview_buy(option):
    print("=" * 70)
    print("7. PREVIEW BUY-TO-OPEN")
    print("=" * 70)

    payload = {
        "class": "option",
        "symbol": UNDERLYING,
        "option_symbol": option["symbol"],
        "side": "buy_to_open",
        "quantity": QUANTITY,
        "type": "market",
        "duration": "day",
        "preview": "true",
    }

    data = post(
        f"/accounts/{ACCOUNT_ID}/orders",
        payload,
    )

    order = data.get(
        "order",
        data,
    )

    if not isinstance(order, dict):
        raise RuntimeError(
            "Unexpected buy preview response."
        )

    print(
        f"Preview status: {order.get('status')}"
    )

    print(
        f"Preview result: {order.get('result')}"
    )

    if (
        order.get("result")
        is False
    ):
        raise RuntimeError(
            "Tradier rejected the paper buy preview."
        )

    print("✓ Preview passed")
    print()


# ============================================================
# PLACE PAPER BUY
# ============================================================

def place_buy(option):
    print("=" * 70)
    print("8. SUBMIT PAPER BUY-TO-OPEN")
    print("=" * 70)

    payload = {
        "class": "option",
        "symbol": UNDERLYING,
        "option_symbol": option["symbol"],
        "side": "buy_to_open",
        "quantity": QUANTITY,
        "type": "market",
        "duration": "day",
        "preview": "false",
        "tag": "sandbox-roundtrip-buy",
    }

    data = post(
        f"/accounts/{ACCOUNT_ID}/orders",
        payload,
    )

    order = data.get(
        "order",
        data,
    )

    if not isinstance(order, dict):
        raise RuntimeError(
            "Unexpected buy submission response."
        )

    order_id = order.get(
        "id"
    )

    if not order_id:
        raise RuntimeError(
            "Buy submission did not return an order ID."
        )

    print("✓ Paper buy submitted")
    print(f"Order ID: {order_id}")
    print()

    return order_id


# ============================================================
# ORDER STATUS
# ============================================================

def get_order(order_id):
    data = get(
        f"/accounts/{ACCOUNT_ID}/orders/{order_id}"
    )

    order = data.get(
        "order",
        data,
    )

    if not isinstance(order, dict):
        raise RuntimeError(
            "Unexpected order-status response."
        )

    return order


def wait_for_fill(
    order_id,
    label,
):
    print("=" * 70)
    print(f"9. WAIT FOR {label} FILL")
    print("=" * 70)

    terminal_failure_states = {
        "rejected",
        "expired",
        "canceled",
        "cancelled",
        "error",
    }

    for attempt in range(
        1,
        MAX_POLL_ATTEMPTS + 1,
    ):
        order = get_order(
            order_id
        )

        status = str(
            order.get("status", "")
        ).lower()

        avg_fill_price = (
            order.get(
                "avg_fill_price"
            )
        )

        exec_quantity = (
            order.get(
                "exec_quantity"
            )
        )

        print(
            f"[{attempt}/{MAX_POLL_ATTEMPTS}] "
            f"status={status} "
            f"exec_qty={exec_quantity} "
            f"avg_fill={avg_fill_price}"
        )

        if status == "filled":
            print()
            print(
                f"✓ {label} FILLED"
            )

            print()

            return (
                "filled",
                order,
            )

        if (
            status
            in terminal_failure_states
        ):
            print()
            print(
                f"{label} ended with status: "
                f"{status}"
            )

            print()

            return (
                "failed",
                order,
            )

        time.sleep(
            POLL_SECONDS
        )

    final_order = get_order(
        order_id
    )

    final_status = str(
        final_order.get(
            "status",
            ""
        )
    ).lower()

    print()
    print(
        f"{label} was not filled within "
        f"the polling window."
    )

    print(
        f"Current order status: "
        f"{final_status}"
    )

    print()

    return (
        "pending",
        final_order,
    )


# ============================================================
# PREVIEW CLOSE
# ============================================================

def preview_close(option):
    print("=" * 70)
    print("10. PREVIEW SELL-TO-CLOSE")
    print("=" * 70)

    payload = {
        "class": "option",
        "symbol": UNDERLYING,
        "option_symbol": option["symbol"],
        "side": "sell_to_close",
        "quantity": QUANTITY,
        "type": "market",
        "duration": "day",
        "preview": "true",
    }

    data = post(
        f"/accounts/{ACCOUNT_ID}/orders",
        payload,
    )

    order = data.get(
        "order",
        data,
    )

    if not isinstance(order, dict):
        raise RuntimeError(
            "Unexpected sell preview response."
        )

    if (
        order.get("result")
        is False
    ):
        raise RuntimeError(
            "Sell-to-close preview failed."
        )

    print("✓ Close preview passed")
    print()


# ============================================================
# PLACE CLOSE
# ============================================================

def place_close(option):
    print("=" * 70)
    print("11. SUBMIT PAPER SELL-TO-CLOSE")
    print("=" * 70)

    payload = {
        "class": "option",
        "symbol": UNDERLYING,
        "option_symbol": option["symbol"],
        "side": "sell_to_close",
        "quantity": QUANTITY,
        "type": "market",
        "duration": "day",
        "preview": "false",
        "tag": "sandbox-roundtrip-close",
    }

    data = post(
        f"/accounts/{ACCOUNT_ID}/orders",
        payload,
    )

    order = data.get(
        "order",
        data,
    )

    if not isinstance(order, dict):
        raise RuntimeError(
            "Unexpected sell submission response."
        )

    order_id = order.get(
        "id"
    )

    if not order_id:
        raise RuntimeError(
            "Close submission did not return an order ID."
        )

    print(
        "✓ Paper close submitted"
    )

    print(
        f"Order ID: {order_id}"
    )

    print()

    return order_id


# ============================================================
# MAIN
# ============================================================

def main():
    safety_checks()

    check_balance()

    if not check_existing_open_orders():
        print(
            "TEST ENDED SAFELY."
        )
        return

    if not check_existing_spy_option_position():
        print(
            "TEST ENDED SAFELY."
        )
        return

    expiration = find_expiration()

    options = get_chain(
        expiration
    )

    option = choose_test_call(
        options
    )

    preview_buy(
        option
    )

    buy_order_id = place_buy(
        option
    )

    buy_result, buy_order = wait_for_fill(
        buy_order_id,
        "BUY",
    )

    if buy_result == "pending":
        print("=" * 70)
        print("BUY STILL PENDING")
        print("=" * 70)

        print(
            "No sell-to-close order was submitted."
        )

        print(
            "This workflow is ending normally."
        )

        return

    if buy_result != "filled":
        print("=" * 70)
        print("BUY DID NOT FILL")
        print("=" * 70)

        print(
            "No sell-to-close order was submitted."
        )

        return

    preview_close(
        option
    )

    close_order_id = place_close(
        option
    )

    close_result, close_order = wait_for_fill(
        close_order_id,
        "SELL-TO-CLOSE",
    )

    print()
    print("=" * 70)
    print("ROUND-TRIP TEST RESULTS")
    print("=" * 70)

    print(
        "Environment: TRADIER SANDBOX"
    )

    print(
        f"Underlying: {UNDERLYING}"
    )

    print(
        f"Option: {option['symbol']}"
    )

    print(
        f"Quantity: {QUANTITY}"
    )

    print(
        f"Buy order ID: {buy_order_id}"
    )

    print(
        f"Buy status: "
        f"{buy_order.get('status')}"
    )

    print(
        f"Buy fill: "
        f"{buy_order.get('avg_fill_price')}"
    )

    print(
        f"Close order ID: {close_order_id}"
    )

    print(
        f"Close status: "
        f"{close_order.get('status')}"
    )

    print(
        f"Close fill: "
        f"{close_order.get('avg_fill_price')}"
    )

    if (
        buy_result == "filled"
        and close_result == "filled"
    ):

        try:
            buy_fill = float(
                buy_order[
                    "avg_fill_price"
                ]
            )

            close_fill = float(
                close_order[
                    "avg_fill_price"
                ]
            )

            pnl = (
                close_fill
                - buy_fill
            ) * 100

            return_pct = (
                close_fill
                / buy_fill
                - 1
            ) * 100

            print(
                f"Approx paper P/L: "
                f"${pnl:,.2f}"
            )

            print(
                f"Approx option return: "
                f"{return_pct:.2f}%"
            )

        except Exception:
            pass

        print()
        print(
            "✓ FULL PAPER ROUND TRIP COMPLETED"
        )

        print(
            "✓ NO REAL MONEY USED"
        )

    elif close_result == "pending":

        print()
        print(
            "Close order is still pending."
        )


if __name__ == "__main__":
    main()
