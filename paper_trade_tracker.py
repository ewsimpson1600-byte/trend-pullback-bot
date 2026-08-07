"""Pure state transitions for one-position paper option tracking."""

from datetime import datetime, timedelta


STARTING_BALANCE = 1000.0
OPTION_TARGET_RETURN = 0.30
MAX_HOLD_MINUTES = 90


def default_paper_state():
    return {
        "paper_account_balance": STARTING_BALANCE,
        "pending_paper_trade": None,
        "open_paper_trade": None,
        "pending_paper_exit": None,
        "closed_paper_trades": [],
    }


def ensure_paper_state(state):
    for key, value in default_paper_state().items():
        state.setdefault(key, value)
    return state


def create_pending_trade(signal_id, signal, option):
    retest_time = signal["retest_time"]
    entry_due = retest_time + timedelta(minutes=5)
    stop_underlying = signal["retest_close"] - 2 * signal["atr"]
    return {
        "signal_id": signal_id,
        "underlying": signal["symbol"],
        "option_symbol": option["symbol"],
        "expiration": option["expiration"],
        "strike": float(option["strike"]),
        "signal_time": retest_time.isoformat(),
        "entry_due_time": entry_due.isoformat(),
        "stop_underlying": float(stop_underlying),
    }


def activate_pending_trade(pending, option_ask, entry_time, account_balance):
    contract_cost = float(option_ask) * 100.0
    if option_ask <= 0:
        raise ValueError("Option ask must be positive")
    if contract_cost > account_balance:
        return None, {
            "reason": "INSUFFICIENT_PAPER_CASH",
            "contract_cost": contract_cost,
            "account_balance": account_balance,
        }

    opened = dict(pending)
    opened.update(
        {
            "entry_time": entry_time.isoformat(),
            "entry_option_price": float(option_ask),
            "contract_cost": contract_cost,
            "target_option_price": float(option_ask)
            * (1.0 + OPTION_TARGET_RETURN),
            "max_exit_time": (
                entry_time + timedelta(minutes=MAX_HOLD_MINUTES)
            ).isoformat(),
            "quantity": 1,
        }
    )
    return opened, None


def evaluate_exit(open_trade, option_bid, underlying_low, checked_at):
    if option_bid <= 0:
        return None

    reason = None
    if float(underlying_low) <= float(open_trade["stop_underlying"]):
        reason = "UNDERLYING_STOP"
    elif float(option_bid) >= float(open_trade["target_option_price"]):
        reason = "OPTION_TARGET"
    elif checked_at >= datetime.fromisoformat(open_trade["max_exit_time"]):
        reason = "TIME_EXIT"

    if reason is None:
        return None

    entry_price = float(open_trade["entry_option_price"])
    exit_price = float(option_bid)
    pnl = (exit_price - entry_price) * 100.0
    return_pct = (exit_price / entry_price - 1.0) * 100.0
    entry_time = datetime.fromisoformat(open_trade["entry_time"])

    return {
        **open_trade,
        "exit_time": checked_at.isoformat(),
        "exit_option_price": exit_price,
        "exit_reason": reason,
        "option_return_pct": return_pct,
        "dollar_pnl": pnl,
        "hold_minutes": (checked_at - entry_time).total_seconds() / 60.0,
    }


def apply_exit_to_state(state, exit_trade):
    balance = float(state["paper_account_balance"]) + float(
        exit_trade["dollar_pnl"]
    )
    state["paper_account_balance"] = balance
    completed = dict(exit_trade)
    completed["account_balance_after"] = balance
    state["closed_paper_trades"].append(completed)
    state["closed_paper_trades"] = state["closed_paper_trades"][-100:]
    state["open_paper_trade"] = None
    state["pending_paper_exit"] = None
    return completed


def build_exit_alert(exit_trade, account_balance_after):
    pnl = float(exit_trade["dollar_pnl"])
    sign = "+" if pnl >= 0 else "-"
    reason_labels = {
        "OPTION_TARGET": "+30% option target",
        "UNDERLYING_STOP": "2-ATR underlying stop",
        "TIME_EXIT": "90-minute time exit",
    }
    return (
        "📤 **PAPER OPTION EXIT — NO REAL ORDER**\n\n"
        f"**Ticker:** {exit_trade['underlying']}\n"
        f"**Contract:** {exit_trade['option_symbol']}\n"
        f"**Exit reason:** {reason_labels[exit_trade['exit_reason']]}\n\n"
        f"Entry premium: ${float(exit_trade['entry_option_price']):.2f}\n"
        f"Exit premium: ${float(exit_trade['exit_option_price']):.2f}\n"
        f"Option return: {float(exit_trade['option_return_pct']):+.2f}%\n"
        f"One-contract P/L: {sign}${abs(pnl):.2f}\n"
        f"Holding time: {float(exit_trade['hold_minutes']):.0f} minutes\n"
        f"Paper account balance: ${float(account_balance_after):.2f}\n\n"
        "Quotes: Tradier sandbox (delayed)\n"
        "⚠️ PAPER TRACKING ONLY — NO OPTION WAS SOLD"
    )
