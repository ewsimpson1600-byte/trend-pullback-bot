import os
import sys
import json
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


# ============================================================
# V1.3 VARIANT A — FORWARD TEST SCANNER
#
# SIGNAL ONLY.
# THIS FILE CONTAINS NO ORDER-SUBMISSION FUNCTION.
# ============================================================

EXECUTE_ORDERS = False

NY = ZoneInfo("America/New_York")

SYMBOLS = [
    "SPY",
    "QQQ",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMD",
    "TSLA",
]


# ============================================================
# FROZEN V1.3 VARIANT A RULES
# ============================================================

SIGNAL_START = "09:45"
SIGNAL_END = "11:15"

RVOL_MIN = 2.0

EMA_FAST = 9
EMA_SLOW = 21

ATR_PERIOD = 14

OPENING_RANGE_END = "09:45"

RETEST_TOLERANCE = 0.0015
MAX_RETEST_BARS = 6

OPTION_TARGET_DTE = 5
OPTION_MIN_DTE = 3
OPTION_MAX_DTE = 10

TARGET_DELTA = 0.50

RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_HISTORY_DAYS = 10


# ============================================================
# API CONFIG
# ============================================================

TWELVE_DATA_API_KEY = os.getenv(
    "TWELVE_DATA_API_KEY"
)

TRADIER_SANDBOX_TOKEN = os.getenv(
    "TRADIER_SANDBOX_TOKEN"
)

DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL"
)


TWELVE_DATA_URL = (
    "https://api.twelvedata.com/time_series"
)

TRADIER_BASE_URL = (
    "https://sandbox.tradier.com/v1"
)


# ============================================================
# STATE
#
# Prevent duplicate Discord alerts if the same completed
# 5-minute candle is scanned more than once.
# ============================================================

STATE_DIR = Path(".bot_state")

STATE_FILE = (
    STATE_DIR
    / "v13_forward_state.json"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

def safety_checks():
    print("=" * 72)
    print("V1.3 VARIANT A — FORWARD SCANNER")
    print("=" * 72)

    if EXECUTE_ORDERS:
        raise RuntimeError(
            "SAFETY STOP: EXECUTE_ORDERS must remain False."
        )

    if (
        TRADIER_BASE_URL
        != "https://sandbox.tradier.com/v1"
    ):
        raise RuntimeError(
            "SAFETY STOP: only Tradier sandbox is allowed."
        )

    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing."
        )

    if not TRADIER_SANDBOX_TOKEN:
        raise RuntimeError(
            "TRADIER_SANDBOX_TOKEN is missing."
        )

    print("✓ Signal-only mode")
    print("✓ Order execution disabled")
    print("✓ Tradier sandbox only")
    print()


# ============================================================
# STATE
# ============================================================

def load_state():
    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not STATE_FILE.exists():
        return {
            "alerted_signals": []
        }

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return {
            "alerted_signals": []
        }


def save_state(state):
    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep state from growing forever.
    state["alerted_signals"] = (
        state.get(
            "alerted_signals",
            [],
        )[-200:]
    )

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
        )


# ============================================================
# TIME HELPERS
# ============================================================

def now_et():
    return datetime.now(
        tz=NY
    )


def current_market_window():
    now = now_et()

    if now.weekday() >= 5:
        return False

    current_time = (
        now.strftime("%H:%M")
    )

    return (
        SIGNAL_START
        <= current_time
        <= "11:20"
    )


# ============================================================
# TWELVE DATA
# ============================================================

def download_symbol(symbol):
    params = {
        "symbol": symbol,
        "interval": "5min",
        "outputsize": 2000,
        "timezone": "America/New_York",
        "order": "ASC",
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }

    response = requests.get(
        TWELVE_DATA_URL,
        params=params,
        timeout=45,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data error for {symbol}: "
            f"{payload.get('message')}"
        )

    values = payload.get(
        "values",
        [],
    )

    if not values:
        raise RuntimeError(
            f"No data returned for {symbol}"
        )

    df = pd.DataFrame(
        values
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    if df["datetime"].dt.tz is None:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_localize(
                "America/New_York",
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        )

    else:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert(
                "America/New_York"
            )
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# REMOVE INCOMPLETE 5-MINUTE BAR
# ============================================================

def completed_bars_only(df):
    now = pd.Timestamp(
        now_et()
    )

    current_floor = (
        now.floor("5min")
    )

    # A bar timestamped 10:00 represents the
    # 10:00-10:05 interval.
    #
    # Therefore only bars strictly earlier than
    # the current 5-minute boundary are complete.
    df = df[
        df["datetime"]
        < current_floor
    ].copy()

    return (
        df.reset_index(
            drop=True
        )
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    df["date"] = (
        df["datetime"]
        .dt.date
    )

    df["time"] = (
        df["datetime"]
        .dt.strftime("%H:%M")
    )


    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema9"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    df["ema21"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        df["close"]
        .shift(1)
    )

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    df["true_range"] = (
        pd.concat(
            [
                tr1,
                tr2,
                tr3,
            ],
            axis=1,
        )
        .max(
            axis=1
        )
    )

    df["atr"] = (
        df["true_range"]
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
        )
        .mean()
    )


    # --------------------------------------------------------
    # VWAP — reset every trading day
    # --------------------------------------------------------

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    df["tpv"] = (
        typical_price
        * df["volume"]
    )

    grouped = df.groupby(
        "date",
        sort=False,
    )

    df["cum_tpv"] = (
        grouped["tpv"]
        .cumsum()
    )

    df["cum_volume"] = (
        grouped["volume"]
        .cumsum()
    )

    df["vwap"] = (
        df["cum_tpv"]
        / df[
            "cum_volume"
        ].replace(
            0,
            np.nan,
        )
    )


    # --------------------------------------------------------
    # 15-MINUTE OPENING RANGE
    #
    # 9:30
    # 9:35
    # 9:40
    # --------------------------------------------------------

    opening_mask = (
        (df["time"] >= "09:30")
        & (
            df["time"]
            < OPENING_RANGE_END
        )
    )

    opening_range = (
        df.loc[
            opening_mask
        ]
        .groupby(
            "date"
        )
        .agg(
            opening_range_high=(
                "high",
                "max",
            ),
            opening_range_low=(
                "low",
                "min",
            ),
        )
    )

    df = df.merge(
        opening_range,
        left_on="date",
        right_index=True,
        how="left",
    )


    # --------------------------------------------------------
    # RVOL
    #
    # Current 5-minute volume vs same 5-minute slot over
    # previous 20 trading sessions.
    # --------------------------------------------------------

    rvol_series = []

    for (
        slot,
        slot_df,
    ) in df.groupby(
        "time",
        sort=False,
    ):

        rolling_average = (
            slot_df["volume"]
            .shift(1)
            .rolling(
                RVOL_LOOKBACK_DAYS,
                min_periods=(
                    RVOL_MIN_HISTORY_DAYS
                ),
            )
            .mean()
        )

        rvol_series.append(
            pd.Series(
                rolling_average.values,
                index=slot_df.index,
            )
        )

    df["slot_avg_volume"] = (
        pd.concat(
            rvol_series
        )
        .sort_index()
    )

    df["rvol"] = (
        df["volume"]
        / df["slot_avg_volume"]
    )

    return df


# ============================================================
# LOAD ALL MARKET DATA
# ============================================================

def load_market_data():
    data = {}

    for symbol in SYMBOLS:
        print(
            f"Downloading {symbol}..."
        )

        df = download_symbol(
            symbol
        )

        df = completed_bars_only(
            df
        )

        df = add_indicators(
            df
        )

        data[symbol] = df

    return data


# ============================================================
# MARKET CONFIRMATION
# ============================================================

def market_is_bullish_at(
    timestamp,
    data,
):
    for symbol in [
        "SPY",
        "QQQ",
    ]:
        df = data[symbol]

        match = df[
            df["datetime"]
            == timestamp
        ]

        if match.empty:
            return False

        bar = match.iloc[-1]

        if not (
            float(bar["close"])
            > float(bar["vwap"])
            and float(bar["ema9"])
            > float(bar["ema21"])
        ):
            return False

    return True


# ============================================================
# FRESH OPENING-RANGE BREAKOUT
# ============================================================

def is_valid_breakout(
    df,
    index,
    data,
):
    if index <= 0:
        return False

    bar = df.iloc[index]

    previous = df.iloc[
        index - 1
    ]

    if (
        bar["date"]
        != previous["date"]
    ):
        return False

    if not (
        SIGNAL_START
        <= bar["time"]
        <= SIGNAL_END
    ):
        return False

    if not np.isfinite(
        bar["rvol"]
    ):
        return False

    if (
        float(bar["rvol"])
        < RVOL_MIN
    ):
        return False

    if not (
        float(bar["ema9"])
        > float(bar["ema21"])
    ):
        return False

    if not (
        float(bar["close"])
        > float(bar["vwap"])
    ):
        return False

    or_high = bar[
        "opening_range_high"
    ]

    if not np.isfinite(
        or_high
    ):
        return False

    fresh_breakout = (
        float(bar["close"])
        > float(or_high)
        and float(
            previous["close"]
        )
        <= float(or_high)
    )

    if not fresh_breakout:
        return False

    if not market_is_bullish_at(
        bar["datetime"],
        data,
    ):
        return False

    return True


# ============================================================
# VARIANT A RETEST
# ============================================================

def is_variant_a_retest(
    retest_bar,
    breakout_bar,
):
    or_high = float(
        breakout_bar[
            "opening_range_high"
        ]
    )

    retest_floor = (
        or_high
        * (
            1
            - RETEST_TOLERANCE
        )
    )

    return (
        float(
            retest_bar["low"]
        )
        <= or_high
        and float(
            retest_bar["low"]
        )
        >= retest_floor
        and float(
            retest_bar["close"]
        )
        > or_high
    )


# ============================================================
# FIND CURRENT SIGNAL
#
# The latest COMPLETED bar must be the retest/reclaim.
# ============================================================

def find_signal(
    symbol,
    data,
):
    df = data[symbol]

    if df.empty:
        return None

    latest_idx = (
        len(df) - 1
    )

    latest = df.iloc[
        latest_idx
    ]

    today = now_et().date()

    if latest["date"] != today:
        return None

    if not (
        SIGNAL_START
        <= latest["time"]
        <= SIGNAL_END
    ):
        return None

    if not np.isfinite(
        latest["atr"]
    ):
        return None


    # --------------------------------------------------------
    # Search backward for a valid fresh breakout no more
    # than 6 completed 5-minute bars earlier.
    # --------------------------------------------------------

    earliest_idx = max(
        1,
        latest_idx
        - MAX_RETEST_BARS,
    )

    for breakout_idx in range(
        latest_idx - 1,
        earliest_idx - 1,
        -1,
    ):

        breakout = df.iloc[
            breakout_idx
        ]

        if (
            breakout["date"]
            != latest["date"]
        ):
            break

        if not is_valid_breakout(
            df,
            breakout_idx,
            data,
        ):
            continue

        if not is_variant_a_retest(
            latest,
            breakout,
        ):
            continue

        return {
            "symbol":
                symbol,

            "breakout_time":
                breakout[
                    "datetime"
                ],

            "retest_time":
                latest[
                    "datetime"
                ],

            "breakout_rvol":
                float(
                    breakout[
                        "rvol"
                    ]
                ),

            "opening_range_high":
                float(
                    breakout[
                        "opening_range_high"
                    ]
                ),

            "retest_close":
                float(
                    latest[
                        "close"
                    ]
                ),

            "ema9":
                float(
                    latest[
                        "ema9"
                    ]
                ),

            "ema21":
                float(
                    latest[
                        "ema21"
                    ]
                ),

            "vwap":
                float(
                    latest[
                        "vwap"
                    ]
                ),

            "atr":
                float(
                    latest[
                        "atr"
                    ]
                ),
        }

    return None


# ============================================================
# TRADIER SANDBOX HELPERS
# ============================================================

def tradier_headers():
    return {
        "Authorization":
            f"Bearer {TRADIER_SANDBOX_TOKEN}",

        "Accept":
            "application/json",
    }


def tradier_get(
    path,
    params=None,
):
    response = requests.get(
        f"{TRADIER_BASE_URL}{path}",
        headers=tradier_headers(),
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    return payload


# ============================================================
# FIND ~5 DTE EXPIRATION
# ============================================================

def find_option_expiration(
    symbol,
):
    payload = tradier_get(
        "/markets/options/expirations",
        params={
            "symbol": symbol,
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )

    expirations = (
        payload.get(
            "expirations",
            {},
        )
        .get(
            "date",
            [],
        )
    )

    if isinstance(
        expirations,
        str,
    ):
        expirations = [
            expirations
        ]

    today = date.today()

    candidates = []

    for expiration in expirations:
        exp_date = (
            date.fromisoformat(
                expiration
            )
        )

        dte = (
            exp_date - today
        ).days

        if (
            OPTION_MIN_DTE
            <= dte
            <= OPTION_MAX_DTE
        ):
            candidates.append(
                (
                    abs(
                        dte
                        - OPTION_TARGET_DTE
                    ),
                    dte,
                    expiration,
                )
            )

    if not candidates:
        return None

    candidates.sort()

    return candidates[0][2]


# ============================================================
# SELECT ~0.50 DELTA CALL
# ============================================================

def select_option_contract(
    symbol,
):
    expiration = (
        find_option_expiration(
            symbol
        )
    )

    if not expiration:
        return None

    payload = tradier_get(
        "/markets/options/chains",
        params={
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true",
        },
    )

    options = (
        payload.get(
            "options",
            {},
        )
        .get(
            "option",
            [],
        )
    )

    if isinstance(
        options,
        dict,
    ):
        options = [
            options
        ]

    candidates = []

    for option in options:
        if (
            option.get(
                "option_type"
            )
            != "call"
        ):
            continue

        greeks = (
            option.get(
                "greeks"
            )
            or {}
        )

        try:
            delta = float(
                greeks.get(
                    "delta"
                )
            )

            bid = float(
                option.get(
                    "bid",
                    0,
                )
            )

            ask = float(
                option.get(
                    "ask",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        # Basic quote sanity only.
        if (
            bid <= 0
            or ask <= 0
            or ask < bid
        ):
            continue

        spread = (
            ask - bid
        )

        midpoint = (
            bid + ask
        ) / 2

        spread_pct = (
            spread / midpoint
            if midpoint > 0
            else np.inf
        )

        candidates.append(
            {
                "symbol":
                    option.get(
                        "symbol"
                    ),

                "expiration":
                    option.get(
                        "expiration_date"
                    ),

                "strike":
                    float(
                        option.get(
                            "strike"
                        )
                    ),

                "delta":
                    delta,

                "bid":
                    bid,

                "ask":
                    ask,

                "mid":
                    midpoint,

                "spread":
                    spread,

                "spread_pct":
                    spread_pct,

                "volume":
                    option.get(
                        "volume"
                    ),

                "open_interest":
                    option.get(
                        "open_interest"
                    ),

                "delta_distance":
                    abs(
                        delta
                        - TARGET_DELTA
                    ),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[
                "delta_distance"
            ],
            item[
                "spread_pct"
            ],
        )
    )

    return candidates[0]


# ============================================================
# DISCORD
# ============================================================

def send_discord(
    message,
):
    if not DISCORD_WEBHOOK_URL:
        print(
            "Discord webhook not configured."
        )

        return

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json={
            "content": message
        },
        timeout=20,
    )

    response.raise_for_status()


# ============================================================
# FORMAT ALERT
# ============================================================

def build_alert(
    signal,
    option,
):
    symbol = signal[
        "symbol"
    ]

    breakout_time = (
        signal[
            "breakout_time"
        ]
        .strftime(
            "%H:%M ET"
        )
    )

    retest_time = (
        signal[
            "retest_time"
        ]
        .strftime(
            "%H:%M ET"
        )
    )

    stop_reference = (
        signal[
            "retest_close"
        ]
        - (
            2
            * signal["atr"]
        )
    )

    if option:
        option_text = (
            f"{option['symbol']}\n"
            f"Expiration: {option['expiration']}\n"
            f"Strike: ${option['strike']:.2f}\n"
            f"Delta: {option['delta']:.3f}\n"
            f"Bid/Ask: "
            f"${option['bid']:.2f} / "
            f"${option['ask']:.2f}\n"
            f"Spread: "
            f"{option['spread_pct'] * 100:.1f}%"
        )

    else:
        option_text = (
            "No usable sandbox option "
            "contract was returned."
        )

    return (
        "🟢 **V1.3 VARIANT A SIGNAL**\n\n"

        f"**Ticker:** {symbol}\n"
        f"**Breakout:** {breakout_time}\n"
        f"**Retest/Reclaim:** {retest_time}\n"
        f"**Breakout RVOL:** "
        f"{signal['breakout_rvol']:.2f}x\n\n"

        f"OR High: "
        f"${signal['opening_range_high']:.2f}\n"

        f"Retest Close: "
        f"${signal['retest_close']:.2f}\n"

        f"EMA9: "
        f"${signal['ema9']:.2f}\n"

        f"EMA21: "
        f"${signal['ema21']:.2f}\n"

        f"VWAP: "
        f"${signal['vwap']:.2f}\n"

        f"ATR: "
        f"${signal['atr']:.2f}\n\n"

        "**Sandbox contract candidate:**\n"
        f"{option_text}\n\n"

        "Strategy entry would occur on "
        "the next 5-minute bar.\n"

        f"Approx 2 ATR underlying stop "
        f"reference: ${stop_reference:.2f}\n"

        "Option target: +30%\n"
        "Maximum hold: 90 minutes\n\n"

        "⚠️ SIGNAL ONLY — NO ORDER SUBMITTED"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    safety_checks()

    now = now_et()

    print(
        f"Current ET time: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )

    print()


    # --------------------------------------------------------
    # Outside our actual strategy window, exit normally.
    # --------------------------------------------------------

    if not current_market_window():
        print(
            "Outside V1.3 scanning window."
        )

        print(
            "No scan required."
        )

        return


    # --------------------------------------------------------
    # Load market data.
    # --------------------------------------------------------

    data = load_market_data()

    state = load_state()

    alerted = set(
        state.get(
            "alerted_signals",
            [],
        )
    )

    signals_found = []


    # --------------------------------------------------------
    # Scan every permitted symbol.
    # --------------------------------------------------------

    for symbol in SYMBOLS:
        signal = find_signal(
            symbol,
            data,
        )

        if signal is None:
            print(
                f"{symbol}: no V1.3 signal"
            )

            continue

        signal_id = (
            f"{symbol}|"
            f"{signal['retest_time'].isoformat()}"
        )

        if signal_id in alerted:
            print(
                f"{symbol}: signal already alerted"
            )

            continue

        print()
        print(
            "=" * 72
        )

        print(
            f"V1.3 SIGNAL FOUND: {symbol}"
        )

        print(
            "=" * 72
        )


        # ----------------------------------------------------
        # Tradier sandbox option selection.
        # ----------------------------------------------------

        try:
            option = (
                select_option_contract(
                    symbol
                )
            )

        except Exception as exc:
            print(
                f"Tradier option lookup failed: {exc}"
            )

            option = None


        # ----------------------------------------------------
        # Print + Discord.
        # ----------------------------------------------------

        alert = build_alert(
            signal,
            option,
        )

        print()
        print(alert)
        print()

        try:
            send_discord(
                alert
            )

        except Exception as exc:
            print(
                f"Discord alert failed: {exc}"
            )


        # ----------------------------------------------------
        # Record that this signal was processed.
        # ----------------------------------------------------

        alerted.add(
            signal_id
        )

        signals_found.append(
            signal
        )


    # --------------------------------------------------------
    # Save state.
    # --------------------------------------------------------

    state[
        "alerted_signals"
    ] = list(
        alerted
    )

    save_state(
        state
    )


    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "SCAN COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"New signals found: "
        f"{len(signals_found)}"
    )

    print(
        "Order execution: DISABLED"
    )

    print(
        "Environment: Tradier sandbox"
    )


if __name__ == "__main__":
    main()
