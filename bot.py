import os
import json
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# ============================================================
# SETTINGS
# ============================================================

WATCHLIST = [
    "SPY",
    "QQQ",
    "NVDA",
    "MSFT",
    "AAPL",
    "AMD",
    "TSLA",
]

ACCOUNT_BALANCE = 20.00
MAX_CAPITAL_PER_TRADE = 20.00

# ------------------------------------------------------------
# PRIMARY MORNING SETUP
# First completed hourly candle = 9:30-10:30 ET
# ------------------------------------------------------------

PRIMARY_MIN_VOLUME_RATIO = 1.00

# ------------------------------------------------------------
# SECONDARY LATER-DAY SETUP
# More selective than the morning setup
# ------------------------------------------------------------

SECONDARY_MIN_VOLUME_RATIO = 1.25
SECONDARY_REQUIRE_VWAP = True

# Stop taking new secondary setups after this time
SECONDARY_END_HOUR = 14
SECONDARY_END_MINUTE = 45

# ------------------------------------------------------------
# EMA / ATR SETTINGS
# ------------------------------------------------------------

MAX_EMA_DISTANCE_ATR = 0.50

STOP_ATR_MULTIPLIER = 1.00
TARGET_ATR_MULTIPLIER = 1.50

STATE_FILE = ".bot_state/alert_state.json"


# ============================================================
# DISCORD
# ============================================================

def send_discord_message(message):

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("Discord webhook not configured.")
        return False

    try:
        response = requests.post(
            webhook_url,
            json={
                "content": message,
                "allowed_mentions": {
                    "parse": []
                }
            },
            timeout=10,
        )

        response.raise_for_status()

        print("Discord notification sent.")

        return True

    except Exception as error:

        print(
            "Discord notification failed:",
            error,
        )

        return False


# ============================================================
# DUPLICATE ALERT STATE
# ============================================================

def load_alert_state():

    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            "Could not load alert state:",
            error,
        )

        return {}


def save_alert_state(state):

    os.makedirs(
        os.path.dirname(STATE_FILE),
        exist_ok=True,
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


def already_alerted(
    ticker,
    signal_id,
):

    state = load_alert_state()

    return (
        state.get(ticker)
        == signal_id
    )


def mark_alerted(
    ticker,
    signal_id,
):

    state = load_alert_state()

    state[ticker] = signal_id

    save_alert_state(state)

    print(
        f"Saved alert state: "
        f"{ticker} / {signal_id}"
    )


# ============================================================
# TIME
# ============================================================

def get_new_york_time():

    return datetime.now(
        ZoneInfo(
            "America/New_York"
        )
    )


def market_is_open():

    now = get_new_york_time()

    if now.weekday() > 4:
        return False

    minutes = (
        now.hour * 60
        + now.minute
    )

    open_minutes = (
        9 * 60
        + 30
    )

    close_minutes = (
        16 * 60
    )

    return (
        open_minutes
        <= minutes
        < close_minutes
    )


def secondary_window_open():

    now = get_new_york_time()

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    start_minutes = (
        11 * 60
        + 30
    )

    end_minutes = (
        SECONDARY_END_HOUR * 60
        + SECONDARY_END_MINUTE
    )

    return (
        start_minutes
        <= current_minutes
        <= end_minutes
    )


# ============================================================
# DOWNLOAD 5-MINUTE DATA
# ============================================================

def download_5m_data(
    ticker,
    period="30d",
):

    data = yf.download(
        ticker,
        period=period,
        interval="5m",
        auto_adjust=False,
        prepost=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker}"
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        try:
            data = data.xs(
                ticker,
                axis=1,
                level=1,
            )

        except Exception:

            data.columns = (
                data.columns
                .get_level_values(0)
            )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    data = data[
        required
    ].copy()

    data = data.dropna()

    if data.index.tz is None:

        data.index = (
            data.index
            .tz_localize("UTC")
        )

    data.index = (
        data.index
        .tz_convert(
            "America/New_York"
        )
    )

    return data


# ============================================================
# BUILD 60-MINUTE CANDLES FROM 5-MINUTE DATA
# ============================================================

def build_hourly_bars(data):

    hourly = data.resample(
        "60min",
        origin="start_day",
        offset="30min",
        label="left",
        closed="left",
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    hourly = hourly.dropna()

    # Keep regular-market hourly bars only
    hourly = hourly[
        (
            (
                hourly.index.hour == 9
            )
            &
            (
                hourly.index.minute == 30
            )
        )
        |
        (
            (
                hourly.index.hour >= 10
            )
            &
            (
                hourly.index.hour < 16
            )
        )
    ]

    return hourly


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(hourly):

    hourly = hourly.copy()

    hourly["EMA20"] = (
        hourly["Close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    hourly["EMA200"] = (
        hourly["Close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    hourly["AvgVolume20"] = (
        hourly["Volume"]
        .rolling(20)
        .mean()
    )

    previous_close = (
        hourly["Close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            hourly["High"]
            - hourly["Low"],

            (
                hourly["High"]
                - previous_close
            ).abs(),

            (
                hourly["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    hourly["ATR14"] = (
        true_range
        .rolling(14)
        .mean()
    )

    return hourly


# ============================================================
# VWAP
# ============================================================

def calculate_current_vwap(data):

    today = (
        get_new_york_time()
        .date()
    )

    today_data = data[
        data.index.date
        == today
    ].copy()

    if today_data.empty:
        return None

    typical_price = (
        today_data["High"]
        + today_data["Low"]
        + today_data["Close"]
    ) / 3

    cumulative_value = (
        typical_price
        * today_data["Volume"]
    ).cumsum()

    cumulative_volume = (
        today_data["Volume"]
        .cumsum()
    )

    if cumulative_volume.iloc[-1] <= 0:
        return None

    vwap = (
        cumulative_value
        / cumulative_volume
    )

    return float(
        vwap.iloc[-1]
    )


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime():

    print("\n======================")
    print("MARKET REGIME")
    print("======================")

    spy_5m = download_5m_data(
        "SPY"
    )

    spy_hourly = build_hourly_bars(
        spy_5m
    )

    spy_hourly = calculate_indicators(
        spy_hourly
    )

    now = get_new_york_time()

    completed = spy_hourly[
        spy_hourly.index
        + pd.Timedelta(hours=1)
        <= now
    ]

    if len(completed) < 200:

        print(
            "Not enough completed "
            "SPY hourly history."
        )

        return False

    current = completed.iloc[-1]

    price = float(
        current["Close"]
    )

    ema200 = float(
        current["EMA200"]
    )

    returns = (
        completed["Close"]
        .pct_change()
        .dropna()
    )

    volatility = float(
        returns
        .tail(20)
        .std()
        * 100
    )

    bullish = (
        price > ema200
    )

    volatility_ok = (
        volatility < 1.5
    )

    approved = (
        bullish
        and volatility_ok
    )

    print(
        "SPY:",
        round(
            price,
            2,
        ),
    )

    print(
        "EMA200:",
        round(
            ema200,
            2,
        ),
    )

    print(
        "Trend:",
        "BULLISH"
        if bullish
        else "BEARISH",
    )

    print(
        "20-hour volatility:",
        round(
            volatility,
            3,
        ),
        "%",
    )

    print(
        "Market:",
        "APPROVED"
        if approved
        else "REJECTED",
    )

    return approved


# ============================================================
# DETERMINE SIGNAL MODE
# ============================================================

def determine_mode(signal_time):

    # --------------------------------------------------------
    # Primary candle:
    # 9:30-10:30 ET
    #
    # Hourly candle is labeled 9:30.
    # --------------------------------------------------------

    if (
        signal_time.hour == 9
        and
        signal_time.minute == 30
    ):

        return "PRIMARY"

    # --------------------------------------------------------
    # Later completed candles
    # --------------------------------------------------------

    if secondary_window_open():

        return "SECONDARY"

    return "NONE"


# ============================================================
# SCAN ONE TICKER
# ============================================================

def scan_ticker(ticker):

    print("\n----------------------")

    print(
        "Scanning:",
        ticker,
    )

    data_5m = download_5m_data(
        ticker
    )

    hourly = build_hourly_bars(
        data_5m
    )

    hourly = calculate_indicators(
        hourly
    )

    now = get_new_york_time()

    completed = hourly[
        hourly.index
        + pd.Timedelta(hours=1)
        <= now
    ]

    if len(completed) < 200:

        raise ValueError(
            f"Not enough completed "
            f"hourly data for {ticker}"
        )

    signal = completed.iloc[-1]

    signal_time = (
        completed.index[-1]
    )

    mode = determine_mode(
        signal_time
    )

    if mode == "NONE":

        print(
            "No eligible signal window."
        )

        return None


    # ========================================================
    # VALUES
    # ========================================================

    close = float(
        signal["Close"]
    )

    low = float(
        signal["Low"]
    )

    ema20 = float(
        signal["EMA20"]
    )

    ema200 = float(
        signal["EMA200"]
    )

    atr = float(
        signal["ATR14"]
    )

    volume = float(
        signal["Volume"]
    )

    avg_volume = float(
        signal["AvgVolume20"]
    )


    if (
        pd.isna(atr)
        or
        pd.isna(avg_volume)
        or
        atr <= 0
        or
        avg_volume <= 0
    ):

        raise ValueError(
            f"Indicators unavailable "
            f"for {ticker}"
        )


    # ========================================================
    # TREND
    # ========================================================

    trend_pass = (
        close > ema200
    )


    # ========================================================
    # VOLUME
    # ========================================================

    volume_ratio = (
        volume
        / avg_volume
    )

    if mode == "PRIMARY":

        required_volume_ratio = (
            PRIMARY_MIN_VOLUME_RATIO
        )

    else:

        required_volume_ratio = (
            SECONDARY_MIN_VOLUME_RATIO
        )

    volume_pass = (
        volume_ratio
        >= required_volume_ratio
    )


    # ========================================================
    # EMA20 RECLAIM
    # ========================================================

    distance_to_ema = abs(
        close - ema20
    )

    distance_atr = (
        distance_to_ema
        / atr
    )

    near_ema = (
        distance_atr
        <= MAX_EMA_DISTANCE_ATR
    )

    touched_ema = (
        low <= ema20
    )

    reclaimed_ema = (
        touched_ema
        and
        close > ema20
    )

    ema_pass = (
        near_ema
        and
        reclaimed_ema
    )


    # ========================================================
    # VWAP
    # ========================================================

    vwap = calculate_current_vwap(
        data_5m
    )

    if vwap is None:

        vwap_pass = False

    else:

        vwap_pass = (
            close > vwap
        )


    # ========================================================
    # FINAL ELIGIBILITY
    # ========================================================

    if mode == "PRIMARY":

        eligible = (
            trend_pass
            and
            volume_pass
            and
            ema_pass
        )

    else:

        eligible = (
            trend_pass
            and
            volume_pass
            and
            ema_pass
            and
            (
                vwap_pass
                if SECONDARY_REQUIRE_VWAP
                else True
            )
        )


    # ========================================================
    # LIVE PRICE
    # ========================================================

    live_price = float(
        data_5m["Close"].iloc[-1]
    )


    # ========================================================
    # TRADE PLAN
    # ========================================================

    entry = live_price

    stop = (
        entry
        - atr
        * STOP_ATR_MULTIPLIER
    )

    target = (
        entry
        + atr
        * TARGET_ATR_MULTIPLIER
    )

    risk = (
        entry - stop
    )

    reward = (
        target - entry
    )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )


    # ========================================================
    # SIGNAL ID
    # ========================================================

    signal_id = (
        f"{ticker}_"
        f"{mode}_"
        f"{signal_time.isoformat()}"
    )


    # ========================================================
    # PRINT DETAILS
    # ========================================================

    print(
        "Mode:",
        mode,
    )

    print(
        "Signal candle:",
        signal_time,
    )

    print(
        "Trend:",
        "PASS"
        if trend_pass
        else "FAIL",
    )

    print(
        "Volume ratio:",
        round(
            volume_ratio,
            2,
        ),
        "x",
    )

    print(
        "Required volume:",
        required_volume_ratio,
        "x",
    )

    print(
        "Volume:",
        "PASS"
        if volume_pass
        else "FAIL",
    )

    print(
        "EMA20:",
        round(
            ema20,
            2,
        ),
    )

    print(
        "ATR:",
        round(
            atr,
            2,
        ),
    )

    print(
        "EMA distance:",
        round(
            distance_atr,
            2,
        ),
        "ATR",
    )

    print(
        "Touched EMA20:",
        touched_ema,
    )

    print(
        "Reclaimed EMA20:",
        reclaimed_ema,
    )

    print(
        "EMA setup:",
        "PASS"
        if ema_pass
        else "FAIL",
    )

    if vwap is not None:

        print(
            "VWAP:",
            round(
                vwap,
                2,
            ),
        )

        print(
            "Above VWAP:",
            vwap_pass,
        )

    print(
        "ELIGIBLE:",
        eligible,
    )


    return {
        "Ticker":
            ticker,

        "Mode":
            mode,

        "Eligible":
            eligible,

        "SignalTime":
            str(signal_time),

        "SignalID":
            signal_id,

        "Entry":
            round(
                entry,
                2,
            ),

        "Stop":
            round(
                stop,
                2,
            ),

        "Target":
            round(
                target,
                2,
            ),

        "RR":
            round(
                rr,
                2,
            ),

        "ATR":
            round(
                atr,
                2,
            ),

        "EMA20":
            round(
                ema20,
                2,
            ),

        "EMA200":
            round(
                ema200,
                2,
            ),

        "VolumeRatio":
            round(
                volume_ratio,
                2,
            ),

        "RequiredVolumeRatio":
            required_volume_ratio,

        "VWAP":
            round(
                vwap,
                2,
            )
            if vwap
            is not None
            else None,

        "TrendPass":
            trend_pass,

        "VolumePass":
            volume_pass,

        "EMAPass":
            ema_pass,

        "VWAPPass":
            vwap_pass,
    }


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position(trade):

    entry = trade["Entry"]

    stop = trade["Stop"]

    risk_per_share = (
        entry - stop
    )

    if risk_per_share <= 0:
        return None

    capital = min(
        ACCOUNT_BALANCE,
        MAX_CAPITAL_PER_TRADE,
    )

    shares = (
        capital / entry
    )

    maximum_loss = (
        shares
        * risk_per_share
    )

    return {
        "Shares":
            round(
                shares,
                6,
            ),

        "Capital":
            round(
                shares
                * entry,
                2,
            ),

        "MaximumLoss":
            round(
                maximum_loss,
                2,
            ),
    }


# ============================================================
# DISCORD SIGNAL
# ============================================================

def send_trade_alert(
    trade,
    position,
):

    if trade["Mode"] == "PRIMARY":

        header = (
            "🔥 **PRIMARY MORNING SIGNAL**"
        )

    else:

        header = (
            "⚡ **SECONDARY PULLBACK SIGNAL**"
        )

    vwap_text = (
        f"${trade['VWAP']:.2f}"
        if trade["VWAP"]
        is not None
        else "N/A"
    )

    message = (
        f"{header}\n\n"

        f"**Ticker:** "
        f"{trade['Ticker']}\n"

        f"**Mode:** "
        f"{trade['Mode']}\n"

        f"**Direction:** LONG\n\n"

        f"**Entry:** "
        f"${trade['Entry']:.2f}\n"

        f"**Stop:** "
        f"${trade['Stop']:.2f}\n"

        f"**Target:** "
        f"${trade['Target']:.2f}\n"

        f"**R/R:** "
        f"{trade['RR']:.2f}:1\n\n"

        f"**EMA20:** "
        f"${trade['EMA20']:.2f}\n"

        f"**EMA200:** "
        f"${trade['EMA200']:.2f}\n"

        f"**VWAP:** "
        f"{vwap_text}\n"

        f"**Relative volume:** "
        f"{trade['VolumeRatio']:.2f}x\n"

        f"**Required volume:** "
        f"{trade['RequiredVolumeRatio']:.2f}x\n"

        f"**ATR:** "
        f"${trade['ATR']:.2f}\n\n"

        f"**Position:** "
        f"{position['Shares']} shares\n"

        f"**Capital:** "
        f"${position['Capital']:.2f}\n"

        f"**Maximum planned loss:** "
        f"${position['MaximumLoss']:.2f}\n\n"

        f"**Signal candle:** "
        f"{trade['SignalTime']}"
    )

    return send_discord_message(
        message
    )


# ============================================================
# SAVE SCAN
# ============================================================

def save_scan_log(results):

    os.makedirs(
        ".bot_state",
        exist_ok=True,
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    rows = []

    for trade in results:

        rows.append(
            {
                "TimeUTC":
                    timestamp,

                "Ticker":
                    trade["Ticker"],

                "Mode":
                    trade["Mode"],

                "Eligible":
                    trade["Eligible"],

                "Entry":
                    trade["Entry"],

                "Stop":
                    trade["Stop"],

                "Target":
                    trade["Target"],

                "RR":
                    trade["RR"],

                "VolumeRatio":
                    trade[
                        "VolumeRatio"
                    ],

                "RequiredVolumeRatio":
                    trade[
                        "RequiredVolumeRatio"
                    ],

                "EMA20":
                    trade["EMA20"],

                "EMA200":
                    trade["EMA200"],

                "VWAP":
                    trade["VWAP"],

                "TrendPass":
                    trade["TrendPass"],

                "VolumePass":
                    trade["VolumePass"],

                "EMAPass":
                    trade["EMAPass"],

                "VWAPPass":
                    trade["VWAPPass"],
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        ".bot_state/latest_scan.csv",
        index=False,
    )

    print(
        "\nScan log saved."
    )


# ============================================================
# MAIN BOT
# ============================================================

def run_bot():

    now = get_new_york_time()

    print("======================")
    print("PULLBACK BOT V3")
    print("======================")

    print(
        "New York time:",
        now.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
    )


    # ========================================================
    # MARKET HOURS
    # ========================================================

    if not market_is_open():

        print("\nMARKET CLOSED")
        print("Scanner exiting.")

        return


    # ========================================================
    # DON'T EVALUATE BEFORE FIRST HOURLY CANDLE CLOSES
    # ========================================================

    if (
        now.hour < 10
        or
        (
            now.hour == 10
            and
            now.minute < 30
        )
    ):

        print("\nNO TRADE")

        print(
            "Waiting for first "
            "9:30-10:30 candle "
            "to complete."
        )

        return


    # ========================================================
    # STOP NEW TRADES AFTER SECONDARY WINDOW
    # ========================================================

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    secondary_end = (
        SECONDARY_END_HOUR * 60
        + SECONDARY_END_MINUTE
    )

    if current_minutes > secondary_end:

        print("\nNO TRADE")

        print(
            "Reason: New-entry window "
            "has ended for today."
        )

        return


    # ========================================================
    # MARKET REGIME
    # ========================================================

    if not market_regime():

        print("\nNO TRADE")

        print(
            "Reason: Market regime "
            "not bullish."
        )

        return


    print("\nMarket approved.")
    print("Scanning watchlist...")


    results = []


    # ========================================================
    # SCAN WATCHLIST
    # ========================================================

    for ticker in WATCHLIST:

        try:

            result = scan_ticker(
                ticker
            )

            if result is not None:

                results.append(
                    result
                )

        except Exception as error:

            print(
                ticker,
                "ERROR:",
                error,
            )


    if not results:

        print(
            "\nNo valid signal candidates."
        )

        return


    # ========================================================
    # QUALIFYING TRADES
    # ========================================================

    eligible = [
        trade
        for trade in results
        if trade["Eligible"]
    ]


    if not eligible:

        print("\n======================")
        print("FINAL DECISION")
        print("======================")

        print("NO TRADE")

        save_scan_log(
            results
        )

        return


    # ========================================================
    # PRIORITIZE PRIMARY MODE
    # ========================================================

    primary_trades = [
        trade
        for trade in eligible
        if trade["Mode"]
        == "PRIMARY"
    ]

    secondary_trades = [
        trade
        for trade in eligible
        if trade["Mode"]
        == "SECONDARY"
    ]


    if primary_trades:

        candidates = (
            primary_trades
        )

    else:

        candidates = (
            secondary_trades
        )


    # ========================================================
    # PICK STRONGEST RELATIVE VOLUME
    # ========================================================

    candidates.sort(
        key=lambda trade:
            trade["VolumeRatio"],
        reverse=True,
    )

    best_trade = (
        candidates[0]
    )


    # ========================================================
    # POSITION SIZE
    # ========================================================

    position = calculate_position(
        best_trade
    )

    if position is None:

        print(
            "Invalid position sizing."
        )

        return


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n======================")
    print("QUALIFYING SETUP")
    print("======================")

    print(
        "Ticker:",
        best_trade["Ticker"],
    )

    print(
        "Mode:",
        best_trade["Mode"],
    )

    print(
        "Entry:",
        best_trade["Entry"],
    )

    print(
        "Stop:",
        best_trade["Stop"],
    )

    print(
        "Target:",
        best_trade["Target"],
    )

    print(
        "Volume Ratio:",
        best_trade["VolumeRatio"],
    )


    # ========================================================
    # DUPLICATE PROTECTION
    # ========================================================

    duplicate = already_alerted(
        best_trade["Ticker"],
        best_trade["SignalID"],
    )

    if duplicate:

        print(
            "Signal already alerted."
        )

    else:

        sent = send_trade_alert(
            best_trade,
            position,
        )

        if sent:

            mark_alerted(
                best_trade["Ticker"],
                best_trade["SignalID"],
            )


    save_scan_log(
        results
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    run_bot()
