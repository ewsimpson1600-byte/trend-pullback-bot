import yfinance as yf
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# ==========================================
# SETTINGS
# ==========================================

WATCHLIST = [
    "SPY",
    "QQQ",
    "NVDA",
    "MSFT",
    "AAPL",
    "AMD",
    "TSLA"
]

ACCOUNT_BALANCE = 20.00
MAX_CAPITAL_PER_TRADE = 20.00

MIN_CONFIDENCE = 70

# Pullback must be within 1% of EMA20
PULLBACK_DISTANCE = 0.01


# ==========================================
# MARKET HOURS
# ==========================================

def market_is_open():

    now_et = datetime.now(
        ZoneInfo("America/New_York")
    )

    # Monday = 0, Friday = 4
    if now_et.weekday() > 4:
        return False

    current_minutes = (
        now_et.hour * 60
        + now_et.minute
    )

    market_open = (
        9 * 60
        + 30
    )

    market_close = (
        16 * 60
    )

    return (
        market_open
        <= current_minutes
        < market_close
    )


# ==========================================
# DOWNLOAD DATA
# ==========================================

def get_data(
    ticker,
    period="6mo",
    interval="1h"
):

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker}"
        )

    # Handle yfinance MultiIndex
    if isinstance(
        data.columns,
        pd.MultiIndex
    ):
        try:
            data = data.xs(
                ticker,
                axis=1,
                level=1
            )
        except Exception:
            data.columns = (
                data.columns
                .get_level_values(0)
            )

    return data.dropna()


# ==========================================
# MARKET REGIME
# ==========================================

def market_regime_check():

    print("\n======================")
    print("MARKET REGIME")
    print("======================")

    spy = get_data("SPY")

    spy["EMA200"] = (
        spy["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    price = float(
        spy["Close"].iloc[-1]
    )

    ema200 = float(
        spy["EMA200"].iloc[-1]
    )

    returns = (
        spy["Close"]
        .pct_change()
        .dropna()
    )

    volatility = float(
        returns.tail(20).std()
        * 100
    )

    bullish = (
        price > ema200
    )

    print(
        "SPY:",
        round(price, 2)
    )

    print(
        "EMA200:",
        round(ema200, 2)
    )

    print(
        "Trend:",
        "BULLISH"
        if bullish
        else "BEARISH"
    )

    print(
        "20-period volatility:",
        round(volatility, 3),
        "%"
    )

    market_ok = (
        bullish
        and volatility < 1.5
    )

    print(
        "Market Status:",
        "ACCEPTABLE"
        if market_ok
        else "NO LONG TRADES"
    )

    return market_ok


# ==========================================
# SPY BENCHMARK RETURN
# ==========================================

def get_spy_return():

    spy = get_data(
        "SPY",
        period="1mo",
        interval="1h"
    )

    return (
        float(
            spy["Close"].iloc[-1]
        )
        /
        float(
            spy["Close"].iloc[0]
        )
    ) - 1


# ==========================================
# STOCK SCANNER
# ==========================================

def scan_stock(
    ticker,
    spy_return
):

    print(
        "Scanning:",
        ticker
    )

    data = get_data(ticker)

    if len(data) < 201:
        raise ValueError(
            f"Not enough data for {ticker}"
        )

    # --------------------------
    # Indicators
    # --------------------------

    data["EMA20"] = (
        data["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    data["EMA200"] = (
        data["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    data["AvgVolume"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    # --------------------------
    # ATR
    # --------------------------

    previous_close = (
        data["Close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            (
                data["High"]
                - data["Low"]
            ),
            (
                data["High"]
                - previous_close
            ).abs(),
            (
                data["Low"]
                - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        tr
        .rolling(14)
        .mean()
    )

    current = data.iloc[-1]
    previous = data.iloc[-2]

    price = float(
        current["Close"]
    )

    ema20 = float(
        current["EMA20"]
    )

    ema200 = float(
        current["EMA200"]
    )

    volume = float(
        current["Volume"]
    )

    avg_volume = float(
        current["AvgVolume"]
    )

    atr = float(
        current["ATR"]
    )

    confidence = 0
    reasons = []


    # ======================================
    # 1. TREND (+20)
    # ======================================

    trend_pass = (
        price > ema200
    )

    if trend_pass:
        confidence += 20
        reasons.append(
            "Trend PASS"
        )
    else:
        reasons.append(
            "Trend FAIL"
        )


    # ======================================
    # 2. PULLBACK (+20)
    # ======================================

    pullback_distance = (
        abs(
            price - ema20
        )
        / price
    )

    pullback_pass = (
        pullback_distance
        <= PULLBACK_DISTANCE
    )

    if pullback_pass:
        confidence += 20
        reasons.append(
            "Pullback PASS"
        )
    else:
        reasons.append(
            "Pullback FAIL"
        )


    # ======================================
    # 3. VOLUME (+15)
    # ======================================

    volume_pass = (
        volume > avg_volume
    )

    if volume_pass:
        confidence += 15
        reasons.append(
            "Volume PASS"
        )
    else:
        reasons.append(
            "Volume FAIL"
        )


    # ======================================
    # 4. REVERSAL (+15)
    # ======================================

    bullish_engulfing = (
        current["Close"]
        > current["Open"]
        and
        previous["Close"]
        < previous["Open"]
        and
        current["Close"]
        > previous["Open"]
        and
        current["Open"]
        <= previous["Close"]
    )

    body = abs(
        float(
            current["Close"]
        )
        -
        float(
            current["Open"]
        )
    )

    lower_wick = (
        min(
            float(
                current["Open"]
            ),
            float(
                current["Close"]
            )
        )
        -
        float(
            current["Low"]
        )
    )

    rejection_wick = (
        current["Close"]
        > current["Open"]
        and
        body > 0
        and
        lower_wick >= body
    )

    higher_low = (
        current["Low"]
        > previous["Low"]
        and
        current["Close"]
        > current["Open"]
    )

    reversal_pass = (
        bullish_engulfing
        or rejection_wick
        or higher_low
    )

    if reversal_pass:
        confidence += 15
        reasons.append(
            "Reversal PASS"
        )
    else:
        reasons.append(
            "Reversal FAIL"
        )


    # ======================================
    # 5. RELATIVE STRENGTH (+10)
    # ======================================

    stock_return = (
        float(
            data["Close"].iloc[-1]
        )
        /
        float(
            data["Close"].iloc[0]
        )
    ) - 1

    relative_strength = (
        stock_return
        - spy_return
    )

    rs_pass = (
        relative_strength > 0
    )

    if rs_pass:
        confidence += 10
        reasons.append(
            "Relative Strength PASS"
        )
    else:
        reasons.append(
            "Relative Strength FAIL"
        )


    # ======================================
    # TRADE PLAN
    # ======================================

    entry = price

    stop = (
        entry
        - atr
    )

    risk_per_share = (
        entry
        - stop
    )

    target = (
        entry
        + risk_per_share * 2
    )

    rr = (
        (target - entry)
        / risk_per_share
    )


    # ======================================
    # REQUIRED CONDITIONS
    # ======================================

    required_conditions = (
        trend_pass
        and pullback_pass
        and reversal_pass
        and rr >= 2
    )

    eligible = (
        confidence
        >= MIN_CONFIDENCE
        and
        required_conditions
    )

    return {
        "Ticker":
            ticker,

        "Score":
            confidence,

        "Eligible":
            eligible,

        "Entry":
            round(
                entry,
                2
            ),

        "Stop":
            round(
                stop,
                2
            ),

        "Target":
            round(
                target,
                2
            ),

        "RR":
            round(
                rr,
                2
            ),

        "ATR":
            round(
                atr,
                2
            ),

        "RelativeStrength":
            round(
                relative_strength
                * 100,
                2
            ),

        "Reasons":
            reasons
    }


# ==========================================
# POSITION SIZING
# ==========================================

def calculate_position(trade):

    entry = trade["Entry"]
    stop = trade["Stop"]

    risk_per_share = (
        entry - stop
    )

    if risk_per_share <= 0:
        return None

    capital_limit = min(
        ACCOUNT_BALANCE,
        MAX_CAPITAL_PER_TRADE
    )

    shares_by_capital = (
        capital_limit
        / entry
    )

    position_size = (
        shares_by_capital
    )

    capital_used = (
        position_size
        * entry
    )

    maximum_loss = (
        position_size
        * risk_per_share
    )

    return {
        "Shares":
            round(
                position_size,
                6
            ),

        "Capital":
            round(
                capital_used,
                2
            ),

        "MaximumLoss":
            round(
                maximum_loss,
                2
            )
    }


# ==========================================
# OUTPUT
# ==========================================

def print_trade(
    trade,
    position
):

    print("\n======================")
    print("TRADE CANDIDATE")
    print("======================")

    print(
        "Ticker:",
        trade["Ticker"]
    )

    print(
        "Direction: LONG"
    )

    print(
        "Confidence:",
        trade["Score"],
        "/80"
    )

    print(
        "Entry:",
        trade["Entry"]
    )

    print(
        "Stop:",
        trade["Stop"]
    )

    print(
        "Target:",
        trade["Target"]
    )

    print(
        "Risk/Reward:",
        trade["RR"]
    )

    print(
        "Relative Strength:",
        trade["RelativeStrength"],
        "%"
    )

    print(
        "Position Size:",
        position["Shares"],
        "shares"
    )

    print(
        "Capital Used: $",
        position["Capital"]
    )

    print(
        "Maximum Planned Loss: $",
        position["MaximumLoss"]
    )

    print("\nConditions:")

    for reason in trade["Reasons"]:
        print(
            "-",
            reason
        )


# ==========================================
# LOGGING
# ==========================================

def save_run_log(
    results,
    best_trade=None
):

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

                "Score":
                    trade["Score"],

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

                "RelativeStrength":
                    trade[
                        "RelativeStrength"
                    ],

                "Selected":
                    (
                        best_trade
                        is not None
                        and
                        trade["Ticker"]
                        ==
                        best_trade["Ticker"]
                    )
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        "latest_scan.csv",
        index=False
    )

    print(
        "\nLatest scan saved "
        "to latest_scan.csv"
    )


# ==========================================
# MAIN BOT
# ==========================================

def run_bot():

    now_et = datetime.now(
        ZoneInfo(
            "America/New_York"
        )
    )

    print("======================")
    print("TREND PULLBACK BOT")
    print("======================")

    print(
        "New York time:",
        now_et.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
    )

    print(
        "UTC:",
        datetime.now(
            timezone.utc
        )
    )

    # --------------------------------------
    # MARKET HOURS CHECK
    # --------------------------------------

    if not market_is_open():

        print(
            "\nMARKET CLOSED"
        )

        print(
            "Scanner exiting."
        )

        return


    # --------------------------------------
    # MARKET REGIME FILTER
    # --------------------------------------

    if not market_regime_check():

        print(
            "\nFINAL DECISION: "
            "NO TRADE"
        )

        print(
            "Reason: Market regime "
            "rejected long setups."
        )

        return


    print(
        "\nMarket approved."
    )

    print(
        "Scanning watchlist...\n"
    )

    spy_return = (
        get_spy_return()
    )

    results = []


    # --------------------------------------
    # SCAN WATCHLIST
    # --------------------------------------

    for ticker in WATCHLIST:

        try:

            result = scan_stock(
                ticker,
                spy_return
            )

            results.append(
                result
            )

            print(
                ticker,
                "Score:",
                result["Score"],
                "Eligible:",
                result["Eligible"]
            )

        except Exception as error:

            print(
                ticker,
                "ERROR:",
                error
            )


    # --------------------------------------
    # RANK RESULTS
    # --------------------------------------

    results.sort(
        key=lambda x:
            x["Score"],
        reverse=True
    )

    eligible = [
        trade
        for trade
        in results
        if trade["Eligible"]
    ]


    # --------------------------------------
    # NO TRADE
    # --------------------------------------

    if not eligible:

        print(
            "\n======================"
        )

        print(
            "FINAL DECISION"
        )

        print(
            "======================"
        )

        print(
            "NO TRADE"
        )

        if results:

            print(
                "Best setup:",
                results[0]["Ticker"]
            )

            print(
                "Score:",
                results[0]["Score"],
                "/80"
            )

        save_run_log(
            results
        )

        return


    # --------------------------------------
    # BEST TRADE
    # --------------------------------------

    best_trade = (
        eligible[0]
    )

    position = (
        calculate_position(
            best_trade
        )
    )

    print_trade(
        best_trade,
        position
    )

    save_run_log(
        results,
        best_trade
    )


# ==========================================
# START BOT
# ==========================================

if __name__ == "__main__":
    run_bot()
