import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# V1.3 VARIANT A — EXPANDED UNIVERSE DEVELOPMENT TEST
#
# PURPOSE:
# Test the SAME frozen Variant A setup across more symbols.
#
# We are NOT changing:
# - RVOL threshold
# - EMA rules
# - VWAP rule
# - market confirmation
# - breakout definition
# - retest definition
# - option target
# - ATR stop
# - max hold
#
# Only the SYMBOL UNIVERSE is expanded.
# ============================================================


# ------------------------------------------------------------
# ORIGINAL 7 SYMBOLS
# ------------------------------------------------------------

CORE_SYMBOLS = [
    "SPY",
    "QQQ",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMD",
    "TSLA",
]


# ------------------------------------------------------------
# EXPANDED UNIVERSE
# ------------------------------------------------------------

SYMBOLS = [
    "SPY",
    "QQQ",

    "NVDA",
    "AAPL",
    "MSFT",
    "AMD",
    "TSLA",

    "META",
    "AMZN",
    "GOOGL",
    "NFLX",
    "AVGO",

    "PLTR",
    "COIN",
    "MSTR",
    "HOOD",

    "JPM",
    "BAC",
    "GS",

    "XOM",
    "CVX",

    "WMT",
    "COST",

    "LLY",
    "UNH",

    "BA",
    "CAT",

    "IWM",
    "DIA",
]


# ============================================================
# TEST PERIOD
# ============================================================

DOWNLOAD_START = "2025-12-01"
DOWNLOAD_END = "2026-09-01"

TEST_START = pd.Timestamp(
    "2026-01-02",
    tz="America/New_York",
)

TEST_END = pd.Timestamp(
    "2026-08-03 16:00",
    tz="America/New_York",
)

INTERVAL = "5min"


# ============================================================
# FROZEN V1.3 VARIANT A RULES
# ============================================================

SIGNAL_START = "09:45"
SIGNAL_END = "11:15"

RVOL_MIN = 2.0

EMA_FAST = 9
EMA_SLOW = 21

OPENING_RANGE_END = "09:45"

ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 2.0

MAX_HOLD_MINUTES = 90

OPTION_DTE = 5
OPTION_TARGET_RETURN = 0.30
OPTION_STOP_RETURN = None
TRAILING_STOP_ACTIVATION_RETURN = None
TRAILING_STOP_DISTANCE = None
TARGET_DELTA = 0.50

ENTRY_FRICTION = 0.01
EXIT_FRICTION = 0.01

RISK_FREE_RATE = 0.04

RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_HISTORY_DAYS = 10

RETEST_TOLERANCE = 0.0015
MAX_RETEST_BARS = 6

ONE_POSITION_AT_A_TIME = True


# ============================================================
# PATHS
# ============================================================

CACHE_DIR = Path(
    "backtest_data_v13_expanded"
)

RESULTS_DIR = Path(
    "backtest_results_v13_expanded"
)

API_BASE = (
    "https://api.twelvedata.com/time_series"
)

FETCH_MAX_ATTEMPTS = 6
FETCH_RETRY_BASE_SECONDS = 30
FETCH_RETRY_MAX_SECONDS = 300


# ============================================================
# BASIC HELPERS
# ============================================================

def require_api_key():
    key = os.getenv(
        "TWELVE_DATA_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing."
        )

    return key


def ensure_directories():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DOWNLOAD WINDOWS
# ============================================================

def build_download_windows():

    starts = pd.date_range(
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        freq="MS",
    )

    windows = []

    for start in starts:

        if start >= pd.Timestamp(
            DOWNLOAD_END
        ):
            break

        next_month = (
            start
            + pd.offsets.MonthBegin(1)
        )

        windows.append(
            (
                start.strftime("%Y-%m-%d"),
                next_month.strftime("%Y-%m-%d"),
            )
        )

    return windows


# ============================================================
# TWELVE DATA
# ============================================================

def fetch_chunk(
    symbol,
    start_date,
    end_date,
    api_key,
):

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "America/New_York",
        "apikey": api_key,
        "format": "JSON",
        "order": "ASC",
    }

    for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):

        response = requests.get(
            API_BASE,
            params=params,
            timeout=60,
        )

        if response.status_code != 429:
            break

        if attempt == FETCH_MAX_ATTEMPTS:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")

        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = (
                FETCH_RETRY_BASE_SECONDS
                * (2 ** (attempt - 1))
            )

        delay = min(delay, FETCH_RETRY_MAX_SECONDS)

        print(
            "  Twelve Data rate limit reached; "
            f"retrying in {delay:g}s "
            f"(attempt {attempt + 1}/{FETCH_MAX_ATTEMPTS})"
        )

        time.sleep(delay)

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data error for {symbol}: "
            f"{payload.get('message', payload)}"
        )

    values = payload.get(
        "values",
        [],
    )

    if not values:
        return pd.DataFrame()

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

    df["symbol"] = symbol

    return df[
        [
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def load_cache(
    cache_file,
):

    df = pd.read_csv(
        cache_file
    )

    df["datetime"] = (
        pd.to_datetime(
            df["datetime"],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    return df


def save_cache(
    df,
    cache_file,
):

    save_df = df.copy()

    save_df["datetime"] = (
        save_df["datetime"]
        .dt.tz_convert("UTC")
        .astype(str)
    )

    save_df.to_csv(
        cache_file,
        index=False,
    )


def download_symbol(
    symbol,
    api_key,
):

    cache_file = (
        CACHE_DIR
        / f"{symbol}_5min.csv"
    )

    cached = pd.DataFrame()

    if cache_file.exists():

        cached = load_cache(
            cache_file
        )

        if not cached.empty:

            cached_end = (
                cached["datetime"].max()
            )

            required_cache_end = (
                TEST_END
                - pd.Timedelta(hours=1)
            )

            if cached_end >= required_cache_end:

                print(
                    f"{symbol}: using cached data"
                )

                return cached

    print()
    print(
        f"Downloading {symbol}"
    )

    frames = [cached] if not cached.empty else []

    for (
        start_date,
        end_date,
    ) in build_download_windows():

        window_start = pd.Timestamp(
            start_date,
            tz="America/New_York",
        )
        window_end = pd.Timestamp(
            end_date,
            tz="America/New_York",
        )

        if (
            not cached.empty
            and (
                (cached["datetime"] >= window_start)
                & (cached["datetime"] < window_end)
            ).any()
        ):
            print(
                f"  {start_date} -> {end_date} (cached)"
            )
            continue

        print(
            f"  {start_date} -> {end_date}"
        )

        chunk = fetch_chunk(
            symbol,
            start_date,
            end_date,
            api_key,
        )

        if not chunk.empty:
            frames.append(
                chunk
            )

            checkpoint = (
                pd.concat(
                    frames,
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=["datetime"]
                )
                .sort_values(
                    "datetime"
                )
                .reset_index(
                    drop=True
                )
            )

            save_cache(
                checkpoint,
                cache_file,
            )

            cached = checkpoint

        # Keep Twelve Data free-tier request rate safe.
        time.sleep(8)

    if not frames:

        raise RuntimeError(
            f"No historical data downloaded for {symbol}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["datetime"]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    save_cache(
        df,
        cache_file,
    )

    return df


def download_all_data():

    api_key = require_api_key()

    data = {}

    for number, symbol in enumerate(
        SYMBOLS,
        start=1,
    ):

        print()
        print(
            f"[{number}/{len(SYMBOLS)}] {symbol}"
        )

        data[symbol] = (
            download_symbol(
                symbol,
                api_key,
            )
        )

    return data


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = (
        df.copy()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    df["date"] = (
        df["datetime"].dt.date
    )

    df["time"] = (
        df["datetime"]
        .dt.strftime("%H:%M")
    )


    # EMA

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


    # ATR

    previous_close = (
        df["close"].shift(1)
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


    # VWAP

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
        / df["cum_volume"].replace(
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

    opening = (
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
        opening,
        left_on="date",
        right_index=True,
        how="left",
    )


    # --------------------------------------------------------
    # RVOL
    # --------------------------------------------------------

    rvol_average = []

    for (
        slot,
        slot_df,
    ) in df.groupby(
        "time",
        sort=False,
    ):

        avg = (
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

        rvol_average.append(
            pd.Series(
                avg.values,
                index=slot_df.index,
            )
        )

    df["slot_avg_volume"] = (
        pd.concat(
            rvol_average
        )
        .sort_index()
    )

    df["rvol"] = (
        df["volume"]
        / df["slot_avg_volume"]
    )

    return df


# ============================================================
# MARKET CONFIRMATION
# ============================================================

def create_market_confirmation(
    data,
):

    spy = data["SPY"][
        [
            "datetime",
            "close",
            "ema9",
            "ema21",
            "vwap",
        ]
    ].copy()

    qqq = data["QQQ"][
        [
            "datetime",
            "close",
            "ema9",
            "ema21",
            "vwap",
        ]
    ].copy()


    spy["spy_bullish"] = (
        (spy["close"] > spy["vwap"])
        & (
            spy["ema9"]
            > spy["ema21"]
        )
    )

    qqq["qqq_bullish"] = (
        (qqq["close"] > qqq["vwap"])
        & (
            qqq["ema9"]
            > qqq["ema21"]
        )
    )


    market = (
        spy[
            [
                "datetime",
                "spy_bullish",
            ]
        ]
        .merge(
            qqq[
                [
                    "datetime",
                    "qqq_bullish",
                ]
            ],
            on="datetime",
            how="inner",
        )
    )


    market["market_bullish"] = (
        market["spy_bullish"]
        & market["qqq_bullish"]
    )

    return market[
        [
            "datetime",
            "market_bullish",
        ]
    ]


# ============================================================
# FIND FRESH BREAKOUTS
# ============================================================

def find_breakouts(
    symbol,
    df,
    market,
):

    x = df.merge(
        market,
        on="datetime",
        how="left",
    )

    x["market_bullish"] = (
        x["market_bullish"]
        .fillna(False)
        .astype(bool)
    )

    x["previous_close"] = (
        x["close"].shift(1)
    )

    x["fresh_breakout"] = (
        (
            x["close"]
            > x["opening_range_high"]
        )
        & (
            x["previous_close"]
            <= x["opening_range_high"]
        )
    )

    valid = (
        (
            x["datetime"]
            >= TEST_START
        )
        & (
            x["datetime"]
            <= TEST_END
        )
        & (
            x["time"]
            >= SIGNAL_START
        )
        & (
            x["time"]
            <= SIGNAL_END
        )
        & (
            x["rvol"]
            >= RVOL_MIN
        )
        & (
            x["ema9"]
            > x["ema21"]
        )
        & (
            x["close"]
            > x["vwap"]
        )
        & x["fresh_breakout"]
        & x["market_bullish"]
        & x["atr"].notna()
    )

    output = (
        x.loc[
            valid
        ]
        .copy()
    )

    output["symbol"] = symbol

    return output


# ============================================================
# VARIANT A RETEST
#
# Breakout ->
# within 6 bars price revisits OR-high area ->
# candle closes back above OR high.
# ============================================================

def find_variant_a_signal(
    breakout,
    df,
):

    breakout_time = (
        breakout["datetime"]
    )

    matches = df.index[
        df["datetime"]
        == breakout_time
    ]

    if len(matches) == 0:
        return None

    breakout_idx = int(
        matches[0]
    )

    or_high = float(
        breakout[
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

    last_idx = min(
        breakout_idx
        + MAX_RETEST_BARS,
        len(df) - 1,
    )

    for i in range(
        breakout_idx + 1,
        last_idx + 1,
    ):

        bar = df.iloc[i]

        if (
            bar["datetime"].date()
            != breakout_time.date()
        ):
            break

        retest = (
            float(bar["low"])
            <= or_high
            and float(bar["low"])
            >= retest_floor
            and float(bar["close"])
            > or_high
        )

        if retest:

            return {
                "signal_idx": i,
                "signal_time":
                    bar["datetime"],
            }

    return None


# ============================================================
# BUILD ALL SIGNALS
# ============================================================

def build_signals(
    data,
    market,
):

    rows = []

    for symbol in SYMBOLS:

        df = data[symbol]

        breakouts = find_breakouts(
            symbol,
            df,
            market,
        )

        for (
            _,
            breakout,
        ) in breakouts.iterrows():

            detected = (
                find_variant_a_signal(
                    breakout,
                    df,
                )
            )

            if detected is None:
                continue

            signal_idx = (
                detected[
                    "signal_idx"
                ]
            )

            signal_bar = (
                df.iloc[
                    signal_idx
                ]
            )

            if (
                signal_bar["time"]
                > SIGNAL_END
            ):
                continue

            rows.append(
                {
                    "symbol":
                        symbol,

                    "universe_group":
                        (
                            "CORE"
                            if symbol
                            in CORE_SYMBOLS
                            else "EXPANDED"
                        ),

                    "breakout_time":
                        breakout[
                            "datetime"
                        ],

                    "signal_time":
                        signal_bar[
                            "datetime"
                        ],

                    "signal_idx":
                        signal_idx,

                    "rvol":
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

                    "atr":
                        float(
                            signal_bar[
                                "atr"
                            ]
                        ),

                    "signal_close":
                        float(
                            signal_bar[
                                "close"
                            ]
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    signals = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # If multiple stocks trigger on the same timestamp,
    # select the highest-RVOL setup.
    # --------------------------------------------------------

    signals = (
        signals
        .sort_values(
            [
                "signal_time",
                "rvol",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "signal_time"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return signals


# ============================================================
# BLACK-SCHOLES MODEL
#
# Same synthetic option approach as our earlier backtest.
# This is NOT exact historical expired-contract data.
# ============================================================

def normal_cdf(x):

    return (
        0.5
        * (
            1
            + math.erf(
                x
                / math.sqrt(2)
            )
        )
    )


def bs_call_price(
    spot,
    strike,
    time_years,
    rate,
    volatility,
):

    if time_years <= 0:

        return max(
            spot - strike,
            0,
        )

    volatility = max(
        volatility,
        0.05,
    )

    sqrt_t = math.sqrt(
        time_years
    )

    d1 = (
        math.log(
            spot / strike
        )
        + (
            rate
            + 0.5
            * volatility**2
        )
        * time_years
    ) / (
        volatility
        * sqrt_t
    )

    d2 = (
        d1
        - volatility
        * sqrt_t
    )

    return max(
        (
            spot
            * normal_cdf(d1)
        )
        - (
            strike
            * math.exp(
                -rate
                * time_years
            )
            * normal_cdf(d2)
        ),
        0.01,
    )


def bs_call_delta(
    spot,
    strike,
    time_years,
    rate,
    volatility,
):

    volatility = max(
        volatility,
        0.05,
    )

    sqrt_t = math.sqrt(
        time_years
    )

    d1 = (
        math.log(
            spot / strike
        )
        + (
            rate
            + 0.5
            * volatility**2
        )
        * time_years
    ) / (
        volatility
        * sqrt_t
    )

    return normal_cdf(
        d1
    )


# ============================================================
# VOLATILITY
# ============================================================

def estimate_volatility(
    df,
    entry_idx,
):

    bars_per_session = 78

    lookback = (
        20
        * bars_per_session
    )

    start = max(
        0,
        entry_idx - lookback,
    )

    history = (
        df.iloc[
            start:entry_idx
        ]["close"]
    )

    returns = np.log(
        history
        / history.shift(1)
    ).dropna()

    if len(returns) < 100:
        return np.nan

    annualized = (
        returns.std()
        * math.sqrt(
            252
            * bars_per_session
        )
    )

    return float(
        np.clip(
            annualized,
            0.10,
            2.50,
        )
    )


# ============================================================
# SELECT ~0.50 DELTA SYNTHETIC CALL
# ============================================================

def select_strike(
    spot,
    volatility,
):

    time_years = (
        OPTION_DTE / 365
    )

    width = max(
        spot * 0.10,
        5,
    )

    candidates = np.linspace(
        spot - width,
        spot + width,
        161,
    )

    best = None

    for strike in candidates:

        if strike <= 0:
            continue

        delta = bs_call_delta(
            spot,
            strike,
            time_years,
            RISK_FREE_RATE,
            volatility,
        )

        distance = abs(
            delta
            - TARGET_DELTA
        )

        if (
            best is None
            or distance
            < best[0]
        ):

            best = (
                distance,
                strike,
                delta,
            )

    return (
        float(best[1]),
        float(best[2]),
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    signal,
    df,
):

    signal_idx = int(
        signal[
            "signal_idx"
        ]
    )

    entry_idx = (
        signal_idx + 1
    )

    if entry_idx >= len(df):
        return None

    entry_bar = (
        df.iloc[
            entry_idx
        ]
    )

    signal_time = (
        signal[
            "signal_time"
        ]
    )

    if (
        entry_bar[
            "datetime"
        ].date()
        != signal_time.date()
    ):
        return None


    entry_time = (
        entry_bar[
            "datetime"
        ]
    )

    entry_stock = float(
        entry_bar[
            "open"
        ]
    )

    atr = float(
        signal[
            "atr"
        ]
    )

    stop_stock = (
        entry_stock
        - STOP_ATR_MULTIPLE
        * atr
    )


    volatility = (
        estimate_volatility(
            df,
            entry_idx,
        )
    )

    if not np.isfinite(
        volatility
    ):
        return None


    (
        strike,
        delta,
    ) = select_strike(
        entry_stock,
        volatility,
    )


    theoretical_entry = (
        bs_call_price(
            entry_stock,
            strike,
            OPTION_DTE / 365,
            RISK_FREE_RATE,
            volatility,
        )
    )

    modeled_entry = (
        theoretical_entry
        * (
            1
            + ENTRY_FRICTION
        )
    )

    target_price = (
        modeled_entry
        * (
            1
            + OPTION_TARGET_RETURN
        )
    )

    option_stop_price = (
        modeled_entry
        * (
            1
            + OPTION_STOP_RETURN
        )
        if OPTION_STOP_RETURN
        is not None
        else None
    )

    trailing_activation_price = (
        modeled_entry
        * (
            1
            + TRAILING_STOP_ACTIVATION_RETURN
        )
        if TRAILING_STOP_ACTIVATION_RETURN
        is not None
        else None
    )

    trailing_stop_price = None


    final_idx = min(
        entry_idx
        + MAX_HOLD_MINUTES // 5,
        len(df) - 1,
    )

    exit_reason = "TIME"
    exit_time = None
    exit_stock = None
    modeled_exit = None


    for i in range(
        entry_idx,
        final_idx + 1,
    ):

        bar = df.iloc[i]

        if (
            bar[
                "datetime"
            ].date()
            != entry_time.date()
        ):
            break

        elapsed = (
            (
                bar[
                    "datetime"
                ]
                - entry_time
            )
            .total_seconds()
            / 60
        )

        remaining_days = max(
            OPTION_DTE
            - elapsed / 1440,
            1 / 1440,
        )

        remaining_years = (
            remaining_days
            / 365
        )


        # ----------------------------------------------------
        # Optional modeled-option stop.
        #
        # Calls are monotonic in the underlying spot, so the
        # bar low is the conservative point for detecting a
        # premium stop. When both stop and target are touched
        # inside the same five-minute bar, stop-first handling
        # is retained.
        # ----------------------------------------------------

        if option_stop_price is not None:

            low_spot = float(
                bar["low"]
            )

            high_spot = float(
                bar["high"]
            )

            low_option = (
                bs_call_price(
                    low_spot,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    volatility,
                )
                * (
                    1
                    - EXIT_FRICTION
                )
            )

            if low_option <= option_stop_price:

                high_option = (
                    bs_call_price(
                        high_spot,
                        strike,
                        remaining_years,
                        RISK_FREE_RATE,
                        volatility,
                    )
                    * (
                        1
                        - EXIT_FRICTION
                    )
                )

                if high_option >= option_stop_price:

                    for _ in range(40):

                        midpoint = (
                            low_spot
                            + high_spot
                        ) / 2

                        price = (
                            bs_call_price(
                                midpoint,
                                strike,
                                remaining_years,
                                RISK_FREE_RATE,
                                volatility,
                            )
                            * (
                                1
                                - EXIT_FRICTION
                            )
                        )

                        if price >= option_stop_price:

                            high_spot = midpoint

                        else:

                            low_spot = midpoint

                    exit_stock = high_spot
                    modeled_exit = option_stop_price

                else:

                    # The full bar is below the stop threshold.
                    # Model a gap fill at the bar open rather
                    # than granting an unavailable stop price.
                    exit_stock = float(
                        bar["open"]
                    )

                    modeled_exit = (
                        bs_call_price(
                            exit_stock,
                            strike,
                            remaining_years,
                            RISK_FREE_RATE,
                            volatility,
                        )
                        * (
                            1
                            - EXIT_FRICTION
                        )
                    )

                exit_reason = "OPTION_STOP"
                exit_time = bar["datetime"]

                break


        # ----------------------------------------------------
        # Optional trailing stop.
        #
        # The trail activates after the modeled option reaches
        # its configured gain and follows the highest modeled
        # option price. Same-bar ambiguity is handled
        # conservatively: if a bar can both establish and break
        # the trail, the trailing stop is assumed to fill first.
        # ----------------------------------------------------

        if (
            trailing_activation_price
            is not None
            and TRAILING_STOP_DISTANCE
            is not None
        ):

            low_spot = float(
                bar["low"]
            )

            high_spot = float(
                bar["high"]
            )

            low_option = (
                bs_call_price(
                    low_spot,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    volatility,
                )
                * (
                    1
                    - EXIT_FRICTION
                )
            )

            high_option = (
                bs_call_price(
                    high_spot,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    volatility,
                )
                * (
                    1
                    - EXIT_FRICTION
                )
            )

            if high_option >= trailing_activation_price:

                candidate_stop = (
                    high_option
                    * (
                        1
                        - TRAILING_STOP_DISTANCE
                    )
                )

                trailing_stop_price = (
                    candidate_stop
                    if trailing_stop_price
                    is None
                    else max(
                        trailing_stop_price,
                        candidate_stop,
                    )
                )

            if (
                trailing_stop_price
                is not None
                and low_option
                <= trailing_stop_price
            ):

                if high_option >= trailing_stop_price:

                    for _ in range(40):

                        midpoint = (
                            low_spot
                            + high_spot
                        ) / 2

                        price = (
                            bs_call_price(
                                midpoint,
                                strike,
                                remaining_years,
                                RISK_FREE_RATE,
                                volatility,
                            )
                            * (
                                1
                                - EXIT_FRICTION
                            )
                        )

                        if price >= trailing_stop_price:

                            high_spot = midpoint

                        else:

                            low_spot = midpoint

                    exit_stock = high_spot
                    modeled_exit = trailing_stop_price

                else:

                    exit_stock = float(
                        bar["open"]
                    )

                    modeled_exit = (
                        bs_call_price(
                            exit_stock,
                            strike,
                            remaining_years,
                            RISK_FREE_RATE,
                            volatility,
                        )
                        * (
                            1
                            - EXIT_FRICTION
                        )
                    )

                exit_reason = "TRAILING_STOP"
                exit_time = bar["datetime"]

                break


        # ----------------------------------------------------
        # Conservative stop-first same-bar handling
        # ----------------------------------------------------

        if (
            float(
                bar["low"]
            )
            <= stop_stock
        ):

            exit_stock = (
                stop_stock
            )

            modeled_exit = (
                bs_call_price(
                    exit_stock,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    volatility,
                )
                * (
                    1
                    - EXIT_FRICTION
                )
            )

            exit_reason = (
                "UNDERLYING_STOP"
            )

            exit_time = (
                bar[
                    "datetime"
                ]
            )

            break


        # ----------------------------------------------------
        # +30% option target
        # ----------------------------------------------------

        high_option = (
            bs_call_price(
                float(
                    bar["high"]
                ),
                strike,
                remaining_years,
                RISK_FREE_RATE,
                volatility,
            )
            * (
                1
                - EXIT_FRICTION
            )
        )

        if high_option >= target_price:

            low_spot = float(
                bar["low"]
            )

            high_spot = float(
                bar["high"]
            )

            for _ in range(40):

                midpoint = (
                    low_spot
                    + high_spot
                ) / 2

                price = (
                    bs_call_price(
                        midpoint,
                        strike,
                        remaining_years,
                        RISK_FREE_RATE,
                        volatility,
                    )
                    * (
                        1
                        - EXIT_FRICTION
                    )
                )

                if price >= target_price:

                    high_spot = (
                        midpoint
                    )

                else:

                    low_spot = (
                        midpoint
                    )

            exit_stock = (
                high_spot
            )

            modeled_exit = (
                bs_call_price(
                    exit_stock,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    volatility,
                )
                * (
                    1
                    - EXIT_FRICTION
                )
            )

            exit_reason = (
                "OPTION_TARGET"
            )

            exit_time = (
                bar[
                    "datetime"
                ]
            )

            break


    # --------------------------------------------------------
    # TIME EXIT
    # --------------------------------------------------------

    if exit_time is None:

        same_day = df[
            (
                df["datetime"]
                >= entry_time
            )
            & (
                df["datetime"].dt.date
                == entry_time.date()
            )
        ]

        if same_day.empty:
            return None

        max_exit_time = (
            entry_time
            + pd.Timedelta(
                minutes=MAX_HOLD_MINUTES
            )
        )

        eligible = same_day[
            same_day["datetime"]
            <= max_exit_time
        ]

        if eligible.empty:
            return None

        final_bar = (
            eligible.iloc[-1]
        )

        exit_time = (
            final_bar[
                "datetime"
            ]
        )

        exit_stock = float(
            final_bar[
                "close"
            ]
        )

        elapsed = (
            (
                exit_time
                - entry_time
            )
            .total_seconds()
            / 60
        )

        remaining_days = max(
            OPTION_DTE
            - elapsed / 1440,
            1 / 1440,
        )

        modeled_exit = (
            bs_call_price(
                exit_stock,
                strike,
                remaining_days / 365,
                RISK_FREE_RATE,
                volatility,
            )
            * (
                1
                - EXIT_FRICTION
            )
        )


    option_return = (
        modeled_exit
        / modeled_entry
        - 1
    )

    underlying_return = (
        exit_stock
        / entry_stock
        - 1
    )


    return {
        "symbol":
            signal[
                "symbol"
            ],

        "universe_group":
            signal[
                "universe_group"
            ],

        "breakout_time":
            signal[
                "breakout_time"
            ],

        "signal_time":
            signal[
                "signal_time"
            ],

        "entry_time":
            entry_time,

        "exit_time":
            exit_time,

        "rvol":
            signal[
                "rvol"
            ],

        "entry_stock":
            entry_stock,

        "stop_stock":
            stop_stock,

        "exit_stock":
            exit_stock,

        "option_strike":
            strike,

        "option_delta":
            delta,

        "estimated_volatility":
            volatility,

        "modeled_option_entry":
            modeled_entry,

        "modeled_option_exit":
            modeled_exit,

        "option_return_pct":
            option_return * 100,

        "underlying_return_pct":
            underlying_return * 100,

        "exit_reason":
            exit_reason,

        "hold_minutes":
            (
                (
                    exit_time
                    - entry_time
                )
                .total_seconds()
                / 60
            ),
    }


# ============================================================
# ONE POSITION AT A TIME
# ============================================================

def run_portfolio(
    signals,
    data,
):

    trades = []

    next_available_time = None

    for (
        _,
        signal,
    ) in signals.sort_values(
        "signal_time"
    ).iterrows():

        if (
            ONE_POSITION_AT_A_TIME
            and next_available_time
            is not None
            and signal[
                "signal_time"
            ]
            < next_available_time
        ):
            continue

        symbol = (
            signal[
                "symbol"
            ]
        )

        trade = simulate_trade(
            signal,
            data[symbol],
        )

        if trade is None:
            continue

        trades.append(
            trade
        )

        if ONE_POSITION_AT_A_TIME:

            next_available_time = (
                trade[
                    "exit_time"
                ]
            )

    return pd.DataFrame(
        trades
    )


# ============================================================
# METRICS
# ============================================================

def profit_factor(
    returns,
):

    gross_profit = (
        returns[
            returns > 0
        ].sum()
    )

    gross_loss = -(
        returns[
            returns < 0
        ].sum()
    )

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return (
        gross_profit
        / gross_loss
    )


def max_drawdown(
    returns,
):

    if len(returns) == 0:
        return np.nan

    equity = (
        1 + returns
    ).cumprod()

    peak = (
        equity.cummax()
    )

    drawdown = (
        equity / peak
        - 1
    )

    return float(
        drawdown.min()
    )


def longest_losing_streak(
    returns,
):

    longest = 0
    current = 0

    for value in returns:

        if value < 0:

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 0

    return longest


def summarize(
    trades,
):

    if trades.empty:

        return {
            "trades": 0,
        }

    returns = (
        trades[
            "option_return_pct"
        ]
        / 100
    )

    underlying = (
        trades[
            "underlying_return_pct"
        ]
        / 100
    )

    return {
        "trades":
            len(trades),

        "wins":
            int(
                (
                    returns > 0
                ).sum()
            ),

        "losses":
            int(
                (
                    returns < 0
                ).sum()
            ),

        "win_rate_pct":
            float(
                (
                    returns > 0
                ).mean()
                * 100
            ),

        "avg_option_return_pct":
            float(
                returns.mean()
                * 100
            ),

        "median_option_return_pct":
            float(
                returns.median()
                * 100
            ),

        "profit_factor":
            float(
                profit_factor(
                    returns
                )
            ),

        "max_drawdown_pct":
            float(
                max_drawdown(
                    returns
                )
                * 100
            ),

        "avg_underlying_return_pct":
            float(
                underlying.mean()
                * 100
            ),

        "underlying_win_rate_pct":
            float(
                (
                    underlying > 0
                )
                .mean()
                * 100
            ),

        "longest_losing_streak":
            longest_losing_streak(
                returns
            ),

        "target_hits":
            int(
                (
                    trades[
                        "exit_reason"
                    ]
                    == "OPTION_TARGET"
                )
                .sum()
            ),

        "stop_hits":
            int(
                (
                    trades[
                        "exit_reason"
                    ]
                    == "UNDERLYING_STOP"
                )
                .sum()
            ),

        "option_stop_hits":
            int(
                (
                    trades[
                        "exit_reason"
                    ]
                    == "OPTION_STOP"
                )
                .sum()
            ),

        "trailing_stop_hits":
            int(
                (
                    trades[
                        "exit_reason"
                    ]
                    == "TRAILING_STOP"
                )
                .sum()
            ),

        "time_exits":
            int(
                (
                    trades[
                        "exit_reason"
                    ]
                    == "TIME"
                )
                .sum()
            ),
    }


# ============================================================
# BY TICKER
# ============================================================

def build_by_ticker(
    trades,
):

    rows = []

    for (
        symbol,
        group,
    ) in trades.groupby(
        "symbol"
    ):

        returns = (
            group[
                "option_return_pct"
            ]
            / 100
        )

        underlying = (
            group[
                "underlying_return_pct"
            ]
            / 100
        )

        rows.append(
            {
                "symbol":
                    symbol,

                "group":
                    (
                        "CORE"
                        if symbol
                        in CORE_SYMBOLS
                        else "EXPANDED"
                    ),

                "trades":
                    len(group),

                "win_rate_pct":
                    (
                        returns > 0
                    ).mean()
                    * 100,

                "avg_option_return_pct":
                    returns.mean()
                    * 100,

                "profit_factor":
                    profit_factor(
                        returns
                    ),

                "avg_underlying_return_pct":
                    underlying.mean()
                    * 100,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "trades",
            "avg_option_return_pct",
        ],
        ascending=[
            False,
            False,
        ],
    )


# ============================================================
# BY MONTH
# ============================================================

def build_by_month(
    trades,
):

    df = trades.copy()

    df["month"] = (
        pd.to_datetime(
            df[
                "entry_time"
            ],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
        .dt.strftime(
            "%Y-%m"
        )
    )

    rows = []

    for (
        month,
        group,
    ) in df.groupby(
        "month"
    ):

        returns = (
            group[
                "option_return_pct"
            ]
            / 100
        )

        rows.append(
            {
                "month":
                    month,

                "trades":
                    len(group),

                "win_rate_pct":
                    (
                        returns > 0
                    ).mean()
                    * 100,

                "avg_option_return_pct":
                    returns.mean()
                    * 100,

                "profit_factor":
                    profit_factor(
                        returns
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print(
        "=" * 75
    )

    print(
        "V1.3 VARIANT A — EXPANDED UNIVERSE BACKTEST"
    )

    print(
        "=" * 75
    )

    print(
        f"Symbols: {len(SYMBOLS)}"
    )

    print(
        f"Development period: "
        f"{TEST_START.date()} "
        f"through "
        f"{TEST_END.date()}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Strategy parameters are unchanged."
    )

    print(
        "Only the symbol universe is expanded."
    )

    print()


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    raw_data = (
        download_all_data()
    )


    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    data = {}

    for symbol in SYMBOLS:

        print(
            f"Indicators: {symbol}"
        )

        data[symbol] = (
            add_indicators(
                raw_data[
                    symbol
                ]
            )
        )


    # --------------------------------------------------------
    # MARKET CONFIRMATION
    # --------------------------------------------------------

    market = (
        create_market_confirmation(
            data
        )
    )


    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    print()
    print(
        "Finding Variant A signals..."
    )

    signals = (
        build_signals(
            data,
            market,
        )
    )

    print(
        f"Portfolio-eligible signal timestamps: "
        f"{len(signals)}"
    )

    signals.to_csv(
        RESULTS_DIR
        / "v13_expanded_signals.csv",
        index=False,
    )


    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    trades = (
        run_portfolio(
            signals,
            data,
        )
    )

    trades.to_csv(
        RESULTS_DIR
        / "v13_expanded_trades.csv",
        index=False,
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        summarize(
            trades
        )
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "v13_expanded_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # BY TICKER / MONTH
    # --------------------------------------------------------

    by_ticker = (
        build_by_ticker(
            trades
        )
        if not trades.empty
        else pd.DataFrame()
    )

    by_month = (
        build_by_month(
            trades
        )
        if not trades.empty
        else pd.DataFrame()
    )

    by_ticker.to_csv(
        RESULTS_DIR
        / "v13_expanded_by_ticker.csv",
        index=False,
    )

    by_month.to_csv(
        RESULTS_DIR
        / "v13_expanded_by_month.csv",
        index=False,
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print()
    print(
        "=" * 75
    )

    print(
        "V1.3 EXPANDED UNIVERSE RESULTS"
    )

    print(
        "=" * 75
    )

    for (
        key,
        value,
    ) in summary.items():

        print(
            f"{key}: {value}"
        )


    print()
    print(
        "ORIGINAL 7-SYMBOL BASELINE"
    )

    print(
        "-" * 75
    )

    print(
        "Trades: 11"
    )

    print(
        "Win rate: 72.7%"
    )

    print(
        "Average modeled option return: +3.70%"
    )

    print(
        "Profit factor: 1.50"
    )


    if not by_ticker.empty:

        print()
        print(
            "RESULTS BY TICKER"
        )

        print(
            "-" * 75
        )

        print(
            by_ticker.to_string(
                index=False
            )
        )


    if not by_month.empty:

        print()
        print(
            "RESULTS BY MONTH"
        )

        print(
            "-" * 75
        )

        print(
            by_month.to_string(
                index=False
            )
        )


    print()
    print(
        "INTERPRETATION GUIDE"
    )

    print(
        "-" * 75
    )

    print(
        "We are NOT requiring the expanded universe "
        "to maintain exactly a 72.7% win rate."
    )

    print(
        "A larger sample with ~55-60%+ wins, PF >1.3, "
        "positive expectancy and diversified results "
        "could be more trustworthy."
    )

    print()
    print(
        "NOTE: option returns are still synthetic/model-based, "
        "not exact historical expired-option quotes."
    )

    print()
    print(
        "Files written to:"
    )

    print(
        RESULTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()
