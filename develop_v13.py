import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# V1.3 DEVELOPMENT CONFIG
#
# DEVELOPMENT DATA:
# January 2, 2026 -> August 3, 2026
#
# IMPORTANT:
# We are NOT changing the strategy rules here.
# We are only expanding the development period.
# ============================================================

SYMBOLS = [
    "SPY",
    "QQQ",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMD",
    "TSLA",
]

DOWNLOAD_START = "2025-12-01"

TEST_START = pd.Timestamp(
    "2026-01-02",
    tz="America/New_York",
)

TEST_END = pd.Timestamp(
    "2026-08-03 16:00",
    tz="America/New_York",
)

INTERVAL = "5min"

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
TARGET_DELTA = 0.50

ENTRY_FRICTION = 0.01
EXIT_FRICTION = 0.01

RISK_FREE_RATE = 0.04

RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_HISTORY_DAYS = 10

ONE_POSITION_AT_A_TIME = True


# ============================================================
# RETEST RULES
#
# These are unchanged from our original V1.3 development test.
# ============================================================

RETEST_TOLERANCE = 0.0015

MAX_RETEST_BARS = 6


# ============================================================
# FILE LOCATIONS
# ============================================================

CACHE_DIR = Path("backtest_data_v13")
RESULTS_DIR = Path("backtest_results_v13")

API_BASE = "https://api.twelvedata.com/time_series"


# ============================================================
# BASIC HELPERS
# ============================================================

def require_api_key():
    api_key = os.getenv("TWELVE_DATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing. "
            "Add it to GitHub repository secrets."
        )

    return api_key


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
#
# We request one month at a time so each request remains
# comfortably below Twelve Data's historical-data limits.
# ============================================================

def build_download_windows():
    month_starts = pd.date_range(
        start=DOWNLOAD_START,
        end="2026-09-01",
        freq="MS",
    )

    windows = []

    for start in month_starts:
        if start >= pd.Timestamp("2026-09-01"):
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
# TWELVE DATA DOWNLOAD
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

    response = requests.get(
        API_BASE,
        params=params,
        timeout=60,
    )

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

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["symbol"] = symbol

    df = df[
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

    return df


def load_cached_symbol(
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


def save_cached_symbol(
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

    if cache_file.exists():
        print(
            f"{symbol}: using cached data"
        )

        cached = load_cached_symbol(
            cache_file
        )

        cached_end = (
            cached["datetime"].max()
        )

        if (
            cached_end
            >= TEST_END
            - pd.Timedelta(days=1)
        ):
            return cached

        print(
            f"{symbol}: cache does not cover "
            f"the full expanded period. "
            f"Redownloading."
        )

    print()
    print(
        f"Downloading {symbol}..."
    )

    frames = []

    for (
        start_date,
        end_date,
    ) in build_download_windows():

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

        # Twelve Data free plan rate limiting.
        time.sleep(8)

    if not frames:
        raise RuntimeError(
            f"No data downloaded for {symbol}"
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

    save_cached_symbol(
        df,
        cache_file,
    )

    return df


def download_all_data():
    api_key = require_api_key()

    data = {}

    for symbol in SYMBOLS:
        data[symbol] = download_symbol(
            symbol,
            api_key,
        )

    return data


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df,
):
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

    true_range_1 = (
        df["high"]
        - df["low"]
    )

    true_range_2 = (
        df["high"]
        - previous_close
    ).abs()

    true_range_3 = (
        df["low"]
        - previous_close
    ).abs()

    df["true_range"] = (
        pd.concat(
            [
                true_range_1,
                true_range_2,
                true_range_3,
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
    # VWAP
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
        / df["cum_volume"].replace(
            0,
            np.nan,
        )
    )


    # --------------------------------------------------------
    # 15-MINUTE OPENING RANGE
    #
    # Bars:
    # 9:30
    # 9:35
    # 9:40
    # --------------------------------------------------------

    opening_mask = (
        (df["time"] >= "09:30")
        & (df["time"] < OPENING_RANGE_END)
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
    # Compare current 5-minute volume against the same
    # intraday time slot over previous sessions.
    # --------------------------------------------------------

    df["minute_slot"] = (
        df["time"]
    )

    rvol_average_series = []

    for (
        slot,
        slot_df,
    ) in df.groupby(
        "minute_slot",
        sort=False,
    ):

        indices = (
            slot_df.index
        )

        average_volume = (
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

        rvol_average_series.append(
            pd.Series(
                average_volume.values,
                index=indices,
            )
        )

    df["slot_avg_volume"] = (
        pd.concat(
            rvol_average_series
        )
        .sort_index()
    )

    df["rvol"] = (
        df["volume"]
        / df[
            "slot_avg_volume"
        ]
    )

    return df


# ============================================================
# MARKET CONFIRMATION
#
# Frozen definition:
#
# SPY:
#   close > VWAP
#   EMA9 > EMA21
#
# QQQ:
#   close > VWAP
#   EMA9 > EMA21
#
# BOTH must be bullish.
# ============================================================

def create_market_confirmation(
    data,
):
    spy = (
        data["SPY"][
            [
                "datetime",
                "close",
                "ema9",
                "ema21",
                "vwap",
            ]
        ]
        .copy()
    )

    qqq = (
        data["QQQ"][
            [
                "datetime",
                "close",
                "ema9",
                "ema21",
                "vwap",
            ]
        ]
        .copy()
    )

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

    market[
        "market_bullish"
    ] = (
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
# BLACK-SCHOLES OPTION MODEL
#
# IMPORTANT:
# This is still a synthetic options model.
# It does NOT represent actual historical option contracts,
# IV, bid/ask spreads, OI, or liquidity.
# ============================================================

def normal_cdf(
    x,
):
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
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
            0.0,
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

    price = (
        spot
        * normal_cdf(d1)
        - strike
        * math.exp(
            -rate
            * time_years
        )
        * normal_cdf(d2)
    )

    return max(
        price,
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
# VOLATILITY ESTIMATE
# ============================================================

def estimate_annualized_volatility(
    df,
    entry_idx,
):
    bars_per_session = 78

    lookback = (
        20
        * bars_per_session
    )

    start_idx = max(
        0,
        entry_idx - lookback,
    )

    history = (
        df.iloc[
            start_idx:entry_idx
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
# SELECT SYNTHETIC ~0.50 DELTA STRIKE
# ============================================================

def select_strike(
    spot,
    volatility,
):
    time_years = (
        OPTION_DTE
        / 365
    )

    width = max(
        spot * 0.10,
        5.0,
    )

    candidate_strikes = (
        np.linspace(
            spot - width,
            spot + width,
            161,
        )
    )

    best_candidate = None

    for strike in candidate_strikes:

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
            best_candidate is None
            or distance
            < best_candidate[0]
        ):
            best_candidate = (
                distance,
                strike,
                delta,
            )

    return (
        float(
            best_candidate[1]
        ),
        float(
            best_candidate[2]
        ),
    )


# ============================================================
# BASE BREAKOUT CANDIDATES
# ============================================================

def base_breakout_candidates(
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
        x["close"]
        .shift(1)
    )

    x["fresh_breakout"] = (
        (
            x["close"]
            > x[
                "opening_range_high"
            ]
        )
        & (
            x["previous_close"]
            <= x[
                "opening_range_high"
            ]
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
# V1.3 VARIANT DEFINITIONS
#
# A:
# breakout -> retest OR high -> close back above OR high
#
# B:
# A + bullish reclaim candle
#
# C:
# B + next candle breaks reclaim candle high
# ============================================================

def detect_variant_signal(
    variant,
    breakout_row,
    df,
):
    breakout_time = (
        breakout_row[
            "datetime"
        ]
    )

    matching_indices = (
        df.index[
            df["datetime"]
            == breakout_time
        ]
    )

    if len(
        matching_indices
    ) == 0:
        return None

    breakout_idx = int(
        matching_indices[0]
    )

    opening_range_high = float(
        breakout_row[
            "opening_range_high"
        ]
    )

    retest_floor = (
        opening_range_high
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


        # ----------------------------------------------------
        # RETEST
        #
        # Price trades back toward OR high but does not close
        # below the OR level, then closes above the level.
        # ----------------------------------------------------

        retest = (
            (
                float(
                    bar["low"]
                )
                <= opening_range_high
            )
            and (
                float(
                    bar["low"]
                )
                >= retest_floor
            )
            and (
                float(
                    bar["close"]
                )
                > opening_range_high
            )
        )

        if not retest:
            continue


        # ----------------------------------------------------
        # VARIANT A
        # ----------------------------------------------------

        if variant == "A":
            return {
                "signal_idx": i,
                "signal_time": (
                    bar["datetime"]
                ),
                "retest_idx": i,
            }


        # ----------------------------------------------------
        # BULLISH RECLAIM
        # ----------------------------------------------------

        bullish_reclaim = (
            float(
                bar["close"]
            )
            > float(
                bar["open"]
            )
        )


        # ----------------------------------------------------
        # VARIANT B
        # ----------------------------------------------------

        if variant == "B":

            if bullish_reclaim:
                return {
                    "signal_idx": i,
                    "signal_time": (
                        bar["datetime"]
                    ),
                    "retest_idx": i,
                }


        # ----------------------------------------------------
        # VARIANT C
        # ----------------------------------------------------

        if variant == "C":

            if not bullish_reclaim:
                continue

            continuation_idx = (
                i + 1
            )

            if (
                continuation_idx
                >= len(df)
            ):
                continue

            continuation = (
                df.iloc[
                    continuation_idx
                ]
            )

            if (
                continuation[
                    "datetime"
                ].date()
                != breakout_time.date()
            ):
                continue

            continuation_break = (
                float(
                    continuation["high"]
                )
                > float(
                    bar["high"]
                )
            )

            if continuation_break:
                return {
                    "signal_idx":
                        continuation_idx,

                    "signal_time":
                        continuation[
                            "datetime"
                        ],

                    "retest_idx":
                        i,
                }

    return None


# ============================================================
# BUILD VARIANT SIGNALS
# ============================================================

def build_variant_signals(
    variant,
    data,
    market,
):
    rows = []

    for symbol in SYMBOLS:

        df = data[
            symbol
        ]

        breakouts = (
            base_breakout_candidates(
                symbol,
                df,
                market,
            )
        )

        for (
            _,
            breakout,
        ) in breakouts.iterrows():

            detected = (
                detect_variant_signal(
                    variant,
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
                    "variant":
                        variant,

                    "symbol":
                        symbol,

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
    # If multiple symbols produce signals at the exact same
    # timestamp, select the highest-RVOL setup.
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
        signal_idx
        + 1
    )

    if (
        entry_idx
        >= len(df)
    ):
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


    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VOLATILITY MODEL
    # --------------------------------------------------------

    volatility = (
        estimate_annualized_volatility(
            df,
            entry_idx,
        )
    )

    if not np.isfinite(
        volatility
    ):
        return None


    # --------------------------------------------------------
    # SYNTHETIC OPTION CONTRACT
    # --------------------------------------------------------

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

    target_option_price = (
        modeled_entry
        * (
            1
            + OPTION_TARGET_RETURN
        )
    )


    # --------------------------------------------------------
    # MAX HOLD
    # --------------------------------------------------------

    maximum_bars = (
        MAX_HOLD_MINUTES
        // 5
    )

    final_idx = min(
        entry_idx
        + maximum_bars,
        len(df) - 1,
    )


    # --------------------------------------------------------
    # DEFAULT EXIT
    # --------------------------------------------------------

    exit_reason = "TIME"

    exit_time = None
    exit_stock = None
    modeled_exit = None


    # --------------------------------------------------------
    # BAR-BY-BAR SIMULATION
    # --------------------------------------------------------

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

        elapsed_minutes = (
            (
                bar["datetime"]
                - entry_time
            )
            .total_seconds()
            / 60
        )

        remaining_days = max(
            OPTION_DTE
            - (
                elapsed_minutes
                / 1440
            ),
            1 / 1440,
        )

        remaining_years = (
            remaining_days
            / 365
        )


        # ----------------------------------------------------
        # STOP CHECK FIRST
        #
        # Conservative same-bar ordering.
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
        # OPTION TARGET CHECK
        # ----------------------------------------------------

        high_option_price = (
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

        if (
            high_option_price
            >= target_option_price
        ):

            low_spot = float(
                bar["low"]
            )

            high_spot = float(
                bar["high"]
            )

            # Binary search for approximate underlying
            # price where option hits target.
            for _ in range(40):

                midpoint = (
                    low_spot
                    + high_spot
                ) / 2

                option_price = (
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

                if (
                    option_price
                    >= target_option_price
                ):
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

        final_bar = (
            df.iloc[
                final_idx
            ]
        )

        if (
            final_bar[
                "datetime"
            ].date()
            != entry_time.date()
        ):
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

            final_bar = (
                same_day.iloc[-1]
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

        elapsed_minutes = (
            (
                exit_time
                - entry_time
            )
            .total_seconds()
            / 60
        )

        remaining_days = max(
            OPTION_DTE
            - (
                elapsed_minutes
                / 1440
            ),
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


    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

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
        "variant":
            signal[
                "variant"
            ],

        "symbol":
            signal[
                "symbol"
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

        "option_delta":
            delta,

        "option_strike":
            strike,

        "estimated_volatility":
            volatility,

        "modeled_option_entry":
            modeled_entry,

        "modeled_option_exit":
            modeled_exit,

        "option_return_pct":
            option_return
            * 100,

        "underlying_return_pct":
            underlying_return
            * 100,

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
# PORTFOLIO SIMULATION
# ============================================================

def run_variant(
    variant,
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
            data[
                symbol
            ],
        )

        if trade is None:
            continue

        trades.append(
            trade
        )

        if (
            ONE_POSITION_AT_A_TIME
        ):
            next_available_time = (
                trade[
                    "exit_time"
                ]
            )

    return pd.DataFrame(
        trades
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def profit_factor(
    returns,
):
    gross_profit = (
        returns[
            returns > 0
        ]
        .sum()
    )

    gross_loss = -(
        returns[
            returns < 0
        ]
        .sum()
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

    equity_curve = (
        1 + returns
    ).cumprod()

    rolling_peak = (
        equity_curve
        .cummax()
    )

    drawdown = (
        equity_curve
        / rolling_peak
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


# ============================================================
# VARIANT SUMMARY
# ============================================================

def summary_for_variant(
    variant,
    trades,
):
    if trades.empty:
        return {
            "variant":
                variant,

            "trades":
                0,

            "win_rate_pct":
                np.nan,

            "avg_option_return_pct":
                np.nan,

            "median_option_return_pct":
                np.nan,

            "profit_factor":
                np.nan,

            "max_drawdown_pct":
                np.nan,

            "avg_underlying_return_pct":
                np.nan,

            "underlying_win_rate_pct":
                np.nan,

            "profitable_months":
                0,

            "months_tested":
                0,

            "target_hits":
                0,

            "stop_hits":
                0,

            "time_exits":
                0,

            "longest_losing_streak":
                0,
        }


    option_returns = (
        trades[
            "option_return_pct"
        ]
        / 100
    )

    underlying_returns = (
        trades[
            "underlying_return_pct"
        ]
        / 100
    )


    # --------------------------------------------------------
    # MONTHLY PERFORMANCE
    # --------------------------------------------------------

    monthly = (
        trades.copy()
    )

    monthly["month"] = (
        pd.to_datetime(
            monthly[
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

    monthly_means = (
        monthly
        .groupby(
            "month"
        )[
            "option_return_pct"
        ]
        .mean()
    )

    profitable_months = int(
        (
            monthly_means
            > 0
        )
        .sum()
    )


    return {
        "variant":
            variant,

        "trades":
            len(
                trades
            ),

        "win_rate_pct":
            option_returns
            .gt(0)
            .mean()
            * 100,

        "avg_option_return_pct":
            option_returns
            .mean()
            * 100,

        "median_option_return_pct":
            option_returns
            .median()
            * 100,

        "profit_factor":
            profit_factor(
                option_returns
            ),

        "max_drawdown_pct":
            max_drawdown(
                option_returns
            )
            * 100,

        "avg_underlying_return_pct":
            underlying_returns
            .mean()
            * 100,

        "underlying_win_rate_pct":
            underlying_returns
            .gt(0)
            .mean()
            * 100,

        "profitable_months":
            profitable_months,

        "months_tested":
            len(
                monthly_means
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

        "longest_losing_streak":
            longest_losing_streak(
                option_returns
            ),
    }


# ============================================================
# ROBUSTNESS RANKING
#
# This ranking is for development only.
# It is NOT part of the trading strategy.
# ============================================================

def robustness_score(
    row,
):
    trades = (
        row[
            "trades"
        ]
    )

    if trades < 5:
        return -999.0

    score = 0.0


    # More observations are preferable.
    score += (
        min(
            trades,
            40,
        )
        * 0.20
    )


    # Positive expectancy.
    score += (
        row[
            "avg_option_return_pct"
        ]
        * 0.50
    )


    # Profit factor.
    pf = (
        row[
            "profit_factor"
        ]
    )

    if np.isfinite(
        pf
    ):
        score += (
            min(
                pf,
                3.0,
            )
            * 3
        )


    # Win rate.
    score += (
        (
            row[
                "win_rate_pct"
            ]
            - 50
        )
        * 0.10
    )


    # Stability across months.
    score += (
        row[
            "profitable_months"
        ]
        * 2
    )


    # Drawdown penalty.
    #
    # Max DD is negative, so adding it reduces score.
    score += (
        row[
            "max_drawdown_pct"
        ]
        * 0.05
    )

    return float(
        score
    )


# ============================================================
# BY-MONTH OUTPUT
# ============================================================

def build_monthly_results(
    trades,
):
    if trades.empty:
        return pd.DataFrame()

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
                "variant":
                    group[
                        "variant"
                    ].iloc[0],

                "month":
                    month,

                "trades":
                    len(
                        group
                    ),

                "win_rate_pct":
                    returns
                    .gt(0)
                    .mean()
                    * 100,

                "avg_option_return_pct":
                    returns
                    .mean()
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
# BY-TICKER OUTPUT
# ============================================================

def build_ticker_results(
    trades,
):
    if trades.empty:
        return pd.DataFrame()

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

        underlying_returns = (
            group[
                "underlying_return_pct"
            ]
            / 100
        )

        rows.append(
            {
                "variant":
                    group[
                        "variant"
                    ].iloc[0],

                "symbol":
                    symbol,

                "trades":
                    len(
                        group
                    ),

                "win_rate_pct":
                    returns
                    .gt(0)
                    .mean()
                    * 100,

                "avg_option_return_pct":
                    returns
                    .mean()
                    * 100,

                "profit_factor":
                    profit_factor(
                        returns
                    ),

                "avg_underlying_return_pct":
                    underlying_returns
                    .mean()
                    * 100,
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
        "=" * 70
    )

    print(
        "V1.3 EXPANDED DEVELOPMENT TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Development period: "
        f"{TEST_START.date()} "
        f"through "
        f"{TEST_END.date()}"
    )

    print(
        "Variants: A, B, C"
    )

    print(
        "Strategy parameters remain unchanged."
    )

    print()


    # --------------------------------------------------------
    # LOAD HISTORICAL DATA
    # --------------------------------------------------------

    raw_data = (
        download_all_data()
    )

    data = {}


    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    for (
        symbol,
        df,
    ) in raw_data.items():

        print(
            f"Calculating indicators: "
            f"{symbol}"
        )

        data[
            symbol
        ] = add_indicators(
            df
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
    # RUN ALL THREE VARIANTS
    # --------------------------------------------------------

    summaries = []

    all_trades = []

    all_monthly = []

    all_ticker = []


    for variant in [
        "A",
        "B",
        "C",
    ]:

        print()
        print(
            "-" * 70
        )

        print(
            f"Running Variant {variant}"
        )

        print(
            "-" * 70
        )


        signals = (
            build_variant_signals(
                variant,
                data,
                market,
            )
        )

        print(
            f"Qualifying signals: "
            f"{len(signals)}"
        )


        # ----------------------------------------------------
        # SAVE SIGNALS
        # ----------------------------------------------------

        signals.to_csv(
            RESULTS_DIR
            / (
                f"v13_variant_"
                f"{variant}_signals.csv"
            ),
            index=False,
        )


        # ----------------------------------------------------
        # RUN TRADES
        # ----------------------------------------------------

        trades = (
            run_variant(
                variant,
                signals,
                data,
            )
        )

        print(
            f"Completed trades: "
            f"{len(trades)}"
        )


        # ----------------------------------------------------
        # SAVE TRADE LOG
        # ----------------------------------------------------

        trades.to_csv(
            RESULTS_DIR
            / (
                f"v13_variant_"
                f"{variant}_trades.csv"
            ),
            index=False,
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = (
            summary_for_variant(
                variant,
                trades,
            )
        )

        summaries.append(
            summary
        )


        if not trades.empty:

            all_trades.append(
                trades
            )

            monthly_results = (
                build_monthly_results(
                    trades
                )
            )

            ticker_results = (
                build_ticker_results(
                    trades
                )
            )

            all_monthly.append(
                monthly_results
            )

            all_ticker.append(
                ticker_results
            )


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    summary_df = (
        pd.DataFrame(
            summaries
        )
    )

    summary_df[
        "robustness_score"
    ] = (
        summary_df.apply(
            robustness_score,
            axis=1,
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            "robustness_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    summary_df.to_csv(
        RESULTS_DIR
        / "v13_comparison.csv",
        index=False,
    )


    # --------------------------------------------------------
    # SAVE ALL TRADES
    # --------------------------------------------------------

    if all_trades:

        combined_trades = (
            pd.concat(
                all_trades,
                ignore_index=True,
            )
        )

        combined_trades.to_csv(
            RESULTS_DIR
            / "v13_all_trades.csv",
            index=False,
        )


    # --------------------------------------------------------
    # SAVE MONTHLY RESULTS
    # --------------------------------------------------------

    if all_monthly:

        combined_monthly = (
            pd.concat(
                all_monthly,
                ignore_index=True,
            )
        )

        combined_monthly.to_csv(
            RESULTS_DIR
            / "v13_by_month.csv",
            index=False,
        )


    # --------------------------------------------------------
    # SAVE TICKER RESULTS
    # --------------------------------------------------------

    if all_ticker:

        combined_ticker = (
            pd.concat(
                all_ticker,
                ignore_index=True,
            )
        )

        combined_ticker.to_csv(
            RESULTS_DIR
            / "v13_by_ticker.csv",
            index=False,
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print()
    print(
        "=" * 70
    )

    print(
        "V1.3 DEVELOPMENT COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # ROBUSTNESS RANKING
    # --------------------------------------------------------

    print()
    print()
    print(
        "ROBUSTNESS RANKING"
    )

    print(
        "-" * 70
    )

    for (
        rank,
        row,
    ) in summary_df.iterrows():

        pf = (
            row.get(
                "profit_factor",
                np.nan,
            )
        )

        avg_return = (
            row.get(
                "avg_option_return_pct",
                np.nan,
            )
        )

        win_rate = (
            row.get(
                "win_rate_pct",
                np.nan,
            )
        )

        drawdown = (
            row.get(
                "max_drawdown_pct",
                np.nan,
            )
        )

        print(
            f"{rank + 1}. "
            f"Variant "
            f"{row['variant']} "
            f"| Trades="
            f"{int(row['trades'])} "
            f"| PF="
            f"{pf:.2f} "
            f"| Avg="
            f"{avg_return:.2f}% "
            f"| Win="
            f"{win_rate:.1f}% "
            f"| DD="
            f"{drawdown:.2f}%"
        )


    # --------------------------------------------------------
    # SAMPLE-SIZE WARNING
    # --------------------------------------------------------

    print()
    print(
        "SAMPLE SIZE CHECK"
    )

    print(
        "-" * 70
    )

    for (
        _,
        row,
    ) in summary_df.iterrows():

        trades = int(
            row[
                "trades"
            ]
        )

        variant = (
            row[
                "variant"
            ]
        )

        if trades >= 30:

            status = (
                "ENOUGH FOR FURTHER REVIEW"
            )

        elif trades >= 15:

            status = (
                "STILL SMALL"
            )

        else:

            status = (
                "TOO SMALL"
            )

        print(
            f"Variant {variant}: "
            f"{trades} trades "
            f"-> {status}"
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
