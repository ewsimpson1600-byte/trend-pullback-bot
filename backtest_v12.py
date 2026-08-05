import os
import math
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import requests


# ============================================================
# FROZEN V1.2 CONFIG
# DO NOT CHANGE THESE PARAMETERS UNTIL OOS TEST IS COMPLETE.
# ============================================================

SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD", "TSLA"]

# Warmup + untouched validation period
DOWNLOAD_START = "2025-12-01"
TEST_START = pd.Timestamp("2026-01-02", tz="America/New_York")
TEST_END = pd.Timestamp("2026-04-30 16:00", tz="America/New_York")

INTERVAL = "5min"

SIGNAL_START = "09:45"
SIGNAL_END = "11:30"

RVOL_MIN = 1.50

EMA_FAST = 9
EMA_SLOW = 21

# 15-minute opening range = 9:30, 9:35, 9:40 bars
OPENING_RANGE_END = "09:45"

ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 2.0

MAX_HOLD_MINUTES = 90

# Frozen modeled option
OPTION_DTE = 5
OPTION_TARGET_RETURN = 0.30
TARGET_DELTA = 0.50

# Conservative modeled round-trip friction:
# +1% worse on entry and -1% worse on exit.
ENTRY_FRICTION = 0.01
EXIT_FRICTION = 0.01

# Approximation used only for historical option repricing.
RISK_FREE_RATE = 0.04

# RVOL compares each 5-minute slot against previous sessions.
RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_HISTORY_DAYS = 10

# One active position at a time across the portfolio.
ONE_POSITION_AT_A_TIME = True

CACHE_DIR = Path("backtest_data_v12")
RESULTS_DIR = Path("backtest_results_v12")

API_BASE = "https://api.twelvedata.com/time_series"


# ============================================================
# BASIC HELPERS
# ============================================================

def require_api_key():
    key = os.getenv("TWELVE_DATA_API_KEY")

    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing.\n"
            "Add it to GitHub Actions secrets or your local environment."
        )

    return key


def ensure_directories():
    CACHE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def timestamp_et(date_string):
    return pd.Timestamp(date_string, tz="America/New_York")


# ============================================================
# TWELVE DATA DOWNLOAD
# ============================================================

def fetch_chunk(symbol, start_date, end_date, api_key):
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

    values = payload.get("values", [])

    if not values:
        print(
            f"WARNING: no data returned for "
            f"{symbol} {start_date} -> {end_date}"
        )
        return pd.DataFrame()

    df = pd.DataFrame(values)

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing columns for {symbol}: {missing}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(
            "America/New_York",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(
            "America/New_York"
        )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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


def build_download_windows():
    """
    Keep each request comfortably below Twelve Data's 5,000-row cap.

    Roughly 78 five-minute bars/day.
    ~20 trading days/month = ~1,560 rows.
    Monthly chunks are therefore safely under 5,000.
    """

    starts = pd.date_range(
        start=DOWNLOAD_START,
        end="2026-05-01",
        freq="MS",
    )

    windows = []

    for start in starts:
        next_month = start + pd.offsets.MonthBegin(1)

        if start >= pd.Timestamp("2026-05-01"):
            break

        windows.append(
            (
                start.strftime("%Y-%m-%d"),
                next_month.strftime("%Y-%m-%d"),
            )
        )

    return windows


def download_symbol(symbol, api_key):
    cache_file = CACHE_DIR / f"{symbol}_5min.csv"

    if cache_file.exists():
        print(f"{symbol}: using cached data")

        df = pd.read_csv(cache_file)

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True,
        ).dt.tz_convert("America/New_York")

        return df

    print(f"\nDownloading {symbol}...")

    frames = []

    for start_date, end_date in build_download_windows():
        print(f"  {start_date} -> {end_date}")

        chunk = fetch_chunk(
            symbol,
            start_date,
            end_date,
            api_key,
        )

        if not chunk.empty:
            frames.append(chunk)

        # Free plan = 8 credits/minute.
        # Seven symbols are eventually requested, but spacing calls
        # makes the script more robust to retries / API behavior.
        time.sleep(8)

    if not frames:
        raise RuntimeError(
            f"No historical data downloaded for {symbol}"
        )

    df = pd.concat(frames, ignore_index=True)

    df = (
        df.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Convert stored timestamps to UTC ISO to avoid timezone ambiguity.
    save_df = df.copy()

    save_df["datetime"] = (
        save_df["datetime"]
        .dt.tz_convert("UTC")
        .astype(str)
    )

    save_df.to_csv(cache_file, index=False)

    print(f"{symbol}: cached {len(df):,} bars")

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

def add_indicators(df):
    df = df.copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")

    df["ema9"] = (
        df["close"]
        .ewm(span=EMA_FAST, adjust=False)
        .mean()
    )

    df["ema21"] = (
        df["close"]
        .ewm(span=EMA_SLOW, adjust=False)
        .mean()
    )

    # ---------------------------
    # True Range / ATR
    # ---------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["true_range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    # Standard Wilder smoothing approximation.
    df["atr"] = (
        df["true_range"]
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
        )
        .mean()
    )

    # ---------------------------
    # Intraday VWAP
    # ---------------------------

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    df["tpv"] = typical_price * df["volume"]

    grouped = df.groupby("date", sort=False)

    df["cum_tpv"] = grouped["tpv"].cumsum()
    df["cum_volume"] = grouped["volume"].cumsum()

    df["vwap"] = (
        df["cum_tpv"]
        / df["cum_volume"].replace(0, np.nan)
    )

    # ---------------------------
    # Opening Range
    # ---------------------------

    opening_mask = (
        (df["time"] >= "09:30")
        & (df["time"] < OPENING_RANGE_END)
    )

    opening = (
        df.loc[opening_mask]
        .groupby("date")
        .agg(
            opening_range_high=("high", "max"),
            opening_range_low=("low", "min"),
        )
    )

    df = df.merge(
        opening,
        left_on="date",
        right_index=True,
        how="left",
    )

    # ---------------------------
    # RVOL by exact 5-min time slot
    # ---------------------------

    df["minute_slot"] = df["time"]

    daily_slot = df[
        [
            "date",
            "minute_slot",
            "volume",
        ]
    ].copy()

    # Previous sessions only.
    rvol_average = []

    for slot, slot_df in daily_slot.groupby(
        "minute_slot",
        sort=False,
    ):
        idx = slot_df.index

        avg = (
            slot_df["volume"]
            .shift(1)
            .rolling(
                RVOL_LOOKBACK_DAYS,
                min_periods=RVOL_MIN_HISTORY_DAYS,
            )
            .mean()
        )

        rvol_average.append(
            pd.Series(
                avg.values,
                index=idx,
            )
        )

    df["slot_avg_volume"] = pd.concat(
        rvol_average
    ).sort_index()

    df["rvol"] = (
        df["volume"]
        / df["slot_avg_volume"]
    )

    return df


# ============================================================
# MARKET CONFIRMATION
# ============================================================

def create_market_confirmation(data):
    """
    Frozen interpretation of:
        "SPY/QQQ bullish"

    Both must:
      - be above VWAP
      - have EMA9 > EMA21

    at the same completed 5-minute bar.
    """

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
        & (spy["ema9"] > spy["ema21"])
    )

    qqq["qqq_bullish"] = (
        (qqq["close"] > qqq["vwap"])
        & (qqq["ema9"] > qqq["ema21"])
    )

    market = spy[
        ["datetime", "spy_bullish"]
    ].merge(
        qqq[
            ["datetime", "qqq_bullish"]
        ],
        on="datetime",
        how="inner",
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
# SIGNAL ENGINE
# ============================================================

def build_signals(symbol, df, market):
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

    x["previous_close"] = x["close"].shift(1)

    # "Opening-range breakout" is interpreted as a fresh close
    # above the 15-minute opening-range high.
    x["fresh_or_breakout"] = (
        (x["close"] > x["opening_range_high"])
        & (
            x["previous_close"]
            <= x["opening_range_high"]
        )
    )

    time_filter = (
        (x["time"] >= SIGNAL_START)
        & (x["time"] <= SIGNAL_END)
    )

    test_filter = (
        (x["datetime"] >= TEST_START)
        & (x["datetime"] <= TEST_END)
    )

    signal = (
        test_filter
        & time_filter
        & (x["rvol"] >= RVOL_MIN)
        & (x["ema9"] > x["ema21"])
        & (x["close"] > x["vwap"])
        & x["fresh_or_breakout"]
        & x["market_bullish"]
        & x["atr"].notna()
    )

    x["signal"] = signal

    signal_rows = x.loc[signal].copy()

    signal_rows["symbol"] = symbol

    return signal_rows


def collect_all_signals(data, market):
    frames = []

    for symbol in SYMBOLS:
        signals = build_signals(
            symbol,
            data[symbol],
            market,
        )

        if not signals.empty:
            frames.append(signals)

    if not frames:
        return pd.DataFrame()

    all_signals = pd.concat(
        frames,
        ignore_index=True,
    )

    # If several setups occur at exactly the same time,
    # choose the highest-RVOL candidate.
    all_signals = (
        all_signals
        .sort_values(
            ["datetime", "rvol"],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return all_signals


# ============================================================
# BLACK-SCHOLES OPTION MODEL
# ============================================================

def normal_cdf(x):
    return 0.5 * (
        1.0 + math.erf(x / math.sqrt(2.0))
    )


def bs_call_price(
    spot,
    strike,
    time_years,
    rate,
    volatility,
):
    if time_years <= 0:
        return max(spot - strike, 0.0)

    volatility = max(volatility, 0.05)

    sqrt_t = math.sqrt(time_years)

    d1 = (
        math.log(spot / strike)
        + (
            rate
            + 0.5 * volatility**2
        ) * time_years
    ) / (
        volatility * sqrt_t
    )

    d2 = (
        d1
        - volatility * sqrt_t
    )

    call = (
        spot * normal_cdf(d1)
        - strike
        * math.exp(
            -rate * time_years
        )
        * normal_cdf(d2)
    )

    return max(call, 0.01)


def bs_call_delta(
    spot,
    strike,
    time_years,
    rate,
    volatility,
):
    if time_years <= 0:
        return 1.0 if spot > strike else 0.0

    volatility = max(volatility, 0.05)

    d1 = (
        math.log(spot / strike)
        + (
            rate
            + 0.5 * volatility**2
        ) * time_years
    ) / (
        volatility * math.sqrt(time_years)
    )

    return normal_cdf(d1)


def estimate_annualized_volatility(df, entry_idx):
    """
    Uses only information known BEFORE entry.
    5-minute log returns over approximately previous 20 sessions.
    """

    bars_per_session = 78
    lookback = 20 * bars_per_session

    start = max(
        0,
        entry_idx - lookback,
    )

    history = df.iloc[
        start:entry_idx
    ]["close"]

    log_returns = np.log(
        history / history.shift(1)
    ).dropna()

    if len(log_returns) < 100:
        return np.nan

    bars_per_year = 252 * bars_per_session

    annualized = (
        log_returns.std()
        * math.sqrt(bars_per_year)
    )

    # Keep pathological estimates from breaking the model.
    return float(
        np.clip(
            annualized,
            0.10,
            2.50,
        )
    )


def select_approximately_50_delta_strike(
    spot,
    volatility,
):
    """
    Search nearby synthetic strikes and choose the strike whose
    Black-Scholes call delta is closest to 0.50.

    This is a modeled approximation, not historical OCC-contract data.
    """

    time_years = OPTION_DTE / 365

    # Generate sensible nearby strikes.
    width = max(
        spot * 0.10,
        5.0,
    )

    candidates = np.linspace(
        spot - width,
        spot + width,
        161,
    )

    best_strike = None
    best_delta = None
    best_distance = float("inf")

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
            delta - TARGET_DELTA
        )

        if distance < best_distance:
            best_distance = distance
            best_strike = strike
            best_delta = delta

    return (
        float(best_strike),
        float(best_delta),
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def find_exact_index(df, timestamp):
    matches = df.index[
        df["datetime"] == timestamp
    ]

    if len(matches) == 0:
        return None

    return int(matches[0])


def simulate_trade(signal, df):
    signal_time = signal["datetime"]

    signal_idx = find_exact_index(
        df,
        signal_time,
    )

    if signal_idx is None:
        return None

    # Entry NEXT 5-minute candle.
    entry_idx = signal_idx + 1

    if entry_idx >= len(df):
        return None

    entry_bar = df.iloc[entry_idx]

    # Must be same session.
    if (
        entry_bar["datetime"].date()
        != signal_time.date()
    ):
        return None

    entry_time = entry_bar["datetime"]
    entry_stock = float(
        entry_bar["open"]
    )

    atr_at_signal = float(
        signal["atr"]
    )

    underlying_stop = (
        entry_stock
        - STOP_ATR_MULTIPLE
        * atr_at_signal
    )

    volatility = estimate_annualized_volatility(
        df,
        entry_idx,
    )

    if not np.isfinite(volatility):
        return None

    strike, modeled_delta = (
        select_approximately_50_delta_strike(
            entry_stock,
            volatility,
        )
    )

    initial_time_years = OPTION_DTE / 365

    theoretical_entry = bs_call_price(
        entry_stock,
        strike,
        initial_time_years,
        RISK_FREE_RATE,
        volatility,
    )

    # Worse fill on purchase.
    modeled_entry = (
        theoretical_entry
        * (1 + ENTRY_FRICTION)
    )

    target_option_price = (
        modeled_entry
        * (1 + OPTION_TARGET_RETURN)
    )

    exit_idx_limit = min(
        entry_idx
        + MAX_HOLD_MINUTES // 5,
        len(df) - 1,
    )

    exit_reason = "TIME"
    exit_time = None
    exit_stock = None
    modeled_exit = None
    theoretical_exit = None

    max_stock = entry_stock
    min_stock = entry_stock

    max_option_return = -np.inf
    min_option_return = np.inf

    for i in range(
        entry_idx,
        exit_idx_limit + 1,
    ):
        bar = df.iloc[i]

        if (
            bar["datetime"].date()
            != entry_time.date()
        ):
            break

        elapsed_minutes = (
            bar["datetime"]
            - entry_time
        ).total_seconds() / 60

        remaining_days = max(
            OPTION_DTE
            - elapsed_minutes
            / (60 * 24),
            1 / (24 * 60),
        )

        remaining_years = (
            remaining_days / 365
        )

        max_stock = max(
            max_stock,
            float(bar["high"]),
        )

        min_stock = min(
            min_stock,
            float(bar["low"]),
        )

        # -----------------------
        # Underlying stop first.
        # Conservative convention:
        # if a single 5-min bar could hit both stop and target,
        # assume stop happens first.
        # -----------------------

        if float(bar["low"]) <= underlying_stop:
            stop_spot = underlying_stop

            theoretical_exit = bs_call_price(
                stop_spot,
                strike,
                remaining_years,
                RISK_FREE_RATE,
                volatility,
            )

            modeled_exit = (
                theoretical_exit
                * (1 - EXIT_FRICTION)
            )

            exit_reason = "UNDERLYING_STOP"
            exit_time = bar["datetime"]
            exit_stock = stop_spot

            break

        # -----------------------
        # Premium target test
        # -----------------------

        theoretical_high_option = bs_call_price(
            float(bar["high"]),
            strike,
            remaining_years,
            RISK_FREE_RATE,
            volatility,
        )

        executable_high_option = (
            theoretical_high_option
            * (1 - EXIT_FRICTION)
        )

        option_high_return = (
            executable_high_option
            / modeled_entry
            - 1
        )

        max_option_return = max(
            max_option_return,
            option_high_return,
        )

        theoretical_low_option = bs_call_price(
            float(bar["low"]),
            strike,
            remaining_years,
            RISK_FREE_RATE,
            volatility,
        )

        executable_low_option = (
            theoretical_low_option
            * (1 - EXIT_FRICTION)
        )

        option_low_return = (
            executable_low_option
            / modeled_entry
            - 1
        )

        min_option_return = min(
            min_option_return,
            option_low_return,
        )

        if executable_high_option >= target_option_price:
            # Solve approximately for the underlying spot where
            # the option would equal target.
            low_spot = float(bar["low"])
            high_spot = float(bar["high"])

            for _ in range(40):
                midpoint = (
                    low_spot + high_spot
                ) / 2

                candidate = (
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

                if candidate >= target_option_price:
                    high_spot = midpoint
                else:
                    low_spot = midpoint

            exit_stock = high_spot

            theoretical_exit = bs_call_price(
                exit_stock,
                strike,
                remaining_years,
                RISK_FREE_RATE,
                volatility,
            )

            modeled_exit = (
                theoretical_exit
                * (1 - EXIT_FRICTION)
            )

            exit_reason = "OPTION_TARGET"
            exit_time = bar["datetime"]

            break

    # ---------------------------
    # Time exit
    # ---------------------------

    if exit_time is None:
        final_idx = min(
            exit_idx_limit,
            len(df) - 1,
        )

        while (
            final_idx > entry_idx
            and df.iloc[final_idx]["datetime"].date()
            != entry_time.date()
        ):
            final_idx -= 1

        final_bar = df.iloc[
            final_idx
        ]

        exit_time = final_bar["datetime"]
        exit_stock = float(
            final_bar["close"]
        )

        elapsed_minutes = (
            exit_time
            - entry_time
        ).total_seconds() / 60

        remaining_days = max(
            OPTION_DTE
            - elapsed_minutes
            / (60 * 24),
            1 / (24 * 60),
        )

        theoretical_exit = bs_call_price(
            exit_stock,
            strike,
            remaining_days / 365,
            RISK_FREE_RATE,
            volatility,
        )

        modeled_exit = (
            theoretical_exit
            * (1 - EXIT_FRICTION)
        )

    option_return = (
        modeled_exit
        / modeled_entry
        - 1
    )

    underlying_return = (
        exit_stock / entry_stock
        - 1
    )

    max_underlying_return = (
        max_stock / entry_stock
        - 1
    )

    max_underlying_drawdown = (
        min_stock / entry_stock
        - 1
    )

    return {
        "symbol": signal["symbol"],
        "signal_time": signal_time,
        "entry_time": entry_time,
        "exit_time": exit_time,

        "rvol": float(signal["rvol"]),

        "opening_range_high": float(
            signal["opening_range_high"]
        ),

        "signal_close": float(
            signal["close"]
        ),

        "entry_stock": entry_stock,
        "underlying_stop": underlying_stop,
        "exit_stock": exit_stock,

        "atr": atr_at_signal,

        "underlying_return_pct":
            underlying_return * 100,

        "max_underlying_return_pct":
            max_underlying_return * 100,

        "max_underlying_drawdown_pct":
            max_underlying_drawdown * 100,

        "modeled_iv":
            volatility,

        "modeled_strike":
            strike,

        "modeled_delta":
            modeled_delta,

        "theoretical_option_entry":
            theoretical_entry,

        "modeled_option_entry":
            modeled_entry,

        "modeled_option_exit":
            modeled_exit,

        "option_return_pct":
            option_return * 100,

        "max_modeled_option_return_pct":
            (
                max_option_return * 100
                if np.isfinite(
                    max_option_return
                )
                else np.nan
            ),

        "max_modeled_option_drawdown_pct":
            (
                min_option_return * 100
                if np.isfinite(
                    min_option_return
                )
                else np.nan
            ),

        "exit_reason":
            exit_reason,

        "hold_minutes":
            (
                exit_time
                - entry_time
            ).total_seconds() / 60,
    }


def run_portfolio(signals, data):
    trades = []

    next_available_time = None

    for _, signal in signals.sort_values(
        "datetime"
    ).iterrows():

        signal_time = signal["datetime"]

        if (
            ONE_POSITION_AT_A_TIME
            and next_available_time is not None
            and signal_time < next_available_time
        ):
            continue

        symbol = signal["symbol"]

        trade = simulate_trade(
            signal,
            data[symbol],
        )

        if trade is None:
            continue

        trades.append(trade)

        if ONE_POSITION_AT_A_TIME:
            next_available_time = (
                trade["exit_time"]
            )

    return pd.DataFrame(trades)


# ============================================================
# STATISTICS
# ============================================================

def profit_factor(returns):
    wins = returns[
        returns > 0
    ].sum()

    losses = -returns[
        returns < 0
    ].sum()

    if losses == 0:
        return np.inf

    return wins / losses


def max_drawdown_from_returns(returns):
    equity = (
        1 + returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return drawdown.min()


def longest_losing_streak(returns):
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


def bootstrap_mean_ci(
    returns,
    iterations=10000,
    seed=42,
):
    if len(returns) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    values = np.asarray(
        returns,
        dtype=float,
    )

    samples = rng.choice(
        values,
        size=(
            iterations,
            len(values),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        np.percentile(
            means,
            2.5,
        ),
        np.percentile(
            means,
            97.5,
        ),
    )


def build_summary(trades):
    if trades.empty:
        return {
            "trades": 0,
            "verdict": "INCONCLUSIVE",
        }

    returns = (
        trades["option_return_pct"]
        / 100
    )

    underlying_returns = (
        trades["underlying_return_pct"]
        / 100
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            returns
        )
    )

    pf = profit_factor(
        returns
    )

    underlying_pf = profit_factor(
        underlying_returns
    )

    summary = {
        "trades":
            len(trades),

        "wins":
            int(
                (returns > 0).sum()
            ),

        "losses":
            int(
                (returns < 0).sum()
            ),

        "win_rate_pct":
            (
                returns.gt(0).mean()
                * 100
            ),

        "avg_option_return_pct":
            returns.mean() * 100,

        "median_option_return_pct":
            returns.median() * 100,

        "profit_factor":
            pf,

        "max_drawdown_pct":
            (
                max_drawdown_from_returns(
                    returns
                )
                * 100
            ),

        "longest_losing_streak":
            longest_losing_streak(
                returns
            ),

        "avg_underlying_return_pct":
            underlying_returns.mean()
            * 100,

        "underlying_win_rate_pct":
            (
                underlying_returns.gt(0)
                .mean()
                * 100
            ),

        "underlying_profit_factor":
            underlying_pf,

        "bootstrap_mean_95_ci_low_pct":
            ci_low * 100,

        "bootstrap_mean_95_ci_high_pct":
            ci_high * 100,

        "target_hits":
            int(
                (
                    trades["exit_reason"]
                    == "OPTION_TARGET"
                ).sum()
            ),

        "underlying_stop_hits":
            int(
                (
                    trades["exit_reason"]
                    == "UNDERLYING_STOP"
                ).sum()
            ),

        "time_exits":
            int(
                (
                    trades["exit_reason"]
                    == "TIME"
                ).sum()
            ),
    }

    # ---------------------------
    # Frozen pass criteria
    # ---------------------------

    pass_conditions = [
        summary["trades"] >= 30,
        summary["win_rate_pct"] > 50,
        summary["profit_factor"] >= 1.30,
        summary["avg_option_return_pct"] > 0,
        summary["avg_underlying_return_pct"] > 0,
    ]

    if (
        summary["trades"] < 20
    ):
        verdict = "INCONCLUSIVE"

    elif all(pass_conditions):
        verdict = "PASS"

    else:
        verdict = "FAIL"

    summary["verdict"] = verdict

    return summary


def ticker_breakdown(trades):
    if trades.empty:
        return pd.DataFrame()

    rows = []

    for symbol, group in trades.groupby(
        "symbol"
    ):
        returns = (
            group["option_return_pct"]
            / 100
        )

        rows.append(
            {
                "symbol":
                    symbol,

                "trades":
                    len(group),

                "win_rate_pct":
                    returns.gt(0).mean()
                    * 100,

                "avg_option_return_pct":
                    returns.mean()
                    * 100,

                "profit_factor":
                    profit_factor(
                        returns
                    ),

                "avg_underlying_return_pct":
                    group[
                        "underlying_return_pct"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


def monthly_breakdown(trades):
    if trades.empty:
        return pd.DataFrame()

    x = trades.copy()

    x["month"] = (
        pd.to_datetime(
            x["entry_time"]
        )
        .dt.strftime("%Y-%m")
    )

    rows = []

    for month, group in x.groupby(
        "month"
    ):
        returns = (
            group["option_return_pct"]
            / 100
        )

        rows.append(
            {
                "month":
                    month,

                "trades":
                    len(group),

                "win_rate_pct":
                    returns.gt(0).mean()
                    * 100,

                "avg_option_return_pct":
                    returns.mean()
                    * 100,

                "profit_factor":
                    profit_factor(
                        returns
                    ),

                "avg_underlying_return_pct":
                    group[
                        "underlying_return_pct"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# OUTPUT
# ============================================================

def print_summary(summary):
    print("\n")
    print("=" * 60)
    print("V1.2 UNTOUCHED VALIDATION")
    print("=" * 60)

    print(
        f"Trades: "
        f"{summary.get('trades', 0)}"
    )

    if summary.get(
        "trades",
        0,
    ) == 0:
        print(
            "VERDICT: INCONCLUSIVE"
        )
        return

    print(
        f"Wins: "
        f"{summary['wins']}"
    )

    print(
        f"Losses: "
        f"{summary['losses']}"
    )

    print(
        f"Win rate: "
        f"{summary['win_rate_pct']:.2f}%"
    )

    print(
        f"Average modeled option return: "
        f"{summary['avg_option_return_pct']:.2f}%"
    )

    print(
        f"Median modeled option return: "
        f"{summary['median_option_return_pct']:.2f}%"
    )

    print(
        f"Profit factor: "
        f"{summary['profit_factor']:.2f}"
    )

    print(
        f"Maximum compounded drawdown: "
        f"{summary['max_drawdown_pct']:.2f}%"
    )

    print(
        f"Longest losing streak: "
        f"{summary['longest_losing_streak']}"
    )

    print()
    print(
        f"Average underlying return: "
        f"{summary['avg_underlying_return_pct']:.3f}%"
    )

    print(
        f"Underlying win rate: "
        f"{summary['underlying_win_rate_pct']:.2f}%"
    )

    print(
        f"Underlying profit factor: "
        f"{summary['underlying_profit_factor']:.2f}"
    )

    print()
    print(
        "95% bootstrap CI for mean modeled option return: "
        f"{summary['bootstrap_mean_95_ci_low_pct']:.2f}% "
        "to "
        f"{summary['bootstrap_mean_95_ci_high_pct']:.2f}%"
    )

    print()
    print(
        f"Targets hit: "
        f"{summary['target_hits']}"
    )

    print(
        f"Underlying stops hit: "
        f"{summary['underlying_stop_hits']}"
    )

    print(
        f"Time exits: "
        f"{summary['time_exits']}"
    )

    print()
    print("=" * 60)
    print(
        f"VERDICT: "
        f"{summary['verdict']}"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_directories()

    print(
        "Starting frozen V1.2 validation..."
    )

    print(
        "NO optimization will be performed."
    )

    raw_data = download_all_data()

    data = {}

    for symbol, df in raw_data.items():
        print(
            f"Calculating indicators: "
            f"{symbol}"
        )

        data[symbol] = add_indicators(
            df
        )

    market = create_market_confirmation(
        data
    )

    signals = collect_all_signals(
        data,
        market,
    )

    print(
        f"\nRaw V1.2 qualifying signals: "
        f"{len(signals)}"
    )

    if not signals.empty:
        signal_export = signals[
            [
                "symbol",
                "datetime",
                "close",
                "rvol",
                "ema9",
                "ema21",
                "vwap",
                "opening_range_high",
                "atr",
            ]
        ].copy()

        signal_export.to_csv(
            RESULTS_DIR
            / "v12_signals.csv",
            index=False,
        )

    trades = run_portfolio(
        signals,
        data,
    )

    trades.to_csv(
        RESULTS_DIR
        / "v12_trades.csv",
        index=False,
    )

    summary = build_summary(
        trades
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "v12_summary.csv",
        index=False,
    )

    by_ticker = ticker_breakdown(
        trades
    )

    by_ticker.to_csv(
        RESULTS_DIR
        / "v12_by_ticker.csv",
        index=False,
    )

    by_month = monthly_breakdown(
        trades
    )

    by_month.to_csv(
        RESULTS_DIR
        / "v12_by_month.csv",
        index=False,
    )

    print_summary(
        summary
    )

    if not by_ticker.empty:
        print("\nBY TICKER")
        print(
            by_ticker.to_string(
                index=False
            )
        )

    if not by_month.empty:
        print("\nBY MONTH")
        print(
            by_month.to_string(
                index=False
            )
        )

    print(
        "\nFiles written to:"
    )

    print(
        RESULTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()
