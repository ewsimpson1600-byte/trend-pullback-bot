import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# V1.3 DEVELOPMENT CONFIG
# Jan-Apr 2026 is DEVELOPMENT DATA now.
# 2025 remains untouched for later validation.
# ============================================================

SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD", "TSLA"]

DOWNLOAD_START = "2025-12-01"
TEST_START = pd.Timestamp("2026-01-02", tz="America/New_York")
TEST_END = pd.Timestamp("2026-04-30 16:00", tz="America/New_York")

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

# Retest definition:
# price must come within 0.15% of opening-range high
# while not closing materially below it.
RETEST_TOLERANCE = 0.0015

# Maximum bars after breakout to find retest.
MAX_RETEST_BARS = 6

CACHE_DIR = Path("backtest_data_v12")
RESULTS_DIR = Path("backtest_results_v13")

API_BASE = "https://api.twelvedata.com/time_series"


# ============================================================
# HELPERS
# ============================================================

def require_api_key():
    key = os.getenv("TWELVE_DATA_API_KEY")

    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing."
        )

    return key


def ensure_directories():
    CACHE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA DOWNLOAD
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
        return pd.DataFrame()

    df = pd.DataFrame(values)

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

        time.sleep(8)

    if not frames:
        raise RuntimeError(
            f"No data downloaded for {symbol}"
        )

    df = pd.concat(frames, ignore_index=True)

    df = (
        df.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    save_df = df.copy()

    save_df["datetime"] = (
        save_df["datetime"]
        .dt.tz_convert("UTC")
        .astype(str)
    )

    save_df.to_csv(cache_file, index=False)

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

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["true_range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr"] = (
        df["true_range"]
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
        )
        .mean()
    )

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

    df["minute_slot"] = df["time"]

    rvol_average = []

    for slot, slot_df in df.groupby(
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
# BLACK-SCHOLES MODEL
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

    d2 = d1 - volatility * sqrt_t

    return max(
        spot * normal_cdf(d1)
        - strike
        * math.exp(-rate * time_years)
        * normal_cdf(d2),
        0.01,
    )


def bs_call_delta(
    spot,
    strike,
    time_years,
    rate,
    volatility,
):
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
    bars_per_session = 78
    lookback = 20 * bars_per_session

    start = max(0, entry_idx - lookback)

    history = df.iloc[
        start:entry_idx
    ]["close"]

    returns = np.log(
        history / history.shift(1)
    ).dropna()

    if len(returns) < 100:
        return np.nan

    annualized = (
        returns.std()
        * math.sqrt(252 * bars_per_session)
    )

    return float(
        np.clip(
            annualized,
            0.10,
            2.50,
        )
    )


def select_strike(spot, volatility):
    time_years = OPTION_DTE / 365

    width = max(
        spot * 0.10,
        5.0,
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

        distance = abs(delta - TARGET_DELTA)

        if best is None or distance < best[0]:
            best = (
                distance,
                strike,
                delta,
            )

    return float(best[1]), float(best[2])


# ============================================================
# V1.3 VARIANT SIGNAL LOGIC
# ============================================================

def base_breakout_candidates(symbol, df, market):
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

    x["fresh_breakout"] = (
        (x["close"] > x["opening_range_high"])
        & (
            x["previous_close"]
            <= x["opening_range_high"]
        )
    )

    valid = (
        (x["datetime"] >= TEST_START)
        & (x["datetime"] <= TEST_END)
        & (x["time"] >= SIGNAL_START)
        & (x["time"] <= SIGNAL_END)
        & (x["rvol"] >= RVOL_MIN)
        & (x["ema9"] > x["ema21"])
        & (x["close"] > x["vwap"])
        & x["fresh_breakout"]
        & x["market_bullish"]
        & x["atr"].notna()
    )

    out = x.loc[valid].copy()
    out["symbol"] = symbol

    return out


def detect_variant_signal(
    variant,
    breakout_row,
    df,
):
    breakout_time = breakout_row["datetime"]

    matches = df.index[
        df["datetime"] == breakout_time
    ]

    if len(matches) == 0:
        return None

    breakout_idx = int(matches[0])

    or_high = float(
        breakout_row["opening_range_high"]
    )

    retest_floor = (
        or_high
        * (1 - RETEST_TOLERANCE)
    )

    last_idx = min(
        breakout_idx + MAX_RETEST_BARS,
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

        # Retest/reclaim:
        # bar low reaches OR-high area
        # and bar closes back above OR high.
        retest = (
            float(bar["low"]) <= or_high
            and float(bar["low"]) >= retest_floor
            and float(bar["close"]) > or_high
        )

        if not retest:
            continue

        if variant == "A":
            return {
                "signal_idx": i,
                "signal_time": bar["datetime"],
                "retest_idx": i,
            }

        bullish_reclaim = (
            float(bar["close"])
            > float(bar["open"])
        )

        if variant == "B":
            if bullish_reclaim:
                return {
                    "signal_idx": i,
                    "signal_time": bar["datetime"],
                    "retest_idx": i,
                }

        if variant == "C":
            if not bullish_reclaim:
                continue

            continuation_idx = i + 1

            if continuation_idx >= len(df):
                continue

            continuation = df.iloc[
                continuation_idx
            ]

            if (
                continuation["datetime"].date()
                != breakout_time.date()
            ):
                continue

            if (
                float(continuation["high"])
                > float(bar["high"])
            ):
                return {
                    "signal_idx": continuation_idx,
                    "signal_time": continuation["datetime"],
                    "retest_idx": i,
                }

    return None


def build_variant_signals(
    variant,
    data,
    market,
):
    rows = []

    for symbol in SYMBOLS:
        df = data[symbol]

        breakouts = base_breakout_candidates(
            symbol,
            df,
            market,
        )

        for _, breakout in breakouts.iterrows():
            detected = detect_variant_signal(
                variant,
                breakout,
                df,
            )

            if detected is None:
                continue

            signal_idx = detected[
                "signal_idx"
            ]

            signal_bar = df.iloc[
                signal_idx
            ]

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
                        breakout["datetime"],

                    "signal_time":
                        signal_bar["datetime"],

                    "signal_idx":
                        signal_idx,

                    "rvol":
                        float(breakout["rvol"]),

                    "opening_range_high":
                        float(
                            breakout[
                                "opening_range_high"
                            ]
                        ),

                    "atr":
                        float(
                            signal_bar["atr"]
                        ),

                    "signal_close":
                        float(
                            signal_bar["close"]
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    signals = pd.DataFrame(rows)

    # At identical signal timestamps choose highest RVOL.
    signals = (
        signals.sort_values(
            ["signal_time", "rvol"],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["signal_time"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return signals


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(signal, df):
    signal_idx = int(
        signal["signal_idx"]
    )

    entry_idx = signal_idx + 1

    if entry_idx >= len(df):
        return None

    entry_bar = df.iloc[
        entry_idx
    ]

    signal_time = signal[
        "signal_time"
    ]

    if (
        entry_bar["datetime"].date()
        != signal_time.date()
    ):
        return None

    entry_time = entry_bar[
        "datetime"
    ]

    entry_stock = float(
        entry_bar["open"]
    )

    atr = float(
        signal["atr"]
    )

    stop_stock = (
        entry_stock
        - STOP_ATR_MULTIPLE
        * atr
    )

    vol = estimate_annualized_volatility(
        df,
        entry_idx,
    )

    if not np.isfinite(vol):
        return None

    strike, delta = select_strike(
        entry_stock,
        vol,
    )

    initial_option = bs_call_price(
        entry_stock,
        strike,
        OPTION_DTE / 365,
        RISK_FREE_RATE,
        vol,
    )

    modeled_entry = (
        initial_option
        * (1 + ENTRY_FRICTION)
    )

    target_price = (
        modeled_entry
        * (1 + OPTION_TARGET_RETURN)
    )

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
            bar["datetime"].date()
            != entry_time.date()
        ):
            break

        elapsed = (
            bar["datetime"]
            - entry_time
        ).total_seconds() / 60

        remaining_days = max(
            OPTION_DTE
            - elapsed / 1440,
            1 / 1440,
        )

        remaining_years = (
            remaining_days / 365
        )

        # Conservative: stop checked before target.
        if float(bar["low"]) <= stop_stock:
            exit_stock = stop_stock

            modeled_exit = (
                bs_call_price(
                    exit_stock,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    vol,
                )
                * (1 - EXIT_FRICTION)
            )

            exit_reason = "UNDERLYING_STOP"
            exit_time = bar["datetime"]

            break

        high_option = (
            bs_call_price(
                float(bar["high"]),
                strike,
                remaining_years,
                RISK_FREE_RATE,
                vol,
            )
            * (1 - EXIT_FRICTION)
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
                    low_spot + high_spot
                ) / 2

                price = (
                    bs_call_price(
                        midpoint,
                        strike,
                        remaining_years,
                        RISK_FREE_RATE,
                        vol,
                    )
                    * (1 - EXIT_FRICTION)
                )

                if price >= target_price:
                    high_spot = midpoint
                else:
                    low_spot = midpoint

            exit_stock = high_spot

            modeled_exit = (
                bs_call_price(
                    exit_stock,
                    strike,
                    remaining_years,
                    RISK_FREE_RATE,
                    vol,
                )
                * (1 - EXIT_FRICTION)
            )

            exit_reason = "OPTION_TARGET"
            exit_time = bar["datetime"]

            break

    if exit_time is None:
        final_bar = df.iloc[
            final_idx
        ]

        exit_time = final_bar[
            "datetime"
        ]

        exit_stock = float(
            final_bar["close"]
        )

        elapsed = (
            exit_time
            - entry_time
        ).total_seconds() / 60

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
                vol,
            )
            * (1 - EXIT_FRICTION)
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
        "variant":
            signal["variant"],

        "symbol":
            signal["symbol"],

        "breakout_time":
            signal["breakout_time"],

        "signal_time":
            signal["signal_time"],

        "entry_time":
            entry_time,

        "exit_time":
            exit_time,

        "rvol":
            signal["rvol"],

        "entry_stock":
            entry_stock,

        "stop_stock":
            stop_stock,

        "exit_stock":
            exit_stock,

        "option_return_pct":
            option_return * 100,

        "underlying_return_pct":
            underlying_return * 100,

        "exit_reason":
            exit_reason,

        "hold_minutes":
            (
                exit_time
                - entry_time
            ).total_seconds() / 60,
    }


def run_variant(
    variant,
    signals,
    data,
):
    trades = []

    next_available_time = None

    for _, signal in signals.sort_values(
        "signal_time"
    ).iterrows():

        if (
            ONE_POSITION_AT_A_TIME
            and next_available_time is not None
            and signal["signal_time"]
            < next_available_time
        ):
            continue

        symbol = signal[
            "symbol"
        ]

        trade = simulate_trade(
            signal,
            data[symbol],
        )

        if trade is None:
            continue

        trades.append(trade)

        if ONE_POSITION_AT_A_TIME:
            next_available_time = trade[
                "exit_time"
            ]

    return pd.DataFrame(trades)


# ============================================================
# METRICS
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


def max_drawdown(returns):
    equity = (
        1 + returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return drawdown.min()


def summary_for_variant(
    variant,
    trades,
):
    if trades.empty:
        return {
            "variant": variant,
            "trades": 0,
        }

    returns = (
        trades["option_return_pct"]
        / 100
    )

    underlying = (
        trades["underlying_return_pct"]
        / 100
    )

    monthly = trades.copy()

    monthly["month"] = pd.to_datetime(
        monthly["entry_time"]
    ).dt.strftime("%Y-%m")

    monthly_means = (
        monthly
        .groupby("month")[
            "option_return_pct"
        ]
        .mean()
    )

    profitable_months = int(
        (monthly_means > 0).sum()
    )

    return {
        "variant":
            variant,

        "trades":
            len(trades),

        "win_rate_pct":
            returns.gt(0).mean()
            * 100,

        "avg_option_return_pct":
            returns.mean()
            * 100,

        "median_option_return_pct":
            returns.median()
            * 100,

        "profit_factor":
            profit_factor(
                returns
            ),

        "max_drawdown_pct":
            max_drawdown(
                returns
            )
            * 100,

        "avg_underlying_return_pct":
            underlying.mean()
            * 100,

        "underlying_win_rate_pct":
            underlying.gt(0).mean()
            * 100,

        "profitable_months":
            profitable_months,

        "months_tested":
            len(monthly_means),

        "target_hits":
            int(
                (
                    trades["exit_reason"]
                    == "OPTION_TARGET"
                ).sum()
            ),

        "stop_hits":
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


def robustness_score(row):
    """
    Ranking helper only.
    Not a trading rule.
    Favors:
      enough trades
      positive expectancy
      PF
      lower DD
      month stability
    """

    if row["trades"] < 5:
        return -999

    score = 0

    score += min(
        row["trades"],
        30,
    ) * 0.2

    score += (
        row["avg_option_return_pct"]
        * 0.5
    )

    pf = row["profit_factor"]

    if np.isfinite(pf):
        score += min(
            pf,
            3.0,
        ) * 3

    score += (
        row["win_rate_pct"]
        - 50
    ) * 0.1

    score += (
        row["profitable_months"]
        * 2
    )

    score += (
        row["max_drawdown_pct"]
        * 0.05
    )

    return score


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_directories()

    print(
        "Starting V1.3 development comparison..."
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

    summaries = []
    all_trades = []

    for variant in ["A", "B", "C"]:
        print(
            f"\nRunning Variant {variant}..."
        )

        signals = build_variant_signals(
            variant,
            data,
            market,
        )

        print(
            f"Qualifying signals: "
            f"{len(signals)}"
        )

        signals.to_csv(
            RESULTS_DIR
            / f"v13_variant_{variant}_signals.csv",
            index=False,
        )

        trades = run_variant(
            variant,
            signals,
            data,
        )

        trades.to_csv(
            RESULTS_DIR
            / f"v13_variant_{variant}_trades.csv",
            index=False,
        )

        summary = summary_for_variant(
            variant,
            trades,
        )

        summaries.append(
            summary
        )

        if not trades.empty:
            all_trades.append(
                trades
            )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df["robustness_score"] = (
        summary_df.apply(
            robustness_score,
            axis=1,
        )
    )

    summary_df = (
        summary_df.sort_values(
            "robustness_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "v13_comparison.csv",
        index=False,
    )

    if all_trades:
        pd.concat(
            all_trades,
            ignore_index=True,
        ).to_csv(
            RESULTS_DIR
            / "v13_all_trades.csv",
            index=False,
        )

    print("\n")
    print("=" * 70)
    print("V1.3 DEVELOPMENT COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n")
    print("ROBUSTNESS RANKING")

    for rank, row in summary_df.iterrows():
        print(
            f"{rank + 1}. "
            f"Variant {row['variant']} "
            f"| Trades={int(row['trades'])} "
            f"| PF={row.get('profit_factor', np.nan):.2f} "
            f"| Avg={row.get('avg_option_return_pct', np.nan):.2f}% "
            f"| Win={row.get('win_rate_pct', np.nan):.1f}% "
            f"| DD={row.get('max_drawdown_pct', np.nan):.2f}%"
        )

    print("\nFiles written to:")
    print(
        RESULTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()
