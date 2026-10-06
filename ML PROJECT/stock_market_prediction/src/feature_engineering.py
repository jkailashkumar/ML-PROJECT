"""
MODULE 3: FEATURE ENGINEERING
==============================
Responsible for engineering predictive features from historical price data:
1. Base & Lagged features (Previous Close, Open, High, Low, Volume)
2. Technical indicators:
   - Simple Moving Average (SMA 20, SMA 50)
   - Exponential Moving Average (EMA 20)
   - Relative Strength Index (RSI 14)
   - Moving Average Convergence Divergence (MACD 12, 26, 9) & MACD Signal
   - Daily Return
   - Rolling Volatility (20-day window)
3. Target Variable:
   - Next Trading Day's Close Price (Close shifted by -1)

NO DATA LEAKAGE:
All features computed for index 't' depend strictly on information available
at or prior to time 't'. Tomorrow's Open, High, Low, or Volume are NEVER used.
"""

import pandas as pd
import numpy as np


# List of final feature column names used for training and inference
FEATURE_COLUMNS = [
    "Prev_Close",
    "Prev_Open",
    "Prev_High",
    "Prev_Low",
    "Prev_Volume",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Daily_Return",
    "Volatility_20"
]

TARGET_COLUMN = "Target_Next_Close"


# ─────────────────────────────────────────────────────────────────
# Technical Indicator Calculations (Pure Pandas/NumPy - Zero Dependencies)
# ─────────────────────────────────────────────────────────────────

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI 14) using Wilder's smoothing method.
    RSI ranges from 0 to 100.
    > 70 indicates overbought; < 30 indicates oversold.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Exponential moving average with alpha = 1 / period (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Replace division by zero if loss is 0
    rsi = rsi.fillna(50.0)
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calculate MACD, MACD Signal Line, and MACD Histogram.
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(signal) of MACD
    """
    ema_fast = calculate_ema(series, span=fast)
    ema_slow = calculate_ema(series, span=slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_daily_return(series: pd.Series) -> pd.Series:
    """Calculate percentage daily return."""
    return series.pct_change()


def calculate_volatility(daily_returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling annualized historical volatility (252 trading days/year)."""
    return daily_returns.rolling(window=window).std() * np.sqrt(252)


# ─────────────────────────────────────────────────────────────────
# Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, include_target: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate all lag features and technical indicators from processed OHLCV data.

    Parameters
    ----------
    df             : Preprocessed DataFrame containing ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    include_target : If True, computes and shifts the target variable (Next_Day_Close)

    Returns
    -------
    (feature_df, feature_names)
    """
    df = df.copy()

    # 1. Base Lagged Features (t-1)
    # Using previous period values ensures strictly zero lookahead bias
    df["Prev_Close"] = df["Close"].shift(1)
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Volume"] = df["Volume"].shift(1)

    # 2. Technical Indicators (calculated on Close price available up to date t)
    df["SMA_20"] = calculate_sma(df["Close"], window=20)
    df["SMA_50"] = calculate_sma(df["Close"], window=50)
    df["EMA_20"] = calculate_ema(df["Close"], span=20)
    df["RSI_14"] = calculate_rsi(df["Close"], period=14)

    macd, macd_signal, macd_hist = calculate_macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    df["Daily_Return"] = calculate_daily_return(df["Close"])
    df["Volatility_20"] = calculate_volatility(df["Daily_Return"], window=20)

    # 3. Target Variable (Next Trading Day's Close)
    if include_target:
        # Shift -1 means the value at index t is tomorrow's Close (t+1)
        df[TARGET_COLUMN] = df["Close"].shift(-1)

        # Drop rows where any feature or the target is NaN
        # (e.g., initial 50 rows for SMA_50, and the very last row where tomorrow's price is yet unknown)
        clean_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    else:
        clean_df = df.dropna(subset=FEATURE_COLUMNS).copy()

    clean_df.reset_index(drop=True, inplace=True)
    return clean_df, FEATURE_COLUMNS


def get_latest_inference_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Extract the most recent day's features to predict TOMORROW'S next-day closing price.
    Unlike training, we do NOT drop the last row.

    Returns
    -------
    (single_row_feature_df, metadata_dict)
    """
    # Calculate features without dropping the last row
    feature_df, _ = build_features(df, include_target=False)

    if feature_df.empty:
        raise ValueError("Insufficient data to compute technical features for inference.")

    latest_row = feature_df.iloc[[-1]].copy()
    features_only = latest_row[FEATURE_COLUMNS].copy()

    metadata = {
        "date": latest_row["Date"].values[0],
        "current_close": float(latest_row["Close"].values[0]),
        "sma_20": float(latest_row["SMA_20"].values[0]),
        "sma_50": float(latest_row["SMA_50"].values[0]),
        "rsi_14": float(latest_row["RSI_14"].values[0]),
        "macd": float(latest_row["MACD"].values[0]),
        "macd_signal": float(latest_row["MACD_Signal"].values[0]),
        "volatility_20": float(latest_row["Volatility_20"].values[0]),
        "daily_return": float(latest_row["Daily_Return"].values[0])
    }

    return features_only, metadata
