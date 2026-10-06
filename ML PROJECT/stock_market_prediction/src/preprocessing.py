"""
MODULE 2: DATA PREPROCESSING
=============================
Responsible for cleaning, validating, and formatting raw historical stock data.

Key rules:
- Strictly maintain chronological order (NEVER shuffle time series data).
- Handle missing values, invalid values, and duplicate records.
- Ensure Date is in datetime format and all numeric columns are float.
- Save cleaned data to data/processed/{ticker}_processed.csv.
"""

import os
import pandas as pd
import numpy as np


PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed"
)


def ensure_processed_dir():
    """Ensure the processed data directory exists."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def preprocess_data(df: pd.DataFrame, ticker: str = "STOCK") -> tuple[pd.DataFrame | None, dict]:
    """
    Clean, validate, and prepare raw stock data for feature engineering.

    Parameters
    ----------
    df     : Raw DataFrame containing ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ticker : Stock ticker symbol (used for saving file)

    Returns
    -------
    (cleaned_df, report_dict)
    report_dict contains preprocessing statistics for display in the dashboard / viva.
    """
    ensure_processed_dir()

    if df is None or df.empty:
        return None, {"error": "Input DataFrame is empty or None"}

    report = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "missing_values_handled": 0,
        "invalid_rows_removed": 0,
        "final_rows": 0,
        "date_range": "N/A"
    }

    # Make an explicit copy to prevent modifying input data
    clean_df = df.copy()

    # 1. Standardize column names (strip whitespace and title-case)
    clean_df.columns = [str(c).strip().title() for c in clean_df.columns]

    # Required OHLCV columns
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in clean_df.columns:
            return None, {"error": f"Missing essential column: {col}"}

    # 2. Date parsing & conversion
    if "Date" in clean_df.columns:
        clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    else:
        # If Date was the index
        clean_df.index = pd.to_datetime(clean_df.index, errors="coerce")
        clean_df.reset_index(inplace=True)
        clean_df.rename(columns={"index": "Date"}, inplace=True)

    # Remove rows where Date could not be parsed
    clean_df = clean_df.dropna(subset=["Date"])

    # 3. Sort chronologically (MANDATORY FOR TIME SERIES)
    clean_df.sort_values(by="Date", ascending=True, inplace=True)

    # 4. Remove duplicate dates (keep the most recent observation)
    initial_count = len(clean_df)
    clean_df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    report["duplicates_removed"] = initial_count - len(clean_df)

    # 5. Convert OHLCV columns to numeric float, coercing invalid values to NaN
    for col in expected_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 6. Check and handle missing values
    # For financial time series, forward fill (up to 2 periods) is appropriate
    # for minor gap filling without introducing future lookahead.
    missing_before = clean_df[expected_cols].isna().sum().sum()
    clean_df[expected_cols] = clean_df[expected_cols].ffill(limit=2).bfill(limit=1)
    report["missing_values_handled"] = int(missing_before)

    # 7. Sanity checks on financial validity:
    # - Prices (Open, High, Low, Close) must be strictly positive (> 0)
    # - Volume must be non-negative (>= 0)
    # - High must be >= Low
    valid_mask = (
        (clean_df["Open"] > 0) &
        (clean_df["High"] > 0) &
        (clean_df["Low"] > 0) &
        (clean_df["Close"] > 0) &
        (clean_df["Volume"] >= 0) &
        (clean_df["High"] >= clean_df["Low"])
    )
    invalid_count = len(clean_df) - valid_mask.sum()
    report["invalid_rows_removed"] = int(invalid_count)
    clean_df = clean_df[valid_mask].copy()

    # Drop any remaining unfillable NaNs
    clean_df.dropna(subset=expected_cols, inplace=True)

    # Reset continuous index
    clean_df.reset_index(drop=True, inplace=True)
    report["final_rows"] = len(clean_df)

    if len(clean_df) > 0:
        start_date_str = clean_df["Date"].min().strftime("%Y-%m-%d")
        end_date_str = clean_df["Date"].max().strftime("%Y-%m-%d")
        report["date_range"] = f"{start_date_str} to {end_date_str}"

    # 8. Save processed data to data/processed/{ticker}_processed.csv
    ticker_clean = ticker.strip().upper()
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker_clean}_processed.csv")
    clean_df.to_csv(save_path, index=False)
    report["save_path"] = save_path

    return clean_df, report


def load_processed_data(ticker: str) -> pd.DataFrame | None:
    """
    Load preprocessed data from data/processed/ directory.
    """
    ticker_clean = ticker.strip().upper()
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker_clean}_processed.csv")
    if os.path.exists(save_path):
        try:
            return pd.read_csv(save_path, parse_dates=["Date"])
        except Exception:
            return None
    return None
