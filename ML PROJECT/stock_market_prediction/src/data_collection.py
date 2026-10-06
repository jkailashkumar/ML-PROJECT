"""
MODULE 1: DATA COLLECTION
==========================
Responsible for downloading historical stock market data using yfinance.
Stores raw CSV files in data/raw/ directory.
Handles errors gracefully (invalid tickers, network issues, etc.)
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2019-01-01"
DEFAULT_END = datetime.today().strftime("%Y-%m-%d")


def ensure_directories():
    """Create required directories if they do not exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_stock_data(ticker: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, str]:
    """
    Download historical OHLCV stock data from Yahoo Finance.

    Parameters
    ----------
    ticker     : Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : Start date string 'YYYY-MM-DD'
    end_date   : End date string 'YYYY-MM-DD'

    Returns
    -------
    (DataFrame, message) — DataFrame is None if download failed.
    """
    ensure_directories()
    ticker = ticker.strip().upper()

    # ── Validate date inputs ──
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt >= end_dt:
            return None, "Start date must be before end date."
        if end_dt > datetime.today():
            end_date = datetime.today().strftime("%Y-%m-%d")
    except ValueError:
        return None, "Invalid date format. Please use YYYY-MM-DD."

    # ── Minimum data requirement ──
    min_days = 365
    if (end_dt - start_dt).days < min_days:
        return None, f"Please select at least {min_days} days of historical data for meaningful analysis."

    try:
        # Download data using yfinance
        raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        # Handle empty response (invalid ticker or no data)
        if raw_df is None or raw_df.empty:
            return None, (
                f"No data found for ticker '{ticker}'. "
                "Please check the ticker symbol or select a different date range."
            )

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        # Keep only the standard OHLCV columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in raw_df.columns]
        if missing:
            return None, f"Downloaded data is missing columns: {missing}"

        df = raw_df[required_cols].copy()

        # Reset index so Date becomes a column
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

        # Ensure Date column is present after reset
        if "Date" not in df.columns:
            df.index.name = "Date"
            df.reset_index(inplace=True)

        # Remove duplicate dates (keep last entry)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)

        # Sort chronologically
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Validate we have sufficient rows
        if len(df) < 100:
            return None, (
                f"Only {len(df)} rows retrieved for '{ticker}'. "
                "Please choose a longer date range (at least 1 year of trading days)."
            )

        # Save to CSV
        save_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        df.to_csv(save_path, index=False)

        msg = (
            f"Successfully downloaded {len(df)} trading days of data for {ticker}. "
            f"Saved to {save_path}"
        )
        return df, msg

    except Exception as e:
        error_msg = str(e)
        # Provide friendly messages for common errors
        if "No data found" in error_msg or "404" in error_msg:
            return None, f"Ticker '{ticker}' not found on Yahoo Finance. Please verify the symbol."
        if "Connection" in error_msg or "network" in error_msg.lower() or "timeout" in error_msg.lower():
            return None, "Network error. Please check your internet connection and try again."
        return None, f"An unexpected error occurred while downloading data: {error_msg}"


def load_cached_data(ticker: str) -> pd.DataFrame | None:
    """
    Load previously downloaded CSV data from local cache.

    Parameters
    ----------
    ticker : Stock ticker symbol

    Returns
    -------
    DataFrame or None if no cache exists.
    """
    ticker = ticker.strip().upper()
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            return df
        except Exception:
            return None
    return None


def get_available_tickers() -> list[str]:
    """Return list of tickers that have been downloaded and cached locally."""
    if not os.path.exists(RAW_DATA_DIR):
        return []
    return [f.replace(".csv", "") for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock information (company name, sector, etc.) from Yahoo Finance.

    Returns a dict with metadata, or an empty dict on failure.
    """
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception:
        return {"company_name": ticker, "sector": "N/A", "industry": "N/A",
                "market_cap": "N/A", "currency": "USD", "exchange": "N/A"}
