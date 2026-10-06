"""
MODULE 7: PREDICTION & EXPLAINABILITY ENGINE
=============================================
Manages forward-looking next-day inference and feature explainability:
1. Generates the next-trading-day closing price prediction.
2. Extracts feature importances (Random Forest, XGBoost) and normalized coefficients (Linear Regression).
3. Produces a responsible, transparent, non-causal explanation of market indicators.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import get_latest_inference_features, FEATURE_COLUMNS


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def predict_next_day(
    model,
    model_name: str,
    scaler,
    latest_features_df: pd.DataFrame
) -> float:
    """
    Predict next trading day's closing price using a trained model.

    Parameters
    ----------
    model               : Trained estimator (LinearRegression, RandomForest, or XGBoost)
    model_name          : Name of the model
    scaler              : Fitted StandardScaler
    latest_features_df  : 1-row DataFrame containing the most recent day's features

    Returns
    -------
    predicted_close : float
    """
    if "Linear Regression" in model_name:
        X_scaled = scaler.transform(latest_features_df)
        pred = model.predict(X_scaled)[0]
    else:
        # Tree models use unscaled features directly
        pred = model.predict(latest_features_df)[0]

    return float(pred)


def get_feature_importance(model, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importance or normalized coefficient magnitude.

    Returns
    -------
    DataFrame with columns ['Feature', 'Importance', 'Percentage']
    sorted in descending order of importance.
    """
    if hasattr(model, "feature_importances_"):
        # Random Forest or XGBoost
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear Regression: absolute coefficients as proxy for relative impact
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to percentages
    total = np.sum(importances)
    if total > 0:
        pcts = (importances / total) * 100.0
    else:
        pcts = np.zeros(len(feature_names))

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Percentage": pcts
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return imp_df


def generate_market_explanation(metadata: dict, expected_return: float) -> list[str]:
    """
    Generate an educational, non-causal explanation synthesizing
    technical indicator readings.

    Parameters
    ----------
    metadata        : dict with latest 'current_close', 'sma_20', 'sma_50', 'rsi_14', 'volatility_20', etc.
    expected_return : predicted return %

    Returns
    -------
    List of explanatory bullet points
    """
    close = metadata["current_close"]
    sma20 = metadata["sma_20"]
    sma50 = metadata["sma_50"]
    rsi = metadata["rsi_14"]
    macd = metadata["macd"]
    macd_signal = metadata["macd_signal"]
    vol = metadata["volatility_20"] * 100.0

    explanations = []

    # 1. Moving Average Trend Alignment
    if close > sma20 > sma50:
        ma_text = (
            f"**Trend Structure (Bullish)**: Today's close (${close:.2f}) trades above both "
            f"SMA 20 (${sma20:.2f}) and SMA 50 (${sma50:.2f}), reflecting positive medium-term momentum."
        )
    elif close < sma20 < sma50:
        ma_text = (
            f"**Trend Structure (Bearish)**: Today's close (${close:.2f}) trades below both "
            f"SMA 20 (${sma20:.2f}) and SMA 50 (${sma50:.2f}), reflecting downward price pressure."
        )
    else:
        ma_text = (
            f"**Trend Structure (Mixed)**: Close is sandwiched between SMA 20 (${sma20:.2f}) "
            f"and SMA 50 (${sma50:.2f}), indicative of a transitional or consolidating market phase."
        )
    explanations.append(ma_text)

    # 2. RSI Momentum
    if rsi >= 70:
        rsi_text = (
            f"**RSI Momentum ({rsi:.1f})**: Relative Strength Index is above 70, indicating technically "
            "overbought conditions where buyer exhaustion or a pullback may occur."
        )
    elif rsi <= 30:
        rsi_text = (
            f"**RSI Momentum ({rsi:.1f})**: Relative Strength Index is below 30, signaling oversold "
            "territory where historical mean-reversion buying interest frequently emerges."
        )
    else:
        rsi_text = (
            f"**RSI Momentum ({rsi:.1f})**: RSI resides in the neutral zone (30-70), signifying balanced "
            "buying and selling pressure without extreme oscillator readings."
        )
    explanations.append(rsi_text)

    # 3. MACD Convergence/Divergence
    if macd > macd_signal:
        macd_text = (
            f"**MACD Signal Line**: MACD ({macd:.2f}) is situated above its Signal Line ({macd_signal:.2f}), "
            "corroborating positive short-term momentum."
        )
    else:
        macd_text = (
            f"**MACD Signal Line**: MACD ({macd:.2f}) lies below its Signal Line ({macd_signal:.2f}), "
            "indicating that short-term momentum is lagging longer-term averages."
        )
    explanations.append(macd_text)

    # 4. Volatility Context
    vol_text = (
        f"**Volatility Regime**: Annualized 20-day volatility is currently {vol:.1f}%. "
        f"Expect standard next-day deviations within approx ±{(vol / np.sqrt(252)):.1f}%."
    )
    explanations.append(vol_text)

    # 5. Non-causal Synthesis Disclaimer
    syn_text = (
        f"**Model Forecast Context**: The ML algorithm weighted these historical lagged patterns "
        f"to forecast a next-day projected change of {expected_return:+.2f}%. "
        "Statistical patterns indicate correlation, not guaranteed market causality."
    )
    explanations.append(syn_text)

    return explanations
