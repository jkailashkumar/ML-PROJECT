"""
MODULE 5: MODEL EVALUATION
===========================
Evaluates regression models using standard and financial metrics:
1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. Coefficient of Determination (R² Score)
4. Directional Accuracy (%):
   Measures how often the predicted next-day price movement (UP or DOWN)
   matches the true next-day price movement relative to today's close.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_directional_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    current_close: pd.Series | np.ndarray
) -> float:
    """
    Compute Directional Accuracy (Hit Ratio).

    Formula:
    actual_dir    = sign(y_true - current_close)
    predicted_dir = sign(y_pred - current_close)
    accuracy      = mean(actual_dir == predicted_dir) * 100%

    Parameters
    ----------
    y_true        : Actual next-day closing prices
    y_pred        : Predicted next-day closing prices
    current_close : Today's closing price (reference point for movement)

    Returns
    -------
    Directional accuracy as a percentage (0.0 to 100.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    current = np.asarray(current_close)

    # True movement: +1 for UP, -1 for DOWN, 0 for FLAT
    true_direction = np.sign(y_true - current)
    pred_direction = np.sign(y_pred - current)

    # Count matching directions
    correct_matches = (true_direction == pred_direction)
    return float(np.mean(correct_matches) * 100.0)


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    current_close: pd.Series | np.ndarray | None = None
) -> dict:
    """
    Compute comprehensive performance metrics for a single model.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    dir_acc = None
    if current_close is not None:
        dir_acc = calculate_directional_accuracy(y_true, y_pred, current_close)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Directional_Accuracy": round(dir_acc, 2) if dir_acc is not None else "N/A"
    }


def compare_models(
    predictions_dict: dict,
    y_true: pd.Series | np.ndarray,
    current_close: pd.Series | np.ndarray | None = None
) -> tuple[pd.DataFrame, str]:
    """
    Generate a comparison table for all trained models.
    Identifies the best-performing model based on the lowest RMSE.

    Parameters
    ----------
    predictions_dict : dict of {model_name: y_pred_array}
    y_true           : actual test target values
    current_close    : today's close prices corresponding to test instances

    Returns
    -------
    (comparison_df, best_model_name)
    """
    rows = []
    best_model = None
    lowest_rmse = float("inf")

    for model_name, y_pred in predictions_dict.items():
        if y_pred is None:
            continue

        metrics = evaluate_model(y_true, y_pred, current_close)
        row = {
            "Model": model_name,
            "MAE ($)": metrics["MAE"],
            "RMSE ($)": metrics["RMSE"],
            "R2 Score": metrics["R2"],
            "Directional Accuracy (%)": metrics["Directional_Accuracy"]
        }
        rows.append(row)

        # Track best model by RMSE
        if metrics["RMSE"] < lowest_rmse:
            lowest_rmse = metrics["RMSE"]
            best_model = model_name

    comparison_df = pd.DataFrame(rows)
    return comparison_df, best_model or "Linear Regression"
