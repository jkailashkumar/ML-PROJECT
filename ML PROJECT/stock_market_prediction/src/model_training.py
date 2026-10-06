"""
MODULE 4: MACHINE LEARNING TRAINING & PREDICTION
=================================================
Trains and compares multiple regression models on historical time-series data:
1. Linear Regression (with standard scaling)
2. Random Forest Regressor
3. XGBoost Regressor (with graceful fallback if xgboost is not installed)

CRITICAL TIME-SERIES RULES:
- CHRONOLOGICAL 80/20 train-test split (NEVER random shuffle).
- Preprocessing scaler is fit ONLY on training data to avoid future lookahead leakage.
- TimeSeriesSplit cross-validation support.
- All models are trained on the exact same data partitions.
- Trained models are serialized using joblib into models/{ticker}_{model_name}.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Try importing XGBoost with graceful fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def ensure_models_dir():
    """Ensure the models directory exists."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def chronological_train_test_split(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, test_ratio: float = 0.20
):
    """
    Split time-series data chronologically.

    First (1 - test_ratio) * 100% of rows -> Training Set
    Last test_ratio * 100% of rows        -> Testing Set

    No shuffling is performed, preserving temporal integrity.
    """
    n_samples = len(df)
    split_idx = int(n_samples * (1.0 - test_ratio))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    ticker: str = "STOCK",
    test_ratio: float = 0.20
) -> dict:
    """
    Train Linear Regression, Random Forest, and XGBoost on chronologically split data.

    Returns
    -------
    Dictionary containing:
      - 'models': trained model objects
      - 'predictions': dict of {model_name: y_pred_test}
      - 'test_actual': y_test series
      - 'test_dates': test_df['Date']
      - 'test_df': full test DataFrame with date and OHLCV
      - 'train_df': full train DataFrame
      - 'scaler': fitted StandardScaler (fit ONLY on X_train)
      - 'xgboost_available': bool
      - 'split_idx': integer index where split occurred
      - 'feature_names': feature_cols
    """
    ensure_models_dir()
    ticker = ticker.strip().upper()

    # 1. Chronological Split
    X_train, X_test, y_train, y_test, train_df, test_df = chronological_train_test_split(
        df, feature_cols, target_col, test_ratio=test_ratio
    )

    # 2. Scale features (fit ONLY on training data to prevent lookahead leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained_models = {}
    test_predictions = {}

    # ── Model 1: Linear Regression ──
    lr = LinearRegression()
    # Linear Regression benefits from scaled features
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    trained_models["Linear Regression"] = lr
    test_predictions["Linear Regression"] = y_pred_lr

    # Save to disk
    joblib.dump(lr, os.path.join(MODELS_DIR, f"{ticker}_linear_regression.pkl"))

    # ── Model 2: Random Forest Regressor ──
    # Tree ensembles are scale-invariant, we can pass original or scaled; using unscaled for interpretability
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    trained_models["Random Forest"] = rf
    test_predictions["Random Forest"] = y_pred_rf

    joblib.dump(rf, os.path.join(MODELS_DIR, f"{ticker}_random_forest.pkl"))

    # ── Model 3: XGBoost Regressor ──
    if XGBOOST_AVAILABLE:
        try:
            xgb = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            trained_models["XGBoost"] = xgb
            test_predictions["XGBoost"] = y_pred_xgb

            joblib.dump(xgb, os.path.join(MODELS_DIR, f"{ticker}_xgboost.pkl"))
        except Exception as e:
            print(f"XGBoost training failed, falling back to LR & RF: {e}")
            trained_models["XGBoost"] = None
    else:
        trained_models["XGBoost"] = None

    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{ticker}_features.pkl"))

    return {
        "models": trained_models,
        "predictions": test_predictions,
        "test_actual": y_test,
        "test_dates": test_df["Date"].reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "train_df": train_df.reset_index(drop=True),
        "scaler": scaler,
        "xgboost_available": (trained_models.get("XGBoost") is not None),
        "split_idx": len(train_df),
        "feature_names": feature_cols
    }


def perform_timeseries_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 5
) -> dict:
    """
    Perform TimeSeriesSplit cross validation to verify model robustness
    without any temporal leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[feature_cols].values
    y = df[target_col].values

    cv_results = {"Linear Regression": [], "Random Forest": []}
    if XGBOOST_AVAILABLE:
        cv_results["XGBoost"] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scaler fit only on fold train
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)

        # LR
        lr = LinearRegression().fit(X_tr_sc, y_tr)
        cv_results["Linear Regression"].append(np.mean(np.abs(y_val - lr.predict(X_val_sc))))

        # RF
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        cv_results["Random Forest"].append(np.mean(np.abs(y_val - rf.predict(X_val))))

        # XGB
        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
                cv_results["XGBoost"].append(np.mean(np.abs(y_val - xgb.predict(X_val))))
            except Exception:
                pass

    return {m: float(np.mean(scores)) for m, scores in cv_results.items() if len(scores) > 0}
