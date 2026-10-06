"""
Comprehensive pipeline validation script for verification and testing.
"""

from src.data_collection import download_stock_data, get_stock_info
from src.preprocessing import preprocess_data
from src.feature_engineering import build_features, get_latest_inference_features, TARGET_COLUMN
from src.model_training import train_models
from src.evaluation import compare_models
from src.signals import build_signal_summary
from src.prediction import predict_next_day, get_feature_importance

def run_tests():
    print("=== 1. Testing Valid Ticker AAPL Data Ingestion ===")
    raw_df, msg = download_stock_data("AAPL", "2022-01-01", "2025-01-01")
    print(msg)
    assert raw_df is not None, "Raw dataframe should not be None"
    print(f"Downloaded shape: {raw_df.shape}")

    print("\n=== 2. Testing Preprocessing ===")
    clean_df, report = preprocess_data(raw_df, ticker="AAPL")
    print("Report:", report)
    assert len(clean_df) > 100, "Clean dataframe should have sufficient rows"

    print("\n=== 3. Testing Feature Engineering ===")
    feat_df, cols = build_features(clean_df, include_target=True)
    print("Engineered features count:", len(cols))
    print("Rows after features & target:", len(feat_df))
    assert len(cols) == 14, f"Expected 14 features, got {len(cols)}"

    print("\n=== 4. Testing Model Training (Chrono Split) ===")
    res = train_models(feat_df, cols, TARGET_COLUMN, ticker="AAPL", test_ratio=0.20)
    print("Trained models:", list(res["models"].keys()))
    print(f"Train set: {len(res['train_df'])} rows | Test set: {len(res['test_df'])} rows")

    print("\n=== 5. Testing Evaluation & Comparison ===")
    comp_df, best_m = compare_models(res["predictions"], res["test_actual"], res["test_df"]["Close"])
    print(comp_df.to_string())
    print("Best performing model:", best_m)

    print("\n=== 6. Testing Inference & Signal ===")
    latest_feat, latest_meta = get_latest_inference_features(clean_df)
    pred = predict_next_day(res["models"][best_m], best_m, res["scaler"], latest_feat)
    print(f"Current close: ${latest_meta['current_close']:.2f}")
    print(f"Predicted next close: ${pred:.2f}")
    sig = build_signal_summary(latest_meta["current_close"], pred, latest_meta["volatility_20"])
    print("Signal Output:", sig["signal"], "| Expected Return:", sig["expected_return"], "% | Risk:", sig["risk_level"])

    print("\n=== 7. Testing Second Valid Ticker MSFT ===")
    msft_raw, msft_msg = download_stock_data("MSFT", "2023-01-01", "2024-01-01")
    print(msft_msg)
    assert msft_raw is not None, "MSFT should download successfully"

    print("\n=== 8. Testing Invalid Ticker Graceful Handling ===")
    bad_df, bad_msg = download_stock_data("INVALIDXYZ999", "2023-01-01", "2024-01-01")
    print("Bad ticker message:", bad_msg)
    assert bad_df is None, "Bad ticker should return None"

    print("\n" + "="*50)
    print("ALL 8 PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    run_tests()
