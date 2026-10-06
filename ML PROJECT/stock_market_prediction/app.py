"""
=============================================================================
Stock Market Price Prediction & Intelligent Trading Signal System
=============================================================================
A complete, modular, college-level Machine Learning project.
5-Module Architecture:
  1. Data Collection (yfinance)
  2. Data Preprocessing (chronological cleaning & sanitization)
  3. Feature Engineering (lag features, SMA, EMA, RSI, MACD, volatility)
  4. ML Training & Inference (Linear Regression, Random Forest, XGBoost)
  5. Evaluation, Signal Generation & Interactive Visualization (Plotly)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import modular pipeline components
from src.data_collection import download_stock_data, load_cached_data, get_stock_info
from src.preprocessing import preprocess_data
from src.feature_engineering import build_features, get_latest_inference_features, TARGET_COLUMN
from src.model_training import train_models
from src.evaluation import compare_models
from src.signals import build_signal_summary, DISCLAIMER_TEXT
from src.prediction import predict_next_day, get_feature_importance, generate_market_explanation
from src.visualization import (
    plot_price_and_sma,
    plot_rsi,
    plot_macd,
    plot_volume,
    plot_actual_vs_predicted,
    plot_feature_importance
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Professional Academic Styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market ML Prediction & Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Blue / Clean White Professional Card Theme
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #CBD5E1;
        margin: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-subtext {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 12.5px;
        line-height: 1.5;
        margin: 16px 0;
    }

    /* Section Header */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 26px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sidebar Navigation & Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.markdown("### **System Configuration**")
    st.caption("College Capstone ML Demonstration")

    st.markdown("---")
    st.markdown("#### **1. Stock Selection**")

    # Common presets + custom input option
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "CUSTOM"]
    selected_preset = st.selectbox("Quick Select Ticker:", popular_stocks, index=0)

    if selected_preset == "CUSTOM":
        ticker_input = st.text_input("Enter Ticker Symbol:", value="META").strip().upper()
    else:
        ticker_input = selected_preset

    # Date range selection (minimum 2 years recommended)
    default_start = (datetime.today() - timedelta(days=5 * 365)).date()
    default_end = datetime.today().date()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=default_start)
    with col_d2:
        end_date = st.date_input("End Date", value=default_end)

    st.markdown("---")
    st.markdown("#### **2. Trading Signal Settings**")
    threshold_pct = st.slider(
        "Signal Threshold (±%)",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.25,
        help="Expected Return % required to trigger a BUY or SELL signal."
    )

    st.markdown("---")
    st.markdown("#### **3. ML Model Selection**")
    model_choice = st.selectbox(
        "Active Model for Prediction:",
        ["Best Model (Auto-Selected)", "Linear Regression", "Random Forest", "XGBoost"],
        index=0
    )

    # Analyze Button
    run_btn = st.button("🚀 Run Analysis & Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("🎓 **Academic Project Details**")
    st.caption("• Architecture: 5-Stage Modular Pipeline\n• Time-Series: Strictly chronological split\n• Lookahead bias: Zero")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Application State & Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
# Cache data processing and model training to keep dashboard responsive
@st.cache_data(show_spinner=False, ttl=3600)
def execute_pipeline(ticker: str, start_str: str, end_str: str):
    """
    Executes Modules 1 through 4 end-to-end with caching.
    """
    # Module 1: Data Collection
    raw_df, dl_msg = download_stock_data(ticker, start_str, end_str)
    if raw_df is None:
        # Check if local cache exists
        cached = load_cached_data(ticker)
        if cached is not None and len(cached) >= 100:
            raw_df = cached
            dl_msg = f"Loaded cached local data for {ticker}."
        else:
            return None, dl_msg, None, None, None

    # Module 2: Data Preprocessing
    clean_df, prep_report = preprocess_data(raw_df, ticker=ticker)
    if clean_df is None or len(clean_df) < 80:
        return None, "Error: Preprocessed data contains insufficient rows for modeling.", None, None, None

    # Module 3: Feature Engineering
    features_df, feature_cols = build_features(clean_df, include_target=True)
    if len(features_df) < 50:
        return None, "Error: Insufficient rows after calculating indicators and target.", None, None, None

    # Latest row for next-day forward inference
    latest_feat, latest_meta = get_latest_inference_features(clean_df)

    # Module 4: Machine Learning Model Training
    train_results = train_models(
        features_df, feature_cols, TARGET_COLUMN, ticker=ticker, test_ratio=0.20
    )

    return clean_df, dl_msg, features_df, train_results, (latest_feat, latest_meta)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Dashboard Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📈 Stock Market Price Prediction & Intelligent Trading Signal System</div>
    <div class="hero-subtitle">
        An end-to-end Machine Learning pipeline utilizing historical OHLCV data, technical indicators, 
        chronological time-series validation, and quantitative risk modeling.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline Execution & Safety Checks
# ─────────────────────────────────────────────────────────────────────────────
start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

with st.spinner(f"Analyzing {ticker_input} historical data and training ML models..."):
    clean_df, msg, features_df, train_results, inference_bundle = execute_pipeline(
        ticker_input, start_str, end_str
    )

# Graceful error handling for missing/invalid data
if clean_df is None or train_results is None:
    st.error(f"❌ **Data Acquisition & Modeling Error:** {msg}")
    st.info(
        "💡 **Troubleshooting Tips:**\n"
        "1. Verify that the stock ticker symbol is valid (e.g., AAPL, MSFT, GOOGL, NVDA).\n"
        "2. Ensure your date range spans at least 1-2 years to provide sufficient training history.\n"
        "3. Check your internet connection for Yahoo Finance API queries."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation, Model Selection & Inference
# ─────────────────────────────────────────────────────────────────────────────
latest_feat, latest_meta = inference_bundle
test_actual = train_results["test_actual"]
predictions = train_results["predictions"]
test_dates = train_results["test_dates"]
test_df = train_results["test_df"]
scaler = train_results["scaler"]
models = train_results["models"]
feature_names = train_results["feature_names"]

# Model comparison table
comparison_df, auto_best_model = compare_models(
    predictions, test_actual, current_close=test_df["Close"]
)

# Determine the user-selected active model
if model_choice.startswith("Best Model"):
    active_model_name = auto_best_model
else:
    active_model_name = model_choice

# Fallback if selected model is unavailable (e.g., XGBoost missing)
if active_model_name not in models or models[active_model_name] is None:
    st.warning(f"⚠️ {active_model_name} is not available in current environment. Reverting to {auto_best_model}.")
    active_model_name = auto_best_model

active_model = models[active_model_name]
active_test_pred = predictions[active_model_name]

# Predict next trading day's closing price
predicted_next_close = predict_next_day(
    active_model, active_model_name, scaler, latest_feat
)
current_close = latest_meta["current_close"]

# Signal and Risk Analysis
signal_data = build_signal_summary(
    current_price=current_close,
    predicted_price=predicted_next_close,
    annualized_volatility=latest_meta["volatility_20"],
    buy_threshold=threshold_pct,
    sell_threshold=-threshold_pct
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Top KPI Summary Cards
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Price</div>
        <div class="kpi-value">${signal_data['current_price']:.2f}</div>
        <div class="kpi-subtext" style="color: #64748B;">Latest Close</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    pred_color = "#10B981" if signal_data['predicted_price'] >= signal_data['current_price'] else "#EF4444"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Predicted Close</div>
        <div class="kpi-value" style="color: {pred_color};">${signal_data['predicted_price']:.2f}</div>
        <div class="kpi-subtext" style="color: #64748B;">Next Trading Day</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    ret_val = signal_data['expected_return']
    ret_color = "#10B981" if ret_val > 0 else ("#EF4444" if ret_val < 0 else "#64748B")
    sign_str = "+" if ret_val > 0 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Expected Return</div>
        <div class="kpi-value" style="color: {ret_color};">{sign_str}{ret_val:.2f}%</div>
        <div class="kpi-subtext" style="color: #64748B;">Forecast Delta</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    sig = signal_data['signal']
    sig_bg = "#DCFCE7" if sig == "BUY" else ("#FEE2E2" if sig == "SELL" else "#FEF3C7")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Indicative Signal</div>
        <div class="kpi-value" style="color: {signal_data['signal_color']};">{sig}</div>
        <div class="kpi-subtext" style="background-color: {sig_bg}; padding: 2px 6px; border-radius: 4px; display: inline-block;">
            Threshold: ±{threshold_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    risk_level = signal_data['risk_level']
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Market Risk</div>
        <div class="kpi-value" style="color: {signal_data['risk_color']};">{risk_level.split()[0]}</div>
        <div class="kpi-subtext" style="color: #64748B;">Vol: {latest_meta['volatility_20']*100:.1f}% Ann.</div>
    </div>
    """, unsafe_allow_html=True)

# Mandatory Educational Disclaimer
st.markdown(f"""
<div class="disclaimer-card">
    <b>⚠️ ACADEMIC & RESEARCH DISCLAIMER:</b> {signal_data['disclaimer']}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SECTION 1: Stock Overview & Key Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1. Asset Overview & Market Profile</div>', unsafe_allow_html=True)

stock_meta = get_stock_info(ticker_input)
recent_change = clean_df["Close"].iloc[-1] - clean_df["Close"].iloc[-2] if len(clean_df) >= 2 else 0.0
recent_change_pct = (recent_change / clean_df["Close"].iloc[-2] * 100.0) if len(clean_df) >= 2 else 0.0
high_52w = clean_df["High"].tail(252).max()
low_52w = clean_df["Low"].tail(252).min()
latest_vol = clean_df["Volume"].iloc[-1]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Company Name", stock_meta.get("company_name", ticker_input)[:18])
m2.metric("Sector", stock_meta.get("sector", "N/A"))
m3.metric("Today's Change", f"${recent_change:+.2f}", f"{recent_change_pct:+.2f}%")
m4.metric("Trading Volume", f"{latest_vol:,.0f}")
m5.metric("52-Week High", f"${high_52w:.2f}")
m6.metric("52-Week Low", f"${low_52w:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SECTION 2: Price Chart with Moving Averages
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">2. Historical Price Action & Moving Averages</div>', unsafe_allow_html=True)
st.caption("Visualizing closing price alongside 20-day (short-term) and 50-day (medium-term) Simple Moving Averages.")

fig_price = plot_price_and_sma(features_df, ticker=ticker_input)
st.plotly_chart(fig_price, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SECTION 3: Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">3. Technical Indicators & Oscillators</div>', unsafe_allow_html=True)
st.caption("Quantitative momentum and volume indicators computed strictly using prior historical data.")

tab_rsi, tab_macd, tab_volume = st.tabs(["📊 RSI (Relative Strength Index)", "📈 MACD & Signal", "📦 Daily Volume"])

with tab_rsi:
    st.plotly_chart(plot_rsi(features_df), use_container_width=True)
    st.markdown("""
    **RSI Interpretation Guide:**
    - **RSI > 70 (Overbought):** Stock may be overextended to the upside; potential pullback risk.
    - **RSI < 30 (Oversold):** Stock may be oversold; potential mean-reversion buying opportunity.
    - **30 - 70 (Neutral):** Standard trading channel.
    """)

with tab_macd:
    st.plotly_chart(plot_macd(features_df), use_container_width=True)
    st.markdown("""
    **MACD Interpretation Guide:**
    - **MACD crossing above Signal Line (Green Histogram):** Bullish momentum acceleration.
    - **MACD crossing below Signal Line (Red Histogram):** Bearish momentum deceleration.
    """)

with tab_volume:
    st.plotly_chart(plot_volume(features_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. SECTION 4: Model Comparison & Validation Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">4. Multi-Model Performance Comparison</div>', unsafe_allow_html=True)
st.caption("Evaluated on out-of-sample data via chronological time-series splitting (80% Train / 20% Test).")

col_comp1, col_comp2 = st.columns([3, 2])

with col_comp1:
    # Highlight the best model in the styled dataframe
    def highlight_best(s):
        is_best = s["Model"] == auto_best_model
        return ["background-color: #DCFCE7; font-weight: bold" if is_best else "" for _ in s]

    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.success(f"🏆 **Top-Performing Model:** `{auto_best_model}` achieved the lowest Root Mean Squared Error (RMSE).")

with col_comp2:
    st.markdown("""
    **Evaluation Metrics Guide for Viva:**
    - **MAE (Mean Absolute Error):** Average dollar distance between prediction and actual price.
    - **RMSE (Root Mean Squared Error):** Penalizes larger outlier forecasting errors more heavily.
    - **R² Score:** Proportion of price variance explained by features (closer to 1.0 is superior).
    - **Directional Accuracy (%):** Percentage of trading days where the model correctly predicted whether tomorrow's close goes **UP or DOWN**.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 12. SECTION 5: Actual vs. Predicted Price
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">5. Out-of-Sample Backtest: Actual vs. Predicted</div>', unsafe_allow_html=True)
st.caption(f"Comparing actual test targets vs. predictions generated by **{active_model_name}** on unseen test data.")

fig_act_pred = plot_actual_vs_predicted(
    dates=test_dates,
    y_actual=test_actual,
    y_pred=active_test_pred,
    model_name=active_model_name
)
st.plotly_chart(fig_act_pred, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 13. SECTION 6: Prediction & Signal Synthesis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">6. Next Trading Day Forecast & Signal Breakdown</div>', unsafe_allow_html=True)

col_sig1, col_sig2 = st.columns(2)

with col_sig1:
    st.markdown("#### **Decision Matrix**")
    st.write(f"- **Current Close Price:** `${signal_data['current_price']:.2f}`")
    st.write(f"- **Forecasted Next Close ({active_model_name}):** `${signal_data['predicted_price']:.2f}`")
    st.write(f"- **Expected Return:** `{signal_data['expected_return']:+.2f}%`")
    st.write(f"- **Trigger Thresholds:** `BUY >= +{threshold_pct:.1f}%` | `SELL <= -{threshold_pct:.1f}%`")
    st.info(f"📌 **Signal Rationale:** {signal_data['signal_explanation']}")

with col_sig2:
    st.markdown("#### **Risk Profile Matrix**")
    st.write(f"- **Classified Risk:** `{signal_data['risk_level']}`")
    st.write(f"- **20-Day Annualized Volatility:** `{latest_meta['volatility_20']*100:.2f}%`")
    st.write(f"- **Daily Expected Fluctuation Band:** `±{(latest_meta['volatility_20']/np.sqrt(252))*100:.2f}%`")
    st.warning(f"🛡️ **Risk Assessment:** {signal_data['risk_explanation']}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. SECTION 7: Explainability & Technical Synthesis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">7. Model Explainability & Market Context</div>', unsafe_allow_html=True)
st.caption("Interpreting the mathematical drivers behind model predictions without false causal assumptions.")

col_imp, col_narrative = st.columns([1, 1])

with col_imp:
    # Feature Importance Bar Chart
    imp_df = get_feature_importance(active_model, active_model_name, feature_names)
    st.plotly_chart(plot_feature_importance(imp_df, top_n=7), use_container_width=True)

with col_narrative:
    st.markdown("#### **Why this Prediction? (Technical Synthesis)**")
    explanations = generate_market_explanation(latest_meta, signal_data["expected_return"])
    for exp in explanations:
        st.markdown(f"- {exp}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. College Viva & PBL Reference Guide (Collapsible Expander)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🎓 **Project Demonstration & Viva Cheat Sheet (Click to Expand)**", expanded=False):
    st.markdown("""
    ### 5-Module Engineering Architecture Overview:
    1. **Module 1 (Data Collection):** Automated ingestion of historical OHLCV data from Yahoo Finance via `yfinance`. Caches data locally in `data/raw/` to prevent redundant network calls.
    2. **Module 2 (Data Preprocessing):** Sanitization pipeline ensuring chronological sorting, removal of duplicate dates, numeric conversion, sanity filtering ($High \ge Low > 0$), and storage in `data/processed/`.
    3. **Module 3 (Feature Engineering):** Generates $t-1$ lag features (Prev Close/Open/High/Low/Volume) and technical indicators (SMA20, SMA50, EMA20, RSI14, MACD, Volatility). Defines Target as $Close_{t+1}$ using a $-1$ shift.
    4. **Module 4 (Model Training):** Strictly chronological 80/20 train/test split. Scaler fit solely on training data. Fits Linear Regression, Random Forest, and XGBoost. Serializes models to `models/`.
    5. **Module 5 (Evaluation & Signals):** Calculates MAE, RMSE, $R^2$, and Directional Accuracy. Generates BUY/HOLD/SELL signals with threshold rules and historical volatility risk bands.

    ### Key Viva Questions & Answers:
    - **Q: Why can't we use standard `train_test_split(shuffle=True)`?**  
      *A: Randomly shuffling time-series data introduces lookahead data leakage, allowing the model to peek into the future and producing falsely inflated accuracies that fail in live markets.*
    - **Q: What is Directional Accuracy and why is it important in trading?**  
      *A: Unlike RMSE which measures dollar-distance, Directional Accuracy measures whether the model correctly foresaw price direction (UP vs DOWN). In financial trading, predicting movement direction is often more critical for profit/loss than predicting the exact dollar cent.*
    - **Q: How is future lookahead prevented in technical indicators?**  
      *A: All indicators and lag features are calculated exclusively using data up to timestamp $t$. Tomorrow's Open, High, Low, or Volume are never used as features.*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 13px;'>"
    "College Capstone ML Project • Built with Streamlit, Scikit-learn, XGBoost & Plotly • "
    "Designed for PBL Demonstration & Academic Viva"
    "</div>",
    unsafe_allow_html=True
)
