# Stock Market Price Prediction & Intelligent Trading Signal System Using Machine Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1+-3F4F75.svg)](https://plotly.com/)

An end-to-end, modular, college-level Machine Learning and quantitative finance capstone project. The system ingests historical stock data, performs robust chronological preprocessing, engineers momentum and volatility indicators without lookahead bias, trains multiple ML regressors (Linear Regression, Random Forest, XGBoost), evaluates them across statistical and financial metrics, and generates an indicative **BUY / HOLD / SELL** signal with risk assessment.

---

## 1. Problem Statement
Financial equity markets exhibit high stochastic volatility, non-linear dynamics, and systemic risk. Retail investors and student analysts frequently struggle to systematically interpret the interaction between lagged price action and classical technical indicators (Moving Averages, RSI, MACD, and Volatility). 

Furthermore, naive machine learning implementations in financial contexts often commit **lookahead data leakage** by randomly shuffling time-series data or utilizing same-day forward prices as predictors. This project demonstrates how to construct a rigorous, leakage-free time-series forecasting and decision-support pipeline suitable for academic evaluation.

---

## 2. Project Objective
1. **Predict:** Forecast the next trading day's closing price ($Close_{t+1}$) using only information available up to market close on day $t$.
2. **Benchmark:** Compare multiple machine learning architectures (Linear Regression, Random Forest, XGBoost) on identical out-of-sample test partitions.
3. **Evaluate:** Assess model performance using both magnitude metrics ($\text{MAE}$, $\text{RMSE}$, $R^2$) and financial directionality (**Directional Accuracy %**).
4. **Signal & Risk:** Produce an indicative, rule-based **BUY / HOLD / SELL** recommendation based on forecasted expected return ($\Delta\%$) combined with annualized historical volatility risk categorization.
5. **Explain:** Demystify model forecasts through feature importance analysis and technical indicator synthesis.

---

## 3. Key System Features
- **Zero Future Data Leakage:** Features use strictly backward-looking lag windows; target is shifted by $-1$; train/test split is strictly chronological.
- **Multi-Model Benchmark:** Automatically trains and ranks Linear Regression, Random Forest, and XGBoost (with graceful fallback if XGBoost is unavailable).
- **Directional Accuracy Metric:** Evaluates the model's ability to forecast whether the price moves **UP** or **DOWN**, which is critical for real-world trading logic.
- **Quantitative Signal & Risk Classification:** Configurable return thresholds ($\pm 1.5\%$) and volatility classification (Low, Medium, High).
- **Interactive Web Dashboard:** Built with Streamlit and Plotly, styled with a professional dark blue / clean academic palette.
- **Viva Cheat Sheet:** Built-in expander designed specifically for project evaluation and defense.

---

## 4. Five-Module Architecture

```
                    ┌────────────────────────────────────────┐
                    │      Module 1: Data Collection         │
                    │   (yfinance API -> data/raw/*.csv)     │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │     Module 2: Data Preprocessing       │
                    │   (Sanitize, Chronological Sorting,    │
                    │      Positive Bounds Validation)       │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │    Module 3: Feature Engineering       │
                    │  (Lag-1 OHLCV, SMA20/50, EMA20, RSI14, │
                    │    MACD, Volatility -> Target Close t+1)│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Module 4: ML Training & Chrono Split   │
                    │ (80% Train / 20% Test, Scaler fit on   │
                    │  Train only, LR, Random Forest, XGBoost)│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Module 5: Evaluation, Signal & UI      │
                    │ (MAE, RMSE, R², Directional Accuracy,  │
                    │  Expected Return %, BUY/HOLD/SELL, UI) │
                    └────────────────────────────────────────┘
```

### Module 1: Data Collection (`src/data_collection.py`)
- Interfaces with Yahoo Finance API via `yfinance`.
- Downloads daily Open, High, Low, Close, and Volume (OHLCV).
- Default ticker: `AAPL` (Apple Inc.); allows custom user selection (`MSFT`, `GOOGL`, `TSLA`, `NVDA`, etc.).
- Persists raw data to `data/raw/{TICKER}.csv`.
- Resilient error handling for missing tickers, network dropouts, or empty date queries.

### Module 2: Data Preprocessing (`src/preprocessing.py`)
- Enforces strict chronological sorting by `Date` (ascending).
- Eliminates duplicate trading days.
- Forward-fills short missing gaps ($\le 2$ days) to avoid lookahead imputation.
- Performs domain sanity verification: $Open, High, Low, Close > 0$, $Volume \ge 0$, and $High \ge Low$.
- Saves cleaned dataset to `data/processed/{TICKER}_processed.csv`.

### Module 3: Feature Engineering (`src/feature_engineering.py`)
Generates 14 engineered predictor variables derived exclusively from $t$ and prior dates:
1. **$t-1$ Lagged Features:** `Prev_Close`, `Prev_Open`, `Prev_High`, `Prev_Low`, `Prev_Volume`.
2. **Moving Averages:** 20-day Simple Moving Average (`SMA_20`), 50-day Simple Moving Average (`SMA_50`), 20-day Exponential Moving Average (`EMA_20`).
3. **Momentum Oscillators:** 14-day Relative Strength Index (`RSI_14` using Wilder's smoothing).
4. **Trend Convergence:** MACD Line (12, 26), 9-day Signal Line, and MACD Histogram.
5. **Return & Volatility:** Percentage Daily Return and 20-day Annualized Historical Volatility ($\sigma_{20} \times \sqrt{252}$).
6. **Target Definition:** `Target_Next_Close` $= Close_{t+1}$ via `shift(-1)`.

### Module 4: Machine Learning Model Training (`src/model_training.py`)
- Splits data **chronologically** (first 80% for training, final 20% for out-of-sample testing).
- Standardizes features using `StandardScaler` fitted **only** on the training partition.
- Models trained:
  1. **Linear Regression:** Baseline parametric regression.
  2. **Random Forest Regressor:** Non-linear ensemble (100 estimators, max depth 8).
  3. **XGBoost Regressor:** Gradient-boosted decision trees (learning rate 0.05, max depth 4).
- Serializes trained artifacts into `models/{TICKER}_{model}.pkl` via `joblib`.

### Module 5: Evaluation, Trading Signals & Visualization (`src/evaluation.py`, `src/signals.py`, `src/visualization.py`)
- Calculates statistical and financial metrics:
  $$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$
  $$\text{Directional Accuracy} = \frac{1}{n} \sum \mathbb{I}\left(\text{sign}(y_i - Close_{i}) = \text{sign}(\hat{y}_i - Close_{i})\right) \times 100\%$$
- Computes Expected Return:
  $$\text{Expected Return } \% = \left(\frac{\text{Predicted Next Close} - \text{Current Close}}{\text{Current Close}}\right) \times 100$$
- Triggers Decision Signal:
  - $\text{Expected Return} \ge +1.5\% \implies \mathbf{BUY}$
  - $\text{Expected Return} \le -1.5\% \implies \mathbf{SELL}$
  - Otherwise $\implies \mathbf{HOLD}$
- Classifies Risk Level:
  - $\sigma_{ann} < 20\% \implies \mathbf{LOW\ RISK}$
  - $20\% \le \sigma_{ann} \le 35\% \implies \mathbf{MEDIUM\ RISK}$
  - $\sigma_{ann} > 35\% \implies \mathbf{HIGH\ RISK}$

---

## 5. Technology Stack
| Layer | Technologies |
|---|---|
| **Language** | Python 3.11+ |
| **Data Ingestion** | `yfinance` |
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `joblib` |
| **Visualization** | `plotly` (interactive HTML5), `altair` |
| **User Interface** | `streamlit` |
| **IDE / Environment** | VS Code / Antigravity |

---

## 6. Directory Structure
```
stock_market_prediction/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Verified python dependencies
├── README.md                  # Comprehensive documentation & viva guide
│
├── data/
│   ├── raw/                   # Raw downloaded CSV files (AAPL.csv, etc.)
│   └── processed/             # Cleaned & validated CSV files
│
├── models/                    # Saved model weights & scalers (*.pkl)
│
├── src/
│   ├── __init__.py            # Package initializer
│   ├── data_collection.py     # Module 1: Yahoo Finance download logic
│   ├── preprocessing.py       # Module 2: Chronological cleaning & checks
│   ├── feature_engineering.py # Module 3: Lags, SMA, EMA, RSI, MACD, Target
│   ├── model_training.py      # Module 4: Chronological split & multi-model fit
│   ├── prediction.py          # Inference engine & explainability logic
│   ├── evaluation.py          # Module 5: MAE, RMSE, R², Directional Accuracy
│   ├── signals.py             # Trading signal heuristic & risk classification
│   └── visualization.py       # Plotly chart generators
│
└── notebooks/
    └── experimentation.ipynb  # Step-by-step Jupyter notebook for PBL viva
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to Project Directory
```bash
cd "stock_market_prediction"
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. How to Run the Application

Launch the Streamlit web dashboard with a single command:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 9. Example Walkthrough & Demonstration Flow
1. **Sidebar Selection:** Choose `AAPL` (Apple Inc.) or enter a custom symbol (e.g. `MSFT`, `NVDA`).
2. **Date Horizon:** Select a historical window (e.g., 2020-01-01 to Present).
3. **Execution:** Click `🚀 Run Analysis & Prediction`.
4. **Top Cards:** Observe the **Current Close**, **Predicted Close**, **Expected Return %**, **Indicative Signal (BUY/HOLD/SELL)**, and **Market Risk Level**.
5. **Section 1 (Market Profile):** Inspect 52-week High/Low, Today's dollar and percentage change, and daily volume.
6. **Section 2 & 3 (Charts):** Zoom and pan on historical prices with SMA 20/50, and inspect the RSI and MACD indicator tabs.
7. **Section 4 (Model Comparison):** Review the performance table comparing Linear Regression, Random Forest, and XGBoost. The model with the lowest RMSE is highlighted in green.
8. **Section 5 (Out-of-Sample Backtest):** Inspect the Actual vs. Predicted line chart to assess how well the model tracked the real price trajectory across the unseen 20% test period.
9. **Section 7 (Explainability):** Examine which features had the highest percentage importance and read the automated technical market explanation.

---

## 10. How to Demonstrate the Project in a Viva / PBL Review
When demonstrating this project to examiners or project judges, follow this 4-step script:

### 1. Highlight Time-Series Integrity (The Most Common Trap)
> *"Most basic student projects use standard `train_test_split` with `shuffle=True`. In stock forecasting, this causes fatal data leakage because future prices leak into the training set. In this project, we enforced a strict chronological split where the model is evaluated exclusively on unseen future dates."*

### 2. Explain Target Construction
> *"Our target variable is $Close_{t+1}$ created via a $-1$ shift. At prediction time $t$, tomorrow's Open, High, Low, and Volume are unknown, so we use only prior-day information ($t-1$ lags and indicators computed up to day $t$)."*

### 3. Justify Directional Accuracy
> *"In equity markets, minimizing RMSE in dollar terms is not enough. A trader needs to know if tomorrow will close higher or lower than today. Our system measures Directional Accuracy (% of correct UP/DOWN calls), providing a meaningful benchmark for signal generation."*

### 4. Explain the Signal & Risk Framework
> *"We do not blindly trust the regression output. We compare the predicted delta against a configurable threshold ($\pm 1.5\%$) to generate BUY, SELL, or HOLD signals, and qualify that signal by computing the 20-day annualized historical volatility to classify risk as Low, Medium, or High."*

---

## 11. Frequently Asked Questions for Viva Defense

**Q1: Why did Linear Regression perform competitively against Random Forest or XGBoost?**  
*Answer:* In daily stock price prediction, yesterday's close ($Close_{t}$) and moving averages ($SMA_{20}$) are strongly correlated with tomorrow's close ($Close_{t+1}$). Because price levels follow a near-martingale or random walk with drift, linear models capture the baseline autocorrelation effectively, whereas unconstrained tree models can sometimes overfit or plateau when predicting prices outside the training range.

**Q2: What is the difference between Simple Moving Average (SMA) and Exponential Moving Average (EMA)?**  
*Answer:* SMA applies equal weight to all $N$ periods in the window. EMA applies exponentially decreasing weights to older observations, giving greater responsiveness to recent price changes.

**Q3: How is RSI calculated and what do 70 and 30 signify?**  
*Answer:* RSI measures the magnitude of recent gains versus recent losses over 14 periods on a 0 to 100 scale. Values above 70 indicate overbought conditions (potential price exhaustion), while values below 30 indicate oversold conditions (potential buying support).

---

## 12. Project Limitations
1. **Efficient Market Dynamics:** Stock prices are heavily influenced by exogenous shocks (earnings surprises, geopolitical conflicts, interest rate decisions) that are absent from historical price charts.
2. **Stationarity:** Raw asset prices are non-stationary. While lag features and moving averages provide strong baseline tracking, structural macroeconomic regime shifts can degrade model weights.
3. **Slippage & Transaction Costs:** The generated indicative signals do not incorporate broker commissions, bid-ask spreads, or liquidity constraints.

---

## 13. Future Scope
- **NLP Sentiment Integration:** Ingest financial news and Twitter/Reddit sentiment via FinBERT or LLM APIs to combine qualitative news sentiment with quantitative indicators.
- **Deep Sequence Architectures:** Implement LSTM (Long Short-Term Memory), GRU, and Temporal Fusion Transformers (TFT) with attention mechanisms.
- **Multi-Asset Portfolio Allocation:** Expand from single-ticker prediction to multi-asset Markowitz Mean-Variance Portfolio Optimization.
- **Intraday Tick Data:** Integrate WebSockets for real-time order-book and minute-interval algorithmic execution.

---

## 14. Academic Disclaimer
> **⚠️ DISCLAIMER:** This software is an educational Machine Learning project developed exclusively for academic coursework, research, and technical demonstration. It does **NOT** constitute financial, investment, or legal advice. Financial markets involve substantial risk of capital loss. Never execute real-world financial trades based purely on automated statistical forecasts.
