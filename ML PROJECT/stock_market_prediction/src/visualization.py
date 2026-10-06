"""
MODULE 8: DATA VISUALIZATION
=============================
Produces interactive Plotly visualizations with an academic, professional dark-blue theme:
1. Stock Price & Moving Averages (Close, SMA 20, SMA 50)
2. Technical Oscillators (RSI with 70/30 bands, MACD & Histogram)
3. Trading Volume Chart
4. Actual vs. Predicted Price Comparison on Test Horizon
5. Model Feature Importance Bar Chart
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Palette definition: Navy/Slate Professional Academic Theme
PRIMARY_BLUE = "#1E3A8A"    # Deep Navy
SECONDARY_BLUE = "#3B82F6"  # Royal Blue
ACCENT_CYAN = "#06B6D4"     # Cyan
ACCENT_AMBER = "#F59E0B"    # Amber
ACCENT_GREEN = "#10B981"    # Emerald
ACCENT_RED = "#EF4444"      # Coral Red
BG_PAPER = "#FFFFFF"
BG_PLOT = "#F8FAFC"
GRID_COLOR = "#E2E8F0"
TEXT_COLOR = "#1E293B"


def plot_price_and_sma(df: pd.DataFrame, ticker: str = "STOCK") -> go.Figure:
    """
    Plot historical Close price along with 20-day and 50-day Simple Moving Averages.
    """
    fig = go.Figure()

    # Close price line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name=f"{ticker} Close",
        line=dict(color=PRIMARY_BLUE, width=2.2),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> $%{y:.2f}<extra></extra>"
    ))

    # SMA 20
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_20"],
            mode="lines",
            name="SMA 20 (Short-term)",
            line=dict(color=ACCENT_AMBER, width=1.5, dash="dash"),
            hovertemplate="<b>SMA 20:</b> $%{y:.2f}<extra></extra>"
        ))

    # SMA 50
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_50"],
            mode="lines",
            name="SMA 50 (Medium-term)",
            line=dict(color=ACCENT_CYAN, width=1.5, dash="dot"),
            hovertemplate="<b>SMA 50:</b> $%{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>{ticker} Historical Price & Moving Averages</b>",
        title_font=dict(size=16, color=TEXT_COLOR, family="Arial"),
        xaxis=dict(
            title="Date",
            gridcolor=GRID_COLOR,
            showspikes=True,
            spikethickness=1,
            spikemode="across"
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor=GRID_COLOR,
            tickprefix="$",
            showspikes=True
        ),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )

    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """
    Plot RSI (14) with overbought (70) and oversold (30) reference thresholds.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["RSI_14"],
        mode="lines",
        name="RSI 14",
        line=dict(color="#8B5CF6", width=1.8),
        hovertemplate="<b>RSI:</b> %{y:.2f}<extra></extra>"
    ))

    # Reference lines: Overbought at 70, Oversold at 30, Neutral at 50
    fig.add_hline(y=70, line_dash="dash", line_color=ACCENT_RED, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color=ACCENT_GREEN, annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="#94A3B8")

    fig.update_layout(
        title="<b>Relative Strength Index (RSI 14)</b>",
        title_font=dict(size=15, color=TEXT_COLOR),
        xaxis=dict(title="Date", gridcolor=GRID_COLOR),
        yaxis=dict(title="RSI Value", range=[0, 100], gridcolor=GRID_COLOR),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=40, r=40, t=50, b=30),
        height=260
    )
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    """
    Plot MACD line, Signal line, and MACD Histogram.
    """
    fig = go.Figure()

    # Histogram colors
    hist_colors = [ACCENT_GREEN if val >= 0 else ACCENT_RED for val in df["MACD_Hist"]]

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["MACD_Hist"],
        name="Histogram",
        marker_color=hist_colors,
        opacity=0.6,
        hovertemplate="<b>Histogram:</b> %{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        mode="lines",
        name="MACD",
        line=dict(color=PRIMARY_BLUE, width=1.8),
        hovertemplate="<b>MACD:</b> %{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD_Signal"],
        mode="lines",
        name="Signal Line",
        line=dict(color=ACCENT_AMBER, width=1.6, dash="dash"),
        hovertemplate="<b>Signal:</b> %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>MACD (Moving Average Convergence Divergence)</b>",
        title_font=dict(size=15, color=TEXT_COLOR),
        xaxis=dict(title="Date", gridcolor=GRID_COLOR),
        yaxis=dict(title="Value", gridcolor=GRID_COLOR),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=30),
        height=260
    )
    return fig


def plot_volume(df: pd.DataFrame) -> go.Figure:
    """
    Plot daily trading volume with bars color-coded by daily price direction.
    """
    fig = go.Figure()

    # Green if Close >= Open, Red if Close < Open
    is_up = df["Close"] >= df["Open"]
    colors = [ACCENT_GREEN if up else ACCENT_RED for up in is_up]

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name="Trading Volume",
        marker_color=colors,
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Volume:</b> %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Daily Trading Volume</b>",
        title_font=dict(size=15, color=TEXT_COLOR),
        xaxis=dict(title="Date", gridcolor=GRID_COLOR),
        yaxis=dict(title="Shares Traded", gridcolor=GRID_COLOR),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=40, r=40, t=50, b=30),
        height=240
    )
    return fig


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_actual: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Best Model"
) -> go.Figure:
    """
    Plot actual testing-set closing prices against model predictions on chronological test horizon.
    """
    fig = go.Figure()

    # Actual price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_actual,
        mode="lines",
        name="Actual Next-Day Close",
        line=dict(color=PRIMARY_BLUE, width=2.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> $%{y:.2f}<extra></extra>"
    ))

    # Predicted price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode="lines",
        name=f"Predicted ({model_name})",
        line=dict(color=ACCENT_RED, width=2.0, dash="dash"),
        hovertemplate="<b>Predicted:</b> $%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Actual vs. Predicted Next-Day Close ({model_name} on Out-of-Sample Test Set)</b>",
        title_font=dict(size=16, color=TEXT_COLOR),
        xaxis=dict(title="Chronological Test Date", gridcolor=GRID_COLOR),
        yaxis=dict(title="Price (USD)", tickprefix="$", gridcolor=GRID_COLOR),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Horizontal bar chart showing relative feature importance.
    """
    df_plot = importance_df.head(top_n).iloc[::-1]  # reverse for top-down descending layout

    fig = go.Figure(go.Bar(
        x=df_plot["Percentage"],
        y=df_plot["Feature"],
        orientation="h",
        marker=dict(
            color=df_plot["Percentage"],
            colorscale="Blues",
            line=dict(color=PRIMARY_BLUE, width=1)
        ),
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Relative Feature Importance (% contribution)</b>",
        title_font=dict(size=15, color=TEXT_COLOR),
        xaxis=dict(title="Relative Importance (%)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Engineered Feature"),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        margin=dict(l=40, r=40, t=50, b=30),
        height=320
    )
    return fig
