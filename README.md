# 📊 Quantify — Interactive Market Dashboard with Backtesting & Predictions

Quantify is a **financial analytics dashboard** built using **Streamlit**, **Backtrader**, **Polygon.io**, and **Plotly**.  
It allows users to visualize technical indicators, run trading strategy backtests, and compare predictions versus actual price movements for selected assets.

<img width="1440" height="900" alt="Screenshot 2025-11-20 at 4 40 07 PM" src="https://github.com/user-attachments/assets/f34737e7-b8ad-4d6d-82d7-ac5874fd442e" />

---

## 🚀 Features

### **1. Technical Indicator Visualizations**
The dashboard computes and displays:
- **RSI (14)**
- **MACD (12, 26, 9)**
- **Bollinger Bands (20)**
- Price charts with overlays
- Fully interactive Plotly visualizations

Supported Symbols:
- **SPY** (S&P 500 ETF)
- **GLD** (Gold ETF)
- **EURUSD** (Forex)

---

## 🤖 Backtesting Engine

Quantify includes a powerful backtesting module using **Backtrader**:

### 🔧 Configurable Inputs
- RSI thresholds (overbought/oversold)
- Stake size per trade
- **User-defined starting capital**
- Strategy selection (RSI, Moving Average Crossover, etc.)

### 📈 Performance Metrics
The system calculates and displays:
- **Sharpe Ratio**
- **Cumulative Returns**
- **Maximum Drawdown**
- **Win/Loss Ratio**
- **Portfolio Equity Curve**
- **Trade Signal Chart**
- **Full Trade Log saved as CSV**

---

## 🔮 Prediction Module
A simple benchmark prediction model is included:

- 5-period moving average forecast  
- Actual vs. predicted visualization  
- Error metrics:
  - **MAE**
  - **RMSE**

This module serves as a baseline for future ML enhancements.

---
