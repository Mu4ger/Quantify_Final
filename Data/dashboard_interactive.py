import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import talib
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import backtrader as bt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(layout="wide", page_title="Quantify Trading Platform", page_icon="📊")

# -----------------------------
# Configuration
# -----------------------------
API_KEY = os.getenv("POLYGON_API_KEY")
symbols = {"SPY": "stocks", "GLD": "stocks", "EURUSD": "forex"}
cache_dir = "/Users/delger/Desktop/Programmer/Machine Learning/Quantify/Data"
os.makedirs(cache_dir, exist_ok=True)


# -----------------------------
# Fetch historical data
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_historical_data(symbol, market, start_date, end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if market == "forex":
        symbol = f"C:{symbol}"
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if "results" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    df['Date'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


# -----------------------------
# Load data
# -----------------------------
years_back = 5
start_date = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
data_dict = {}
for symbol, market in symbols.items():
    cache_file = os.path.join(cache_dir, f"{symbol}_historical.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=['Date'])
    else:
        df = fetch_historical_data(symbol, market, start_date=start_date)
        if not df.empty:
            df.to_csv(cache_file, index=False)
    data_dict[symbol] = df


# -----------------------------
# Enhanced Trading Strategies
# -----------------------------
class RSI_Strategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.params.rsi_period)
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []
        self.entry_price = None

    def next(self):
        current_price = self.data.close[0]

        # Stop loss and take profit logic
        if self.position:
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct <= -self.params.stop_loss_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Stop Loss)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return
                elif pnl_pct >= self.params.take_profit_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Take Profit)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return

        # Entry/exit signals
        if self.rsi < self.params.rsi_low and not self.position:
            self.buy()
            self.entry_price = current_price
            self.trades.append((self.data.datetime.date(0), "BUY", current_price))
            self.buy_signals.append(current_price)
            self.sell_signals.append(None)
        elif self.rsi > self.params.rsi_high and self.position:
            self.sell()
            self.trades.append((self.data.datetime.date(0), "SELL", current_price))
            self.sell_signals.append(current_price)
            self.buy_signals.append(None)
            self.entry_price = None
        else:
            self.buy_signals.append(None)
            self.sell_signals.append(None)

        self.portfolio_values.append(self.broker.getvalue())


class SMA_Crossover(bt.Strategy):
    params = (
        ('short_period', 20),
        ('long_period', 50),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
    )

    def __init__(self):
        self.sma_short = bt.indicators.SMA(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SMA(self.data.close, period=self.params.long_period)
        self.cross = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []
        self.entry_price = None

    def next(self):
        current_price = self.data.close[0]

        # Stop loss and take profit logic
        if self.position:
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct <= -self.params.stop_loss_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Stop Loss)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return
                elif pnl_pct >= self.params.take_profit_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Take Profit)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return

        if self.cross > 0 and not self.position:
            self.buy()
            self.entry_price = current_price
            self.trades.append((self.data.datetime.date(0), "BUY", current_price))
            self.buy_signals.append(current_price)
            self.sell_signals.append(None)
        elif self.cross < 0 and self.position:
            self.sell()
            self.trades.append((self.data.datetime.date(0), "SELL", current_price))
            self.sell_signals.append(current_price)
            self.buy_signals.append(None)
            self.entry_price = None
        else:
            self.buy_signals.append(None)
            self.sell_signals.append(None)

        self.portfolio_values.append(self.broker.getvalue())


class BollingerBand_Strategy(bt.Strategy):
    params = (
        ('period', 20),
        ('devfactor', 2),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
    )

    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=self.params.period,
                                                   devfactor=self.params.devfactor)
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []
        self.entry_price = None

    def next(self):
        current_price = self.data.close[0]

        # Stop loss and take profit logic
        if self.position:
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct <= -self.params.stop_loss_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Stop Loss)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return
                elif pnl_pct >= self.params.take_profit_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Take Profit)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return

        # Buy when price touches lower band
        if current_price <= self.bbands.lines.bot[0] and not self.position:
            self.buy()
            self.entry_price = current_price
            self.trades.append((self.data.datetime.date(0), "BUY", current_price))
            self.buy_signals.append(current_price)
            self.sell_signals.append(None)
        # Sell when price touches upper band
        elif current_price >= self.bbands.lines.top[0] and self.position:
            self.sell()
            self.trades.append((self.data.datetime.date(0), "SELL", current_price))
            self.sell_signals.append(current_price)
            self.buy_signals.append(None)
            self.entry_price = None
        else:
            self.buy_signals.append(None)
            self.sell_signals.append(None)

        self.portfolio_values.append(self.broker.getvalue())


class MeanReversion_Strategy(bt.Strategy):
    params = (
        ('period', 20),
        ('threshold', 1.5),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.period)
        self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.params.period)
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []
        self.entry_price = None

    def next(self):
        current_price = self.data.close[0]

        # Stop loss and take profit logic
        if self.position:
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct <= -self.params.stop_loss_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Stop Loss)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return
                elif pnl_pct >= self.params.take_profit_pct:
                    self.sell()
                    self.trades.append((self.data.datetime.date(0), "SELL (Take Profit)", current_price))
                    self.sell_signals.append(current_price)
                    self.buy_signals.append(None)
                    self.entry_price = None
                    self.portfolio_values.append(self.broker.getvalue())
                    return

        # Calculate z-score
        if self.stddev[0] != 0:
            z_score = (current_price - self.sma[0]) / self.stddev[0]

            # Buy when price is below mean (oversold)
            if z_score < -self.params.threshold and not self.position:
                self.buy()
                self.entry_price = current_price
                self.trades.append((self.data.datetime.date(0), "BUY", current_price))
                self.buy_signals.append(current_price)
                self.sell_signals.append(None)
            # Sell when price reverts to mean
            elif z_score > 0 and self.position:
                self.sell()
                self.trades.append((self.data.datetime.date(0), "SELL", current_price))
                self.sell_signals.append(current_price)
                self.buy_signals.append(None)
                self.entry_price = None
            else:
                self.buy_signals.append(None)
                self.sell_signals.append(None)
        else:
            self.buy_signals.append(None)
            self.sell_signals.append(None)

        self.portfolio_values.append(self.broker.getvalue())


# -----------------------------
# Volatility-based Position Sizing
# -----------------------------
class VolatilitySizer(bt.Sizer):
    params = (('risk_per_trade', 0.02),)  # Risk 2% per trade

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            # Calculate ATR for volatility
            atr = bt.indicators.ATR(data, period=14)
            if len(atr) > 0 and atr[0] > 0:
                position_risk = cash * self.params.risk_per_trade
                shares = int(position_risk / atr[0])
                return max(shares, 1)
        return 1


# -----------------------------
# Backtesting Function
# -----------------------------
def run_backtest(df, strategy_class, start_cash, stop_loss_pct, take_profit_pct, **kwargs):
    df_bt = df.copy()
    df_bt.set_index('Date', inplace=True)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, **kwargs)
    cerebro.adddata(bt.feeds.PandasData(dataname=df_bt))
    cerebro.broker.setcash(start_cash)
    cerebro.addsizer(VolatilitySizer)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    results = cerebro.run()
    strategy = results[0]

    return strategy, cerebro.broker.getvalue()


# -----------------------------
# Performance Metrics Calculation
# -----------------------------
def calculate_metrics(strategy, start_cash):
    portfolio_values = pd.Series(strategy.portfolio_values)
    returns = portfolio_values.pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod()

    # Sharpe Ratio
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(
        downside_returns) > 0 and downside_returns.std() != 0 else 0

    # Maximum Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = (cum_returns.iloc[-1] - 1) / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win Rate
    trade_log = pd.DataFrame(strategy.trades, columns=["Date", "Action", "Price"])
    trades_with_pnl = []
    buy_price = None

    for _, row in trade_log.iterrows():
        if "BUY" in row['Action']:
            buy_price = row['Price']
        elif "SELL" in row['Action'] and buy_price:
            pnl = row['Price'] - buy_price
            trades_with_pnl.append(pnl)
            buy_price = None

    wins = sum(1 for pnl in trades_with_pnl if pnl > 0)
    total_trades = len(trades_with_pnl)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        'final_value': portfolio_values.iloc[-1],
        'total_return': (portfolio_values.iloc[-1] - start_cash) / start_cash,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'portfolio_values': portfolio_values,
        'returns': returns
    }


# -----------------------------
# Main App
# -----------------------------
st.title("📊 Quantify Trading Platform")
st.markdown("### Professional Multi-Strategy Backtesting & Analysis")

# Sidebar configuration
with st.sidebar:
    st.markdown("Quantify")

    selected_symbol = st.selectbox("📈 Select Asset", list(symbols.keys()), index=0)
    st.markdown("---")

    st.markdown("### ⚙️ Risk Management")
    start_cash = st.number_input("💰 Starting Capital ($)", min_value=1000, max_value=1000000, value=100000, step=1000)
    stop_loss_pct = st.slider("🛡️ Stop Loss (%)", 1, 20, 5) / 100
    take_profit_pct = st.slider("🎯 Take Profit (%)", 5, 50, 10) / 100

    st.markdown("---")
    st.markdown("### 📊 Data Info")
    st.info(f"**Time Period:** {years_back} years\n\n**Update Frequency:** Daily")

# Main content
df = data_dict[selected_symbol].copy()

if df.empty:
    st.error(f"⚠️ No data available for {selected_symbol}")
else:
    # Compute indicators
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['SMA_Short'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_Long'] = talib.SMA(df['Close'], timeperiod=50)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26,
                                                                signalperiod=9)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.dropna(inplace=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Technical Analysis", "🤖 Strategy Backtesting", "⚖️ Strategy Comparison"])

    # -----------------------------
    # Tab 1: Technical Analysis
    # -----------------------------
    with tab1:
        st.markdown("### Market Overview & Indicators")

        # Key metrics
        latest_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = latest_close - prev_close
        change_pct = (change / prev_close) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${latest_close:.2f}", f"{change_pct:+.2f}%")
        with col2:
            st.metric("RSI (14)", f"{df['RSI_14'].iloc[-1]:.2f}",
                      "Oversold" if df['RSI_14'].iloc[-1] < 30 else "Overbought" if df['RSI_14'].iloc[
                                                                                        -1] > 70 else "Neutral")
        with col3:
            st.metric("ATR (14)", f"{df['ATR'].iloc[-1]:.2f}", "Volatility")
        with col4:
            trend = "Bullish" if df['SMA_Short'].iloc[-1] > df['SMA_Long'].iloc[-1] else "Bearish"
            st.metric("Trend", trend, "SMA Cross")

        st.markdown("---")

        # Price chart with indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{selected_symbol} Price Action', 'RSI Indicator', 'MACD')
        )

        # Price + Bollinger Bands + SMA
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00b894',
            decreasing_line_color='#ff7675'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                                 line=dict(dash='dash', color='#ff7675', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Middle'], name='BB Middle',
                                 line=dict(dash='dash', color='#fdcb6e', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                                 line=dict(dash='dash', color='#00b894', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Short'], name='SMA 20',
                                 line=dict(color='#00d4ff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Long'], name='SMA 50',
                                 line=dict(color='#6c5ce7', width=2)), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI',
                                 line=dict(color='#fdcb6e', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff7675", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00b894", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                                 line=dict(color='#00d4ff', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
                                 line=dict(color='#ff7675', width=2)), row=3, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram',
                             marker_color='#6c5ce7'), row=3, col=1)

        fig.update_layout(
            height=900,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Tab 2: Strategy Backtesting
    # -----------------------------
    with tab2:
        st.markdown("### Single Strategy Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:
            strategy_choice = st.radio(
                "Select Strategy",
                ["RSI Strategy", "SMA Crossover", "Bollinger Bands", "Mean Reversion"],
                help="Choose a trading strategy to backtest"
            )

            st.markdown("---")

            # Strategy-specific parameters
            if strategy_choice == "RSI Strategy":
                rsi_low = st.slider("RSI Oversold", 10, 40, 30)
                rsi_high = st.slider("RSI Overbought", 60, 90, 70)
                strategy_params = {'rsi_low': rsi_low, 'rsi_high': rsi_high}
                strategy_class = RSI_Strategy
            elif strategy_choice == "SMA Crossover":
                short_period = st.slider("Short SMA", 5, 50, 20)
                long_period = st.slider("Long SMA", 30, 200, 50)
                strategy_params = {'short_period': short_period, 'long_period': long_period}
                strategy_class = SMA_Crossover
            elif strategy_choice == "Bollinger Bands":
                bb_period = st.slider("BB Period", 10, 50, 20)
                bb_dev = st.slider("Std Deviation", 1.0, 3.0, 2.0, 0.5)
                strategy_params = {'period': bb_period, 'devfactor': bb_dev}
                strategy_class = BollingerBand_Strategy
            else:  # Mean Reversion
                mr_period = st.slider("Lookback Period", 10, 50, 20)
                mr_threshold = st.slider("Z-Score Threshold", 0.5, 3.0, 1.5, 0.25)
                strategy_params = {'period': mr_period, 'threshold': mr_threshold}
                strategy_class = MeanReversion_Strategy

            st.markdown("---")
            run_backtest_btn = st.button("🚀 Run Backtest", use_container_width=True)

        with col2:
            if run_backtest_btn:
                with st.spinner("Running backtest..."):
                    strategy, final_value = run_backtest(
                        df, strategy_class, start_cash,
                        stop_loss_pct, take_profit_pct,
                        **strategy_params
                    )
                    metrics = calculate_metrics(strategy, start_cash)

                    # Performance metrics
                    st.markdown("#### 📊 Performance Metrics")

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Final Portfolio Value", f"${metrics['final_value']:,.2f}",
                                  f"{metrics['total_return'] * 100:+.2f}%")
                    with metric_col2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                                  "Good" if metrics['sharpe_ratio'] > 1 else "Poor")
                    with metric_col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%",
                                  delta_color="inverse")
                    with metric_col4:
                        st.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%",
                                  f"{metrics['total_trades']} trades")

                    metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
                    with metric_col5:
                        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                    with metric_col6:
                        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
                    with metric_col7:
                        profit = metrics['final_value'] - start_cash
                        st.metric("Total Profit", f"${profit:,.2f}")
                    with metric_col8:
                        ann_return = ((metrics['final_value'] / start_cash) ** (252 / len(df)) - 1) * 100
                        st.metric("Annualized Return", f"{ann_return:.2f}%")

                    st.markdown("---")

                    # Equity curve and drawdown
                    fig_equity = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Portfolio Equity Curve', 'Drawdown')
                    )

                    # Equity curve
                    fig_equity.add_trace(
                        go.Scatter(y=metrics['portfolio_values'], mode='lines',
                                   name='Portfolio Value',
                                   line=dict(color='#00d4ff', width=2),
                                   fill='tozeroy',
                                   fillcolor='rgba(0, 212, 255, 0.1)'),
                        row=1, col=1
                    )

                    # Drawdown
                    cum_returns = (1 + metrics['returns']).cumprod()
                    running_max = cum_returns.cummax()
                    drawdown = (cum_returns - running_max) / running_max * 100

                    fig_equity.add_trace(
                        go.Scatter(y=drawdown, mode='lines',
                                   name='Drawdown %',
                                   line=dict(color='#ff7675', width=2),
                                   fill='tozeroy',
                                   fillcolor='rgba(255, 118, 117, 0.2)'),
                        row=2, col=1
                    )

                    fig_equity.update_layout(
                        height=600,
                        template="plotly_dark",
                        showlegend=True,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_equity, use_container_width=True)

                    # Trade signals on price chart
                    st.markdown("#### 📍 Entry & Exit Points")

                    fig_trades = go.Figure()

                    fig_trades.add_trace(go.Candlestick(
                        x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price',
                        increasing_line_color='#00b894',
                        decreasing_line_color='#ff7675'
                    ))

                    # Buy signals
                    buy_dates = [df['Date'].iloc[i] for i, signal in enumerate(strategy.buy_signals) if
                                 signal is not None]
                    buy_prices = [signal for signal in strategy.buy_signals if signal is not None]

                    fig_trades.add_trace(go.Scatter(
                        x=buy_dates, y=buy_prices,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='#00b894', symbol='triangle-up', size=12,
                                    line=dict(color='white', width=1))
                    ))

                    # Sell signals
                    sell_dates = [df['Date'].iloc[i] for i, signal in enumerate(strategy.sell_signals) if
                                  signal is not None]
                    sell_prices = [signal for signal in strategy.sell_signals if signal is not None]

                    fig_trades.add_trace(go.Scatter(
                        x=sell_dates, y=sell_prices,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='#ff7675', symbol='triangle-down', size=12,
                                    line=dict(color='white', width=1))
                    ))

                    fig_trades.update_layout(
                        height=500,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_trades, use_container_width=True)

                    # Trade log
                    st.markdown("#### 📋 Recent Trades")
                    trade_log = pd.DataFrame(strategy.trades, columns=["Date", "Action", "Price"])
                    st.dataframe(
                        trade_log.tail(10).style.set_properties(**{
                            'background-color': '#1a1d29',
                            'color': '#ffffff',
                            'border-color': '#6c5ce7'
                        }),
                        use_container_width=True
                    )

                    # Save trade log
                    trade_file = os.path.join(cache_dir,
                                              f"{selected_symbol}_{strategy_choice.replace(' ', '_')}_trades.csv")
                    trade_log.to_csv(trade_file, index=False)
                    st.success(f"✅ Trade log saved to: {trade_file}")
            else:
                st.info("👈 Configure your strategy parameters and click 'Run Backtest' to begin")

    # -----------------------------
    # Tab 3: Strategy Comparison
    # -----------------------------
    with tab3:
        st.markdown("### Multi-Strategy Performance Comparison")

        compare_btn = st.button("🔄 Compare All Strategies", use_container_width=True)

        if compare_btn:
            with st.spinner("Running comparative analysis on all strategies..."):
                strategies_config = [
                    ("RSI Strategy", RSI_Strategy, {'rsi_low': 30, 'rsi_high': 70}),
                    ("SMA Crossover", SMA_Crossover, {'short_period': 20, 'long_period': 50}),
                    ("Bollinger Bands", BollingerBand_Strategy, {'period': 20, 'devfactor': 2}),
                    ("Mean Reversion", MeanReversion_Strategy, {'period': 20, 'threshold': 1.5})
                ]

                results_comparison = []

                # Run all strategies
                for name, strategy_class, params in strategies_config:
                    strategy, final_value = run_backtest(
                        df, strategy_class, start_cash,
                        stop_loss_pct, take_profit_pct,
                        **params
                    )
                    metrics = calculate_metrics(strategy, start_cash)

                    results_comparison.append({
                        'Strategy': name,
                        'Final Value': metrics['final_value'],
                        'Total Return': metrics['total_return'] * 100,
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Sortino Ratio': metrics['sortino_ratio'],
                        'Max Drawdown': metrics['max_drawdown'] * 100,
                        'Win Rate': metrics['win_rate'] * 100,
                        'Total Trades': metrics['total_trades'],
                        'Calmar Ratio': metrics['calmar_ratio'],
                        'portfolio_values': metrics['portfolio_values']
                    })

                comparison_df = pd.DataFrame(results_comparison)

                # Summary metrics comparison
                st.markdown("#### 📊 Strategy Performance Overview")

                col1, col2, col3, col4 = st.columns(4)

                best_return_idx = comparison_df['Total Return'].idxmax()
                best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
                best_winrate_idx = comparison_df['Win Rate'].idxmax()
                lowest_dd_idx = comparison_df['Max Drawdown'].idxmax()  # Closest to 0

                with col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #00b894, #00d4ff); border-radius: 12px;'>
                        <h4 style='margin:0; color: white;'>🏆 Best Return</h4>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0; color: white; font-weight: bold;'>
                            {comparison_df.iloc[best_return_idx]['Strategy']}
                        </p>
                        <p style='margin:0; color: white;'>{comparison_df.iloc[best_return_idx]['Total Return']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #6c5ce7, #a29bfe); border-radius: 12px;'>
                        <h4 style='margin:0; color: white;'>📈 Best Sharpe</h4>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0; color: white; font-weight: bold;'>
                            {comparison_df.iloc[best_sharpe_idx]['Strategy']}
                        </p>
                        <p style='margin:0; color: white;'>{comparison_df.iloc[best_sharpe_idx]['Sharpe Ratio']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #fdcb6e, #e17055); border-radius: 12px;'>
                        <h4 style='margin:0; color: white;'>🎯 Best Win Rate</h4>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0; color: white; font-weight: bold;'>
                            {comparison_df.iloc[best_winrate_idx]['Strategy']}
                        </p>
                        <p style='margin:0; color: white;'>{comparison_df.iloc[best_winrate_idx]['Win Rate']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #00b894, #55efc4); border-radius: 12px;'>
                        <h4 style='margin:0; color: white;'>🛡️ Lowest Drawdown</h4>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0; color: white; font-weight: bold;'>
                            {comparison_df.iloc[lowest_dd_idx]['Strategy']}
                        </p>
                        <p style='margin:0; color: white;'>{comparison_df.iloc[lowest_dd_idx]['Max Drawdown']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Detailed comparison table
                st.markdown("#### 📋 Detailed Metrics Comparison")

                display_df = comparison_df.drop('portfolio_values', axis=1).copy()
                display_df['Final Value'] = display_df['Final Value'].apply(lambda x: f"${x:,.2f}")
                display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2f}%")
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Sortino Ratio'] = display_df['Sortino Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
                display_df['Calmar Ratio'] = display_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}")

                st.dataframe(display_df, use_container_width=True, height=250)

                # Equity curves comparison
                st.markdown("#### 📈 Equity Curves Comparison")

                fig_comparison = go.Figure()

                colors = ['#00d4ff', '#6c5ce7', '#00b894', '#fdcb6e']

                for i, result in enumerate(results_comparison):
                    fig_comparison.add_trace(go.Scatter(
                        y=result['portfolio_values'],
                        mode='lines',
                        name=result['Strategy'],
                        line=dict(color=colors[i], width=2)
                    ))

                fig_comparison.add_hline(y=start_cash, line_dash="dash",
                                         line_color="white", opacity=0.5,
                                         annotation_text="Starting Capital")

                fig_comparison.update_layout(
                    height=500,
                    template="plotly_dark",
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig_comparison, use_container_width=True)

                # Performance radar chart
                st.markdown("#### 🎯 Multi-Dimensional Performance Analysis")

                fig_radar = go.Figure()

                # Normalize metrics for radar chart (0-100 scale)
                for i, result in enumerate(results_comparison):
                    metrics_normalized = [
                        min(result['Total Return'], 100),  # Cap at 100%
                        min(result['Sharpe Ratio'] * 20, 100),  # Scale Sharpe
                        result['Win Rate'],
                        min((1 + result['Max Drawdown'] / 100) * 100, 100),  # Invert drawdown
                        min(result['Calmar Ratio'] * 15, 100)  # Scale Calmar
                    ]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=metrics_normalized,
                        theta=['Return', 'Sharpe', 'Win Rate', 'Risk Control', 'Calmar'],
                        fill='toself',
                        name=result['Strategy'],
                        line=dict(color=colors[i], width=2)
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100]),
                        bgcolor='rgba(26, 29, 41, 0.5)'
                    ),
                    template="plotly_dark",
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(fig_radar, use_container_width=True)

                # Strategy recommendation
                st.markdown("#### 💡 Strategy Recommendation")

                # Calculate composite score
                for result in results_comparison:
                    score = (
                            result['Total Return'] * 0.3 +
                            result['Sharpe Ratio'] * 20 * 0.25 +
                            result['Win Rate'] * 0.2 +
                            (100 + result['Max Drawdown']) * 0.15 +
                            result['Calmar Ratio'] * 10 * 0.1
                    )
                    result['Composite Score'] = score

                best_strategy = max(results_comparison, key=lambda x: x['Composite Score'])


                st.success(f"""
                **Recommended Strategy: {best_strategy['Strategy']}**

                Based on a weighted composite score considering returns, risk-adjusted performance, 
                and consistency, the {best_strategy['Strategy']} shows the best overall performance 
                for the selected asset ({selected_symbol}).

                - **Total Return:** {best_strategy['Total Return']:.2f}%
                - **Sharpe Ratio:** {best_strategy['Sharpe Ratio']:.2f}
                - **Win Rate:** {best_strategy['Win Rate']:.1f}%
                - **Max Drawdown:** {best_strategy['Max Drawdown']:.2f}%
                """)
        else:
            st.info("👆 Click 'Compare All Strategies' to run a comprehensive analysis across all trading strategies")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a8b2d1; padding: 2rem 0;'>
    <p style='margin: 0;'>⚠️ <strong>Disclaimer:</strong> Past performance does not guarantee future results. This tool is for educational purposes only.</p>
    <p style='margin: 0.5rem 0 0 0;'>Built with Streamlit • Powered by Backtrader & TA-Lib</p>
</div>
""", unsafe_allow_html=True)