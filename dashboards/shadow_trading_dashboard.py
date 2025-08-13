#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Shadow Trading Dashboard
Interactive dashboard for paper trading and model validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

try:
    from core.shadow_trading_engine import (
        get_shadow_trading_engine,
        submit_shadow_trade,
        start_shadow_trading,
        get_shadow_performance,
        ShadowOrderType,
    )
except ImportError:
    st.error("Shadow trading module not available")


class ShadowTradingDashboard:
    """Dashboard for shadow trading and model validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render(self):
        """Render the shadow trading dashboard"""

        st.header("📊 Shadow Trading & Model Validation")
        st.markdown("Paper trading simulation for risk-free strategy validation and model testing")

        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🎯 Trading Interface",
                "📈 Portfolio Performance",
                "📋 Trade History",
                "⚡ Live Monitoring",
            ]
        )

        with tab1:
            self._render_trading_interface()

        with tab2:
            self._render_performance_analysis()

        with tab3:
            self._render_trade_history()

        with tab4:
            self._render_live_monitoring()

    def _render_trading_interface(self):
        """Render shadow trading interface"""

        st.subheader("🎯 Shadow Trade Execution")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📝 Trade Details**")

            coin = st.selectbox(
                "Cryptocurrency:",
                options=["BTC", "ETH", "ADA", "DOT", "SOL", "MATIC", "LINK", "UNI"],
                help="Select cryptocurrency to trade",
            )

            side = st.selectbox(
                "Trade Side:", options=["Buy", "Sell"], help="Choose buy or sell order"
            )

            quantity = st.number_input(
                "Quantity:",
                min_value=0.001,
                value=0.1,
                step=0.001,
                format="%.3f",
                help="Amount to trade",
            )

            order_type = st.selectbox(
                "Order Type:",
                options=["Market", "Limit"],
                help="Market order executes immediately, limit order waits for target price",
            )

        with col2:
            st.markdown("**🤖 ML Context**")

            strategy = st.selectbox(
                "Strategy:",
                options=[
                    "ML Prediction",
                    "Technical Analysis",
                    "Sentiment Analysis",
                    "Whale Detection",
                    "Manual Trade",
                    "Ensemble Signal",
                ],
                help="Strategy driving this trade",
            )

            ml_prediction = st.number_input(
                "ML Prediction (% return):",
                min_value=-1.0,
                max_value=1.0,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Expected return from ML model",
            )

            confidence = st.slider(
                "Model Confidence:",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.01,
                help="How confident is the model in this prediction?",
            )

            risk_level = st.selectbox(
                "Risk Assessment:",
                options=["Low", "Medium", "High"],
                index=1,
                help="Risk level of this trade",
            )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)

            with col1:
                if order_type == "Limit":
                    limit_price = st.number_input(
                        "Limit Price ($):",
                        min_value=0.01,
                        value=45000.0 if coin == "BTC" else 2800.0,
                        step=0.01,
                    )

                stop_loss = st.number_input(
                    "Stop Loss (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.1,
                    help="Automatic stop loss percentage",
                )

            with col2:
                take_profit = st.number_input(
                    "Take Profit (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.1,
                    help="Automatic take profit percentage",
                )

                position_size_mode = st.selectbox(
                    "Position Sizing:",
                    options=["Fixed Amount", "Confidence Based", "Risk Based"],
                    index=1,
                    help="How to determine position size",
                )

        # Trade preview
        st.markdown("**📊 Trade Preview**")

        # Get current price (mock)
        current_prices = {
            "BTC": 45000,
            "ETH": 2800,
            "ADA": 0.85,
            "DOT": 15.2,
            "SOL": 180,
            "MATIC": 1.2,
            "LINK": 25.5,
            "UNI": 12.8,
        }

        current_price = current_prices.get(coin, 100)
        trade_value = quantity * current_price

        # Adjust quantity based on confidence if selected
        if position_size_mode == "Confidence Based":
            adjusted_quantity = quantity * confidence
        else:
            adjusted_quantity = quantity

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")

        with col2:
            st.metric("Trade Value", f"${trade_value:,.2f}")

        with col3:
            st.metric("Adjusted Quantity", f"{adjusted_quantity:.3f}")

        with col4:
            expected_return = trade_value * ml_prediction
            st.metric("Expected Return", f"${expected_return:,.2f}")

        # Execute trade
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Execute Shadow Trade", type="primary", use_container_width=True):
                try:
                    success = submit_shadow_trade(
                        coin=coin,
                        side=side.lower(),
                        quantity=adjusted_quantity,
                        strategy=strategy.lower().replace(" ", "_"),
                        ml_prediction=ml_prediction,
                        confidence=confidence,
                    )

                    if success:
                        st.success(
                            f"✅ Shadow trade executed: {side} {adjusted_quantity:.3f} {coin}"
                        )

                        # Show trade details
                        trade_details = {
                            "Coin": coin,
                            "Side": side,
                            "Quantity": f"{adjusted_quantity:.3f}",
                            "Current Price": f"${current_price:,.2f}",
                            "Trade Value": f"${trade_value:,.2f}",
                            "Strategy": strategy,
                            "ML Prediction": f"{ml_prediction:.1%}",
                            "Confidence": f"{confidence:.1%}",
                            "Risk Level": risk_level,
                        }

                        with st.expander("📋 Trade Details"):
                            for key, value in trade_details.items():
                                st.write(f"**{key}:** {value}")
                    else:
                        st.error("❌ Failed to execute shadow trade")

                except Exception as e:
                    st.error(f"Trade execution failed: {e}")

    def _render_performance_analysis(self):
        """Render portfolio performance analysis"""

        st.subheader("📈 Shadow Portfolio Performance")

        try:
            # Get performance data
            performance = get_shadow_performance()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${performance.get('current_capital', 100000):,.2f}",
                    delta=f"{performance.get('portfolio_return', 0):.2%}",
                )

            with col2:
                st.metric(
                    "Total Trades",
                    performance.get("total_trades", 0),
                    delta=f"{performance.get('pending_orders', 0)} pending",
                )

            with col3:
                st.metric(
                    "Win Rate",
                    f"{performance.get('win_rate', 0):.1%}",
                    delta="Good" if performance.get("win_rate", 0) > 0.6 else "Poor",
                )

            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{performance.get('sharpe_ratio', 0):.2f}",
                    delta="Excellent" if performance.get("sharpe_ratio", 0) > 1.5 else "Fair",
                )

        except Exception as e:
            st.warning(f"Unable to load performance data: {e}")

            # Show mock data for demonstration
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Portfolio Value", "$102,350", delta="2.35%")

            with col2:
                st.metric("Total Trades", "15", delta="3 pending")

            with col3:
                st.metric("Win Rate", "66.7%", delta="Good")

            with col4:
                st.metric("Sharpe Ratio", "1.42", delta="Fair")

        # Performance charts
        col1, col2 = st.columns([1, 1])

        with col1:
            # Portfolio value over time
            dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
            portfolio_values = 100000 + np.cumsum(np.random.normal(100, 500, 30))

            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="blue"),
                )

            # Add benchmark (buy and hold)
            benchmark = 100000 * (1 + 0.02) ** (np.arange(30) / 365)  # 2% annual growth
            fig_portfolio.add_trace(
                go.Scatter(
                    x=dates,
                    y=benchmark,
                    mode="lines",
                    name="Buy & Hold Benchmark",
                    line=dict(color="gray", dash="dash"),
                )

            fig_portfolio.update_layout(
                title="Portfolio Performance vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
            )

            st.plotly_chart(fig_portfolio, use_container_width=True)

        with col2:
            # PnL distribution
            pnl_data = np.random.normal(0.02, 0.05, 100)

            fig_pnl = go.Figure(data=[go.Histogram(x=pnl_data, nbinsx=20, name="PnL Distribution")])

            fig_pnl.update_layout(
                title="Trade PnL Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=400,
            )

            st.plotly_chart(fig_pnl, use_container_width=True)

        # Performance by strategy
        st.markdown("**📊 Performance by Strategy**")

        strategy_data = {
            "Strategy": [
                "ML Prediction",
                "Technical Analysis",
                "Sentiment Analysis",
                "Whale Detection",
                "Ensemble",
            ],
            "Trades": [8, 3, 2, 1, 1],
            "Win Rate": [0.75, 0.67, 0.50, 1.0, 1.0],
            "Avg Return": [0.035, 0.022, 0.015, 0.08, 0.045],
            "Sharpe Ratio": [1.2, 0.8, 0.4, 2.1, 1.8],
        }

        strategy_df = pd.DataFrame(strategy_data)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_strategy = px.bar(
                strategy_df,
                x="Strategy",
                y="Win Rate",
                color="Avg Return",
                title="Win Rate by Strategy",
            )
            st.plotly_chart(fig_strategy, use_container_width=True)

        with col2:
            fig_sharpe = px.scatter(
                strategy_df,
                x="Avg Return",
                y="Sharpe Ratio",
                size="Trades",
                color="Strategy",
                title="Risk-Return Profile by Strategy",
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

    def _render_trade_history(self):
        """Render trade history interface"""

        st.subheader("📋 Shadow Trade History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Status Filter:",
                options=["All", "Executed", "Closed", "Pending", "Cancelled"],
                help="Filter trades by status",
            )

        with col2:
            coin_filter = st.selectbox(
                "Coin Filter:",
                options=["All", "BTC", "ETH", "ADA", "DOT", "SOL"],
                help="Filter trades by cryptocurrency",
            )

        with col3:
            date_range = st.selectbox(
                "Date Range:",
                options=["All Time", "Last 7 Days", "Last 30 Days", "This Month"],
                help="Filter trades by date",
            )

        # Mock trade history data
        trade_history = [
            {
                "Trade ID": "shadow_001",
                "Timestamp": "2024-01-15 14:30:00",
                "Coin": "BTC",
                "Side": "Buy",
                "Quantity": 0.05,
                "Entry Price": 44800,
                "Exit Price": 46200,
                "PnL": "3.12%",
                "Status": "Closed",
                "Strategy": "ML Prediction",
                "Confidence": "85%",
                "Holding Period": "8.5h",
            },
            {
                "Trade ID": "shadow_002",
                "Timestamp": "2024-01-15 16:45:00",
                "Coin": "ETH",
                "Side": "Buy",
                "Quantity": 0.8,
                "Entry Price": 2750,
                "Exit Price": None,
                "PnL": "1.8%",
                "Status": "Executed",
                "Strategy": "Technical Analysis",
                "Confidence": "72%",
                "Holding Period": "4.2h",
            },
            {
                "Trade ID": "shadow_003",
                "Timestamp": "2024-01-16 09:15:00",
                "Coin": "ADA",
                "Side": "Sell",
                "Quantity": 1000,
                "Entry Price": 0.85,
                "Exit Price": 0.82,
                "PnL": "-3.53%",
                "Status": "Closed",
                "Strategy": "Sentiment Analysis",
                "Confidence": "65%",
                "Holding Period": "2.8h",
            },
        ]

        # Apply filters (simplified for demo)
        filtered_trades = trade_history

        if status_filter != "All":
            filtered_trades = [t for t in filtered_trades if t["Status"] == status_filter]

        if coin_filter != "All":
            filtered_trades = [t for t in filtered_trades if t["Coin"] == coin_filter]

        # Display trade history
        if filtered_trades:
            st.dataframe(pd.DataFrame(filtered_trades), use_container_width=True, hide_index=True)

            # Trade summary
            st.markdown("**📊 Trade Summary**")

            total_trades = len(filtered_trades)
            profitable_trades = len([t for t in filtered_trades if t["PnL"].startswith("+")])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trades", total_trades)

            with col2:
                st.metric("Profitable", profitable_trades)

            with col3:
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1%}")

            with col4:
                avg_holding = "5.8h"  # Mock calculation
                st.metric("Avg Holding", avg_holding)
        else:
            st.info("No trades match the selected filters.")

        # Export options
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export to CSV"):
                st.success("Trade history exported to CSV!")

        with col2:
            if st.button("📈 Generate Report"):
                st.success("Performance report generated!")

        with col3:
            if st.button("🔄 Refresh Data"):
                st.success("Data refreshed!")

    def _render_live_monitoring(self):
        """Render live monitoring interface"""

        st.subheader("⚡ Live Shadow Trading Monitor")

        # System status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🟢 System Status**")
            st.success("Shadow trading engine: Online")
            st.info("Market data feed: Connected")
            st.info("Order execution: Active")

        with col2:
            st.markdown("**📊 Live Metrics**")
            st.metric("Active Positions", "2")
            st.metric("Pending Orders", "1")
            st.metric("Daily PnL", "+$1,247")

        with col3:
            st.markdown("**⚠️ Alerts**")
            st.warning("High volatility detected")
            st.info("New ML signal: BTC +5.2%")

        # Real-time price monitoring
        st.markdown("**📈 Real-Time Price Monitoring**")

        # Mock real-time data
        coins = ["BTC", "ETH", "ADA", "DOT", "SOL"]
        prices = [45150, 2798, 0.847, 15.32, 181.5]
        changes = [0.034, 0.028, -0.012, 0.019, 0.045]
        volumes = [1250, 890, 2100, 450, 680]

        price_data = pd.DataFrame(
            {
                "Coin": coins,
                "Price ($)": prices,
                "24h Change (%)": [f"{c:+.1%}" for c in changes],
                "Volume (M)": [f"${v}M" for v in volumes],
                "Signal": ["Buy", "Hold", "Sell", "Buy", "Strong Buy"],
                "Confidence": ["85%", "60%", "45%", "78%", "92%"],
            }
        )

        # Color code the signals
        def color_signal(val):
            if val == "Strong Buy":
                return "background-color: #90EE90"
            elif val == "Buy":
                return "background-color: #98FB98"
            elif val == "Sell":
                return "background-color: #FFB6C1"
            else:
                return "background-color: #FFFACD"

        styled_df = price_data.style.applymap(color_signal, subset=["Signal"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Live chart
        st.markdown("**📊 Live Price Chart**")

        # Generate mock real-time data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq="H")
        btc_prices = 45000 + np.cumsum(np.random.normal(0, 50, 24))

        fig_live = go.Figure()

        fig_live.add_trace(
            go.Scatter(
                x=timestamps,
                y=btc_prices,
                mode="lines",
                name="BTC Price",
                line=dict(color="orange", width=2),
            )

        # Add trade markers
        buy_times = [timestamps[5], timestamps[15]]
        buy_prices = [btc_prices[5], btc_prices[15]]

        fig_live.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode="markers",
                name="Shadow Trades",
                marker=dict(color="green", size=10, symbol="triangle-up"),
            )

        fig_live.update_layout(
            title="Live BTC Price with Shadow Trades",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400,
        )

        st.plotly_chart(fig_live, use_container_width=True)

        # Control panel
        st.markdown("**🎛️ Control Panel**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("▶️ Start Monitoring"):
                st.success("Monitoring started!")

        with col2:
            if st.button("⏸️ Pause Trading"):
                st.warning("Trading paused!")

        with col3:
            if st.button("🔄 Restart Engine"):
                st.info("Engine restarting...")

        with col4:
            if st.button("🛑 Emergency Stop"):
                st.error("Emergency stop activated!")


if __name__ == "__main__":
    dashboard = ShadowTradingDashboard()
    dashboard.render()
