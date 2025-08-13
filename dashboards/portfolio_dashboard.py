import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time


class PortfolioDashboard:
    """Portfolio management and risk analysis dashboard"""

    def __init__(self, config_manager, health_monitor):
        self.config_manager = config_manager
        self.health_monitor = health_monitor

    def render(self):
        """Render portfolio dashboard"""
        st.title("💼 Portfolio Dashboard")

        # Portfolio overview
        self._render_portfolio_overview()

        # Risk metrics
        self._render_risk_metrics()

        # Position details
        self._render_position_details()

        # Performance analytics
        self._render_performance_analytics()

        # Trade history and signals
        self._render_trade_history()

        # Risk management settings
        self._render_risk_management()

    def _render_portfolio_overview(self):
        """Render portfolio overview section"""
        st.subheader("📊 Portfolio Overview")

        # Generate sample portfolio data
        portfolio_data = self._generate_portfolio_data()

        # Main portfolio metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_value = portfolio_data["total_value"]
            st.metric(
                "Total Value",
                f"${total_value:,.2f}",
                delta=f"{portfolio_data['total_change']:+.2f}%",
            )

        with col2:
            daily_pnl = portfolio_data["daily_pnl"]
            st.metric(
                "Daily P&L", f"${daily_pnl:+,.2f}", delta=f"{portfolio_data['daily_pnl_pct']:+.2f}%"
            )

        with col3:
            total_positions = len(portfolio_data["positions"])
            st.metric("Total Positions", total_positions)

        with col4:
            cash_balance = portfolio_data["cash_balance"]
            st.metric("Cash Balance", f"${cash_balance:,.2f}")

        with col5:
            portfolio_beta = portfolio_data["portfolio_beta"]
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")

        # Portfolio allocation chart
        self._render_portfolio_allocation(portfolio_data)

    def _generate_portfolio_data(self) -> dict:
        """Generate sample portfolio data"""
        import random

        # Sample positions
        positions = [
            {
                "symbol": "BTC/USD",
                "quantity": 0.5,
                "price": 45000,
                "entry_price": 42000,
                "allocation": 35,
            },
            {
                "symbol": "ETH/USD",
                "quantity": 5.2,
                "price": 3200,
                "entry_price": 3000,
                "allocation": 25,
            },
            {
                "symbol": "ADA/USD",
                "quantity": 1500,
                "price": 0.85,
                "entry_price": 0.80,
                "allocation": 15,
            },
            {
                "symbol": "SOL/USD",
                "quantity": 25,
                "price": 180,
                "entry_price": 170,
                "allocation": 12,
            },
            {"symbol": "DOT/USD", "quantity": 80, "price": 28, "entry_price": 26, "allocation": 8},
            {"symbol": "AVAX/USD", "quantity": 45, "price": 42, "entry_price": 40, "allocation": 5},
        ]

        # Calculate totals
        total_value = sum(pos["quantity"] * pos["price"] for pos in positions)
        total_cost = sum(pos["quantity"] * pos["entry_price"] for pos in positions)
        total_change = ((total_value - total_cost) / total_cost) * 100

        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_change": total_change,
            "daily_pnl": random.uniform(-2000, 5000),
            "daily_pnl_pct": random.uniform(-3, 8),
            "cash_balance": random.uniform(5000, 15000),
            "portfolio_beta": random.uniform(0.8, 1.5),
            "positions": positions,
        }

    def _render_portfolio_allocation(self, portfolio_data: dict):
        """Render portfolio allocation chart"""
        positions = portfolio_data["positions"]

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart for allocation
            symbols = [pos["symbol"] for pos in positions]
            allocations = [pos["allocation"] for pos in positions]

            fig_pie = go.Figure(
                data=[
                    go.Pie(labels=symbols, values=allocations, hole=0.3, textinfo="label+percent")
                ]
            )

            fig_pie.update_layout(title="Portfolio Allocation", height=400)

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Value chart
            values = [pos["quantity"] * pos["price"] for pos in positions]

            fig_bar = go.Figure(data=[go.Bar(x=symbols, y=values, marker_color="lightblue")])

            fig_bar.update_layout(
                title="Position Values", xaxis_title="Symbol", yaxis_title="Value ($)", height=400
            )

            st.plotly_chart(fig_bar, use_container_width=True)

    def _render_risk_metrics(self):
        """Render risk assessment metrics"""
        st.subheader("⚠️ Risk Assessment")

        # Generate risk metrics
        risk_data = self._generate_risk_metrics()

        # Risk metrics grid
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

        with risk_col1:
            var = risk_data["value_at_risk"]
            var_color = "🔴" if var > 5 else "🟡" if var > 3 else "🟢"
            st.metric("Value at Risk (1%)", f"{var_color} {var:.1f}%")

        with risk_col2:
            sharpe = risk_data["sharpe_ratio"]
            sharpe_color = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
            st.metric("Sharpe Ratio", f"{sharpe_color} {sharpe:.2f}")

        with risk_col3:
            max_dd = risk_data["max_drawdown"]
            dd_color = "🔴" if max_dd > 10 else "🟡" if max_dd > 5 else "🟢"
            st.metric("Max Drawdown", f"{dd_color} {max_dd:.1f}%")

        with risk_col4:
            correlation = risk_data["avg_correlation"]
            corr_color = "🔴" if correlation > 0.8 else "🟡" if correlation > 0.6 else "🟢"
            st.metric("Avg Correlation", f"{corr_color} {correlation:.2f}")

        # Risk breakdown
        self._render_risk_breakdown(risk_data)

    def _generate_risk_metrics(self) -> dict:
        """Generate sample risk metrics"""
        import random

        return {
            "value_at_risk": random.uniform(2.5, 8.0),
            "sharpe_ratio": random.uniform(0.8, 2.5),
            "max_drawdown": random.uniform(3.0, 15.0),
            "avg_correlation": random.uniform(0.4, 0.9),
            "volatility": random.uniform(15, 45),
            "beta": random.uniform(0.8, 1.5),
            "risk_score": random.uniform(3, 8),
        }

    def _render_risk_breakdown(self, risk_data: dict):
        """Render detailed risk breakdown"""
        col1, col2 = st.columns(2)

        with col1:
            # Risk score gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_data["risk_score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Risk Score (1-10)"},
                    gauge={
                        "axis": {"range": [None, 10]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 3], "color": "lightgreen"},
                            {"range": [3, 6], "color": "yellow"},
                            {"range": [6, 8], "color": "orange"},
                            {"range": [8, 10], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 7,
                        },
                    },
                )

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Risk metrics radar chart
            categories = ["Volatility", "Correlation", "Concentration", "Beta", "Liquidity"]
            values = [
                min(risk_data["volatility"] / 50 * 10, 10),
                min(risk_data["avg_correlation"] * 10, 10),
                7.5,  # Concentration risk
                min(abs(risk_data["beta"] - 1) * 10, 10),
                6.2,  # Liquidity risk
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(r=values, theta=categories, fill="toself", name="Risk Profile")

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                title="Risk Profile",
                height=300,
            )

            st.plotly_chart(fig_radar, use_container_width=True)

    def _render_position_details(self):
        """Render detailed position information"""
        st.subheader("📈 Position Details")

        portfolio_data = self._generate_portfolio_data()
        positions = portfolio_data["positions"]

        # Create detailed positions table
        position_details = []
        for pos in positions:
            current_value = pos["quantity"] * pos["price"]
            cost_basis = pos["quantity"] * pos["entry_price"]
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis) * 100

            position_details.append(
                {
                    "Symbol": pos["symbol"],
                    "Quantity": f"{pos['quantity']:,.4f}",
                    "Entry Price": f"${pos['entry_price']:,.2f}",
                    "Current Price": f"${pos['price']:,.2f}",
                    "Current Value": f"${current_value:,.2f}",
                    "P&L": f"${pnl:+,.2f}",
                    "P&L %": f"{pnl_pct:+.2f}%",
                    "Allocation": f"{pos['allocation']}%",
                }
            )

        df_positions = pd.DataFrame(position_details)
        st.dataframe(df_positions, use_container_width=True)

        # Position performance chart
        self._render_position_performance(positions)

    def _render_position_performance(self, positions: list):
        """Render position performance chart"""
        symbols = [pos["symbol"] for pos in positions]
        pnl_pcts = [
            ((pos["price"] - pos["entry_price"]) / pos["entry_price"]) * 100 for pos in positions
        ]

        # Color code based on performance
        colors = ["green" if pnl > 0 else "red" for pnl in pnl_pcts]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=symbols,
                    y=pnl_pcts,
                    marker_color=colors,
                    text=[f"{pnl:+.1f}%" for pnl in pnl_pcts],
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title="Position Performance (%)",
            xaxis_title="Symbol",
            yaxis_title="P&L (%)",
            height=400,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_analytics(self):
        """Render performance analytics section"""
        st.subheader("📊 Performance Analytics")

        # Generate performance data
        performance_data = self._generate_performance_data()

        # Performance overview
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Total Return", f"{performance_data['total_return']:+.2f}%")

        with perf_col2:
            st.metric("Annualized Return", f"{performance_data['annualized_return']:+.2f}%")

        with perf_col3:
            st.metric("Win Rate", f"{performance_data['win_rate']:.1f}%")

        with perf_col4:
            st.metric("Profit Factor", f"{performance_data['profit_factor']:.2f}")

        # Performance chart
        self._render_performance_chart(performance_data)

    def _generate_performance_data(self) -> dict:
        """Generate sample performance data"""
        import random

        # Generate historical performance data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        portfolio_values = []

        initial_value = 100000
        current_value = initial_value

        for _ in dates:
            daily_return = random.uniform(-0.03, 0.05)  # -3% to +5% daily
            current_value *= 1 + daily_return
            portfolio_values.append(current_value)

        total_return = ((current_value - initial_value) / initial_value) * 100

        return {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "total_return": total_return,
            "annualized_return": total_return * (365 / 30),
            "win_rate": random.uniform(55, 75),
            "profit_factor": random.uniform(1.2, 2.8),
        }

    def _render_performance_chart(self, performance_data: dict):
        """Render portfolio performance chart"""
        dates = performance_data["dates"]
        values = performance_data["portfolio_values"]

        # Calculate returns
        returns = [(values[i] / values[i - 1] - 1) * 100 for i in range(1, len(values))]
        return_dates = dates[1:]

        # Create subplot
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Portfolio Value", "Daily Returns (%)"),
            vertical_spacing=0.1,
        )

        # Portfolio value line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Daily returns bar chart
        colors = ["green" if r > 0 else "red" for r in returns]
        fig.add_trace(
            go.Bar(x=return_dates, y=returns, name="Daily Returns", marker_color=colors),
            row=2,
            col=1,
        )

        fig.update_layout(height=600, showlegend=False, title="Portfolio Performance Over Time")

        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def _render_trade_history(self):
        """Render trade history and signals"""
        st.subheader("📋 Trade History & Signals")

        # Generate sample trade data
        trades = self._generate_sample_trades()

        # Trade history table
        st.subheader("Recent Trades")
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades, use_container_width=True)

        # Current signals
        st.subheader("Current Trading Signals")
        signals = self._generate_current_signals()

        for signal in signals:
            signal_type = signal["signal"]
            confidence = signal["confidence"]
            symbol = signal["symbol"]

            if signal_type == "buy":
                st.success(f"🟢 **BUY** {symbol} - Confidence: {confidence:.1%}")
            elif signal_type == "sell":
                st.error(f"🔴 **SELL** {symbol} - Confidence: {confidence:.1%}")
            else:
                st.info(f"🟡 **HOLD** {symbol} - Confidence: {confidence:.1%}")

            st.caption(f"Reasoning: {signal['reasoning']}")

    def _generate_sample_trades(self) -> list:
        """Generate sample trade data"""
        import random

        trades = []
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD"]

        for i in range(10):
            symbol = random.choice(symbols)
            trade_type = random.choice(["BUY", "SELL"])
            quantity = random.uniform(0.1, 5.0)
            price = random.uniform(100, 50000)
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 72))

            trades.append(
                {
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Symbol": symbol,
                    "Type": trade_type,
                    "Quantity": f"{quantity:.4f}",
                    "Price": f"${price:,.2f}",
                    "Value": f"${quantity * price:,.2f}",
                    "Status": random.choice(["Completed", "Pending", "Cancelled"]),
                }
            )

        return sorted(trades, key=lambda x: x["Timestamp"], reverse=True)

    def _generate_current_signals(self) -> list:
        """Generate current trading signals"""
        import random

        signals = []
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "AVAX/USD"]

        for symbol in symbols[:5]:
            signal = random.choice(["buy", "sell", "hold"])
            confidence = random.uniform(0.6, 0.95)

            reasoning_options = [
                "Technical indicators showing strong momentum",
                "Positive sentiment analysis results",
                "ML model prediction indicates upward trend",
                "Risk management suggests position reduction",
                "Market correlation analysis suggests opportunity",
            ]

            signals.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": random.choice(reasoning_options),
                }
            )

        return signals

    def _render_risk_management(self):
        """Render risk management settings"""
        with st.expander("⚙️ Risk Management Settings", expanded=False):
            st.subheader("Risk Parameters")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Position Limits")
                max_position_size = st.slider("Max Position Size (%)", 1, 25, 10)
                max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 5, 50, 20)
                max_correlation = st.slider("Max Position Correlation", 0.1, 1.0, 0.7)

                st.subheader("Stop Loss & Take Profit")
                default_stop_loss = st.slider("Default Stop Loss (%)", 1, 20, 5)
                default_take_profit = st.slider("Default Take Profit (%)", 5, 50, 15)

            with col2:
                st.subheader("Risk Monitoring")
                enable_var_monitoring = st.checkbox("Enable VaR Monitoring", value=True)
                var_threshold = st.slider("VaR Alert Threshold (%)", 1, 10, 5)

                enable_drawdown_alerts = st.checkbox("Enable Drawdown Alerts", value=True)
                drawdown_threshold = st.slider("Max Drawdown Threshold (%)", 5, 30, 10)

                st.subheader("Automated Actions")
                auto_rebalance = st.checkbox("Enable Auto-Rebalancing", value=False)
                emergency_stop = st.checkbox("Enable Emergency Stop", value=True)

            if st.button("💾 Save Risk Settings"):
                st.success("Risk management settings saved successfully!")

                # Show updated settings summary
                settings_summary = {
                    "Max Position Size": f"{max_position_size}%",
                    "Max Portfolio Risk": f"{max_portfolio_risk}%",
                    "Default Stop Loss": f"{default_stop_loss}%",
                    "Default Take Profit": f"{default_take_profit}%",
                    "VaR Threshold": f"{var_threshold}%",
                    "Drawdown Threshold": f"{drawdown_threshold}%",
                }

                st.json(settings_summary)
