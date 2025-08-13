#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - RL Portfolio Dashboard
Interactive dashboard for reinforcement learning portfolio allocation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reinforcement_portfolio_allocator import (
    get_rl_portfolio_allocator,
    train_rl_allocator,
    get_optimal_allocation,
    evaluate_rl_performance,
)


class RLPortfolioDashboard:
    """Interactive dashboard for RL portfolio allocation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize RL allocator
        self.allocator = get_rl_portfolio_allocator()

    def render(self):
        """Render the main dashboard"""
        st.title("🤖 Reinforcement Learning Portfolio Allocation")
        st.markdown("AI-powered dynamic portfolio optimization using PPO agents")

        # Sidebar controls
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🚀 Training",
                "💼 Portfolio Optimization",
                "📊 Performance Analysis",
                "🎯 Real-time Allocation",
                "📈 Results & Metrics",
            ]
        )

        with tab1:
            self._render_training_tab()

        with tab2:
            self._render_optimization_tab()

        with tab3:
            self._render_performance_tab()

        with tab4:
            self._render_allocation_tab()

        with tab5:
            self._render_results_tab()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🤖 RL Portfolio Controls")

        # Training status
        training_status = "Trained" if self.allocator.is_trained else "Not Trained"
        status_color = "green" if self.allocator.is_trained else "red"
        st.sidebar.markdown(f"**Status:** :{status_color}[{training_status}]")

        st.sidebar.markdown("---")

        # Demo data generation
        st.sidebar.subheader("📊 Demo Data")

        if st.sidebar.button("🎲 Generate Demo Portfolio Data", use_container_width=True):
            demo_data = self._generate_demo_portfolio_data()
            st.session_state["portfolio_demo_data"] = demo_data
            st.sidebar.success("Demo data generated!")

        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")

        if st.sidebar.button("🏋️ Train RL Agent", use_container_width=True):
            if "portfolio_demo_data" in st.session_state:
                with st.spinner("Training RL agent..."):
                    results = train_rl_allocator(
                        st.session_state["portfolio_demo_data"], n_episodes=200
                    )
                    st.session_state["training_results"] = results
                st.sidebar.success("Training completed!")
            else:
                st.sidebar.warning("Generate demo data first!")

        if st.sidebar.button("📊 Get Optimal Allocation", use_container_width=True):
            current_state = {
                "prices": [45000, 2800, 0.85, 15.2, 180],
                "returns": [0.02, 0.015, -0.01, 0.03, 0.008],
                "volatilities": [0.05, 0.06, 0.08, 0.09, 0.04],
                "market_features": [0.03, 0.01],
                "current_allocation": [0.2, 0.2, 0.2, 0.2, 0.2],
                "cash_ratio": 0.0,
                "coin_names": ["BTC", "ETH", "ADA", "DOT", "SOL"],
            }

            allocation = get_optimal_allocation(current_state)
            st.session_state["current_allocation"] = allocation
            st.sidebar.success("Allocation calculated!")

        # Settings
        st.sidebar.subheader("⚙️ RL Settings")

        episodes = st.sidebar.slider("Training Episodes", 100, 2000, 500, 100)
        learning_rate = st.sidebar.selectbox("Learning Rate", [1e-4, 3e-4, 1e-3], index=1)
        risk_tolerance = st.sidebar.slider("Risk Tolerance", 0.1, 2.0, 1.0, 0.1)

    def _render_training_tab(self):
        """Render RL training tab"""
        st.header("🚀 RL Agent Training")
        st.markdown("Train the PPO agent to learn optimal portfolio allocation strategies")

        # Check if training results available
        if "training_results" not in st.session_state:
            st.info("🏋️ Run 'Train RL Agent' from the sidebar to see training results")

            # Explain RL training
            st.subheader("🧠 How RL Portfolio Training Works")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **🎯 Training Process:**
                - Agent learns from market data
                - Explores different allocation strategies
                - Maximizes reward signals
                - Balances return vs. risk
                - Minimizes transaction costs
                """)

            with col2:
                st.markdown("""
                **📊 Reward Function:**
                - Portfolio returns (+)
                - Risk diversification (+)
                - Transaction costs (-)
                - Concentration penalty (-)
                - Sharpe ratio optimization
                """)

            # Architecture overview
            st.subheader("🏗️ PPO Architecture")

            st.markdown("""
            **Actor Network:** Predicts optimal allocation percentages
            - Input: Market state (prices, returns, volatility, indicators)
            - Output: Portfolio allocation weights (sums to 100%)
            - Activation: Softmax for valid probability distribution
            
            **Critic Network:** Estimates value of current state
            - Input: Same market state as actor
            - Output: State value estimation
            - Purpose: Guides actor learning through advantage estimation
            """)

            return

        # Display training results
        results = st.session_state["training_results"]

        if not results.get("success", False):
            st.error(f"Training failed: {results.get('error', 'Unknown error')}")
            return

        # Training metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Episodes", results.get("episodes_trained", 0))

        with col2:
            final_value = results.get("final_portfolio_value", 0)
            st.metric("Final Portfolio Value", f"${final_value:,.2f}")

        with col3:
            sharpe = results.get("final_sharpe_ratio", 0)
            st.metric("Final Sharpe Ratio", f"{sharpe:.4f}")

        with col4:
            avg_reward = results.get("average_reward", 0)
            st.metric("Avg Reward", f"{avg_reward:.4f}")

        # Training progress visualization
        st.subheader("📈 Training Progress")

        training_data = results.get("training_results", {})

        if training_data:
            episodes = training_data.get("episodes", [])
            rewards = training_data.get("rewards", [])
            portfolio_values = training_data.get("portfolio_values", [])
            sharpe_ratios = training_data.get("sharpe_ratios", [])

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Episode Rewards",
                    "Portfolio Value",
                    "Sharpe Ratio",
                    "Learning Progress",
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Episode rewards
            fig.add_trace(
                go.Scatter(
                    x=episodes, y=rewards, mode="lines", name="Rewards", line=dict(color="blue"),
                row=1,
                col=1,
            )

            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=portfolio_values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

            # Sharpe ratio
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=sharpe_ratios,
                    mode="lines",
                    name="Sharpe Ratio",
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )

            # Moving average of rewards (learning progress)
            if len(rewards) > 10:
                ma_rewards = pd.Series(rewards).rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=ma_rewards,
                        mode="lines",
                        name="50-Episode MA",
                        line=dict(color="red"),
                    ),
                    row=2,
                    col=2,
                )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Training statistics
            st.subheader("📊 Training Statistics")

            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.write("**Reward Statistics:**")
                st.write(f"- Max Reward: {max(rewards):.4f}")
                st.write(f"- Min Reward: {min(rewards):.4f}")
                st.write(f"- Final 100 Episodes Avg: {np.mean(rewards[-100:]):.4f}")

            with stats_col2:
                st.write("**Performance Statistics:**")
                st.write(f"- Best Portfolio Value: ${max(portfolio_values):,.2f}")
                st.write(f"- Best Sharpe Ratio: {max(sharpe_ratios):.4f}")
                st.write(
                    f"- Improvement: {((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.1f}%"
                )

        else:
            st.info("No detailed training data available.")

    def _render_optimization_tab(self):
        """Render portfolio optimization tab"""
        st.header("💼 Portfolio Optimization")
        st.markdown("Optimize portfolio allocation using trained RL agent")

        # Current allocation display
        if "current_allocation" in st.session_state:
            allocation = st.session_state["current_allocation"]

            st.subheader("🎯 Optimal Allocation")

            # Allocation metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Confidence", f"{allocation.confidence:.1%}")

            with col2:
                transaction_cost = allocation.transaction_cost
                st.metric("Transaction Cost", f"${transaction_cost:.2f}")

            with col3:
                total_rebalance = np.sum(np.abs(allocation.rebalance_amount))
                st.metric("Rebalance Amount", f"{total_rebalance:.1%}")

            with col4:
                allocation_time = allocation.timestamp.strftime("%H:%M:%S")
                st.metric("Generated At", allocation_time)

            # Allocation visualization
            st.subheader("📊 Asset Allocation")

            coin_names = ["BTC", "ETH", "ADA", "DOT", "SOL"]
            allocations = allocation.target_allocation * 100

            # Pie chart
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=coin_names,
                        values=allocations,
                        hole=0.3,
                        textinfo="label+percent",
                        textposition="inside",
                    )
                ]
            )

            fig_pie.update_layout(title="Recommended Portfolio Allocation", height=400)

            st.plotly_chart(fig_pie, use_container_width=True)

            # Allocation details table
            allocation_data = []
            for i, coin in enumerate(coin_names):
                allocation_data.append(
                    {
                        "Asset": coin,
                        "Target %": f"{allocations[i]:.1f}%",
                        "Rebalance": f"{allocation.rebalance_amount[i]:.1%}",
                        "Action": "Buy"
                        if allocation.rebalance_amount[i] > 0
                        else "Sell"
                        if allocation.rebalance_amount[i] < 0
                        else "Hold",
                    }
                )

            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, use_container_width=True)

            # Reasoning
            st.subheader("💡 Allocation Reasoning")
            st.info(allocation.reasoning)

        else:
            st.info("🎯 Generate optimal allocation from the sidebar to see results")

            # Show sample allocations
            st.subheader("📈 Sample Allocation Strategies")

            strategies = [
                {
                    "name": "Aggressive Growth",
                    "allocation": [40, 30, 15, 10, 5],
                    "description": "High allocation to BTC/ETH for maximum growth potential",
                },
                {
                    "name": "Balanced",
                    "allocation": [25, 25, 20, 15, 15],
                    "description": "Even distribution across top cryptocurrencies",
                },
                {
                    "name": "Conservative",
                    "allocation": [50, 25, 10, 10, 5],
                    "description": "Heavy focus on established cryptocurrencies",
                },
            ]

            for strategy in strategies:
                with st.expander(f"📊 {strategy['name']} Strategy"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(strategy["description"])

                        # Show allocation breakdown
                        coins = ["BTC", "ETH", "ADA", "DOT", "SOL"]
                        for i, coin in enumerate(coins):
                            st.write(f"- {coin}: {strategy['allocation'][i]}%")

                    with col2:
                        # Mini pie chart
                        fig = go.Figure(
                            data=[go.Pie(labels=coins, values=strategy["allocation"], hole=0.3)]
                        )
                        fig.update_layout(height=200, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_tab(self):
        """Render performance analysis tab"""
        st.header("📊 Performance Analysis")
        st.markdown("Analyze RL agent performance and compare strategies")

        # Performance metrics
        st.subheader("📈 Performance Metrics")

        # Simulated performance data
        performance_data = {
            "Strategy": ["RL Agent", "Buy & Hold", "Equal Weight", "Traditional 60/40"],
            "Total Return": [15.2, 8.5, 12.1, 6.8],
            "Sharpe Ratio": [1.45, 0.85, 1.12, 0.76],
            "Max Drawdown": [8.2, 18.5, 12.3, 9.8],
            "Win Rate": [68, 52, 58, 48],
            "Volatility": [12.5, 22.1, 15.8, 8.9],
        }

        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)

        # Performance comparison chart
        st.subheader("📊 Strategy Comparison")

        metrics = ["Total Return", "Sharpe Ratio", "Win Rate"]
        strategies = performance_df["Strategy"].tolist()

        fig = go.Figure()

        for metric in metrics:
            values = performance_df[metric].tolist()
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=strategies,
                    y=values,
                    text=[f"{v:.1f}" for v in values],
                    textposition="auto",
                )

        fig.update_layout(
            title="Performance Comparison Across Strategies",
            xaxis_title="Strategy",
            yaxis_title="Performance Metric",
            barmode="group",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk-Return Analysis
        st.subheader("🎯 Risk-Return Analysis")

        returns = performance_df["Total Return"].tolist()
        volatilities = performance_df["Volatility"].tolist()
        strategies_names = performance_df["Strategy"].tolist()

        fig_scatter = go.Figure()

        fig_scatter.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode="markers+text",
                text=strategies_names,
                textposition="top center",
                marker=dict(size=15, color=["red", "blue", "green", "orange"], opacity=0.7),
                hovertemplate="<b>%{text}</b><br>"
                + "Return: %{y:.1f}%<br>"
                + "Volatility: %{x:.1f}%<extra></extra>",
            )

        fig_scatter.update_layout(
            title="Risk vs Return Analysis",
            xaxis_title="Volatility (%)",
            yaxis_title="Total Return (%)",
            height=400,
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Portfolio evolution
        st.subheader("📈 Portfolio Value Evolution")

        # Generate sample portfolio evolution
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

        rl_returns = np.random.normal(0.0008, 0.02, len(dates))
        buy_hold_returns = np.random.normal(0.0005, 0.025, len(dates))

        rl_values = [100000]
        buy_hold_values = [100000]

        for i in range(1, len(dates)):
            rl_values.append(rl_values[-1] * (1 + rl_returns[i]))
            buy_hold_values.append(buy_hold_values[-1] * (1 + buy_hold_returns[i]))

        fig_evolution = go.Figure()

        fig_evolution.add_trace(
            go.Scatter(
                x=dates, y=rl_values, mode="lines", name="RL Agent", line=dict(color="red", width=2)
        )

        fig_evolution.add_trace(
            go.Scatter(
                x=dates,
                y=buy_hold_values,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="blue", width=2),
            )

        fig_evolution.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
        )

        st.plotly_chart(fig_evolution, use_container_width=True)

    def _render_allocation_tab(self):
        """Render real-time allocation tab"""
        st.header("🎯 Real-time Portfolio Allocation")
        st.markdown("Live portfolio allocation recommendations")

        # Market state input
        st.subheader("📊 Current Market State")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Asset Prices:**")
            btc_price = st.number_input("BTC Price ($)", value=45000.0, step=100.0)
            eth_price = st.number_input("ETH Price ($)", value=2800.0, step=10.0)
            ada_price = st.number_input("ADA Price ($)", value=0.85, step=0.01, format="%.2f")

        with col2:
            st.write("**Market Indicators:**")
            market_volatility = st.slider("Market Volatility", 0.0, 0.1, 0.03, 0.005, format="%.3f")
            market_momentum = st.slider("Market Momentum", -0.05, 0.05, 0.01, 0.005, format="%.3f")
            sentiment_score = st.slider("Sentiment Score", 0.0, 1.0, 0.7, 0.05)

        # Generate allocation
        if st.button("🎯 Generate Allocation Recommendation", use_container_width=True):
            current_state = {
                "prices": [btc_price, eth_price, ada_price, 15.2, 180],
                "returns": [0.02, 0.015, -0.01, 0.03, 0.008],
                "volatilities": [
                    market_volatility,
                    market_volatility * 1.2,
                    market_volatility * 1.5,
                    market_volatility * 1.8,
                    market_volatility * 0.8,
                ],
                "market_features": [market_volatility, market_momentum],
                "current_allocation": [0.2, 0.2, 0.2, 0.2, 0.2],
                "cash_ratio": 0.0,
                "coin_names": ["BTC", "ETH", "ADA", "DOT", "SOL"],
            }

            allocation = get_optimal_allocation(current_state)

            # Display results
            st.subheader("💡 Allocation Recommendation")

            # Quick metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Confidence", f"{allocation.confidence:.1%}")

            with metrics_col2:
                st.metric("Est. Transaction Cost", f"${allocation.transaction_cost:.2f}")

            with metrics_col3:
                top_allocation = max(allocation.target_allocation)
                st.metric("Max Allocation", f"{top_allocation:.1%}")

            # Allocation bars
            coin_names = ["BTC", "ETH", "ADA", "DOT", "SOL"]
            allocations = allocation.target_allocation * 100

            fig_bars = go.Figure(
                data=[
                    go.Bar(
                        x=coin_names,
                        y=allocations,
                        text=[f"{a:.1f}%" for a in allocations],
                        textposition="auto",
                        marker_color=["#F7931A", "#627EEA", "#0033AD", "#E6007A", "#00D4AA"],
                    )
                ]
            )

            fig_bars.update_layout(
                title="Recommended Asset Allocation",
                xaxis_title="Cryptocurrency",
                yaxis_title="Allocation Percentage (%)",
                height=400,
            )

            st.plotly_chart(fig_bars, use_container_width=True)

            # Reasoning
            st.info(f"**Reasoning:** {allocation.reasoning}")

            # Market context
            st.subheader("📈 Market Context")

            context_text = f"""
            Based on current market conditions:
            - Market volatility: {market_volatility:.1%} ({"High" if market_volatility > 0.04 else "Moderate" if market_volatility > 0.02 else "Low"})
            - Market momentum: {market_momentum:+.1%} ({"Bullish" if market_momentum > 0.01 else "Bearish" if market_momentum < -0.01 else "Neutral"})
            - Sentiment: {sentiment_score:.1%} ({"Positive" if sentiment_score > 0.6 else "Negative" if sentiment_score < 0.4 else "Neutral"})
            
            The RL agent recommends this allocation to optimize risk-adjusted returns given current market dynamics.
            """

            st.markdown(context_text)

    def _render_results_tab(self):
        """Render comprehensive results tab"""
        st.header("📈 Comprehensive Results & Metrics")
        st.markdown("Complete analysis of RL portfolio allocation performance")

        # Get allocation summary
        try:
            summary = self.allocator.get_allocation_summary()

            # Overall status
            st.subheader("🎯 System Status")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                training_status = (
                    "✅ Trained" if summary.get("training_status") else "❌ Not Trained"
                )
                st.metric("Training Status", training_status)

            with col2:
                episodes = summary.get("training_episodes", 0)
                st.metric("Training Episodes", episodes)

            with col3:
                evaluations = summary.get("performance_evaluations", 0)
                st.metric("Performance Tests", evaluations)

            with col4:
                last_updated = (
                    summary.get("last_updated", "")[:19] if summary.get("last_updated") else "Never"
                )
                st.metric(
                    "Last Updated",
                    last_updated.split("T")[1] if "T" in last_updated else last_updated,
                )

            # Training results
            training_results = summary.get("training_results", {})
            if training_results:
                st.subheader("🏋️ Training Results")

                train_col1, train_col2, train_col3 = st.columns(3)

                with train_col1:
                    avg_reward = training_results.get("average_recent_reward", 0)
                    st.metric("Average Recent Reward", f"{avg_reward:.4f}")

                with train_col2:
                    best_value = training_results.get("best_portfolio_value", 0)
                    st.metric("Best Portfolio Value", f"${best_value:,.2f}")

                with train_col3:
                    final_value = training_results.get("final_portfolio_value", 0)
                    st.metric("Final Portfolio Value", f"${final_value:,.2f}")

            # Latest performance
            latest_performance = summary.get("latest_performance", {})
            if latest_performance:
                st.subheader("📊 Latest Performance Evaluation")

                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                with perf_col1:
                    returns = latest_performance.get("returns", 0) * 100
                    st.metric("Total Return", f"{returns:.1f}%")

                with perf_col2:
                    sharpe = latest_performance.get("sharpe_ratio", 0)
                    st.metric("Sharpe Ratio", f"{sharpe:.4f}")

                with perf_col3:
                    max_dd = latest_performance.get("max_drawdown", 0) * 100
                    st.metric("Max Drawdown", f"{max_dd:.1f}%")

                with perf_col4:
                    win_rate = latest_performance.get("win_rate", 0) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")

        except Exception as e:
            st.error(f"Error loading results: {e}")

            # Fallback content
            st.info(
                "No comprehensive results available yet. Train the RL agent to see detailed metrics."
            )

        # RL Capabilities
        st.subheader("🤖 RL Capabilities")

        capabilities = [
            "🎯 **PPO-based Learning:** Advanced policy optimization for continuous action spaces",
            "💼 **Dynamic Allocation:** Real-time portfolio rebalancing based on market conditions",
            "📊 **Risk Management:** Automatic position sizing with transaction cost optimization",
            "🎮 **Reward Engineering:** Multi-objective optimization (returns, Sharpe, drawdown)",
            "🧠 **Market Adaptation:** Learning from diverse market regimes and volatility patterns",
            "⚡ **Real-time Inference:** Fast allocation decisions for live trading",
            "📈 **Performance Tracking:** Comprehensive backtesting and performance attribution",
            "🔄 **Continuous Learning:** Online learning capabilities for market adaptation",
        ]

        for capability in capabilities:
            st.markdown(capability)

        # Architecture overview
        st.subheader("🏗️ Technical Architecture")

        with st.expander("View Technical Details"):
            st.markdown("""
            **PPO Actor-Critic Architecture:**
            
            **Actor Network:**
            - Input Layer: Market state vector (prices, returns, volatility, indicators)
            - Hidden Layers: 256 → 256 → 128 neurons (ReLU activation)
            - Output Layer: Softmax allocation probabilities
            - Constraint: Allocations sum to 100%
            
            **Critic Network:**
            - Input Layer: Same market state vector
            - Hidden Layers: 256 → 256 → 128 neurons (ReLU activation)  
            - Output Layer: Single value estimate
            
            **Training Environment:**
            - Action Space: Continuous allocation weights [0,1]^n
            - State Space: Prices, returns, volatilities, technical indicators
            - Reward Function: Portfolio return - risk penalty - transaction costs
            - Episode Length: Variable (market data dependent)
            
            **Key Features:**
            - Cross-fitting for robust learning
            - Risk-aware position sizing
            - Transaction cost modeling
            - Multi-horizon optimization
            - Regime-aware adaptation
            """)

    def _generate_demo_portfolio_data(self) -> pd.DataFrame:
        """Generate demo portfolio data for RL training"""
        np.random.seed(42)

        # Generate 2000 time points (more data for RL training)
        n_points = 2000
        dates = pd.date_range(start="2023-01-01", periods=n_points, freq="H")

        # Generate correlated cryptocurrency prices
        # Bitcoin (base asset)
        btc_returns = np.random.normal(0.0005, 0.025, n_points)
        btc_returns = np.cumsum(btc_returns)
        btc_price = 30000 + btc_returns * 2000

        # Ethereum (correlated with BTC)
        eth_correlation = 0.8
        eth_returns = eth_correlation * btc_returns + np.sqrt(1 - eth_correlation**2) * np.cumsum(
            np.random.normal(0, 0.03, n_points)
        eth_price = 2000 + eth_returns * 150

        # ADA (moderate correlation)
        ada_correlation = 0.6
        ada_returns = ada_correlation * btc_returns + np.sqrt(1 - ada_correlation**2) * np.cumsum(
            np.random.normal(0, 0.04, n_points)
        ada_price = 0.5 + ada_returns * 0.05
        ada_price = np.clip(ada_price, 0.1, 3.0)  # Reasonable bounds

        # DOT (lower correlation)
        dot_correlation = 0.5
        dot_returns = dot_correlation * btc_returns + np.sqrt(1 - dot_correlation**2) * np.cumsum(
            np.random.normal(0, 0.05, n_points)
        dot_price = 10 + dot_returns * 1.5
        dot_price = np.clip(dot_price, 2, 50)

        # SOL (high volatility)
        sol_correlation = 0.4
        sol_returns = sol_correlation * btc_returns + np.sqrt(1 - sol_correlation**2) * np.cumsum(
            np.random.normal(0, 0.06, n_points)
        sol_price = 100 + sol_returns * 20
        sol_price = np.clip(sol_price, 20, 500)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "btc_price": btc_price,
                "eth_price": eth_price,
                "ada_price": ada_price,
                "dot_price": dot_price,
                "sol_price": sol_price,
            }
        )

        # Add market features for RL training
        for col in ["btc_price", "eth_price", "ada_price", "dot_price", "sol_price"]:
            data[f"{col}_return"] = data[col].pct_change()
            data[f"{col}_volatility"] = data[f"{col}_return"].rolling(24).std()

        # Market indicators
        data["market_volatility"] = data[
            ["btc_price_return", "eth_price_return", "ada_price_return"]
        ].std(axis=1)
        data["market_momentum"] = data[
            ["btc_price_return", "eth_price_return", "ada_price_return"]
        ].mean(axis=1)

        # Technical indicators (simplified)
        for col in ["btc_price", "eth_price", "ada_price", "dot_price", "sol_price"]:
            # Simple moving averages
            data[f"{col}_sma_short"] = data[col].rolling(12).mean()
            data[f"{col}_sma_long"] = data[col].rolling(24).mean()

            # RSI approximation
            delta = data[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data[f"{col}_rsi"] = 100 - (100 / (1 + rs))

        # Drop NaN values
        data = data.dropna()

        return data


def main():
    """Main dashboard entry point"""
    try:
        dashboard = RLPortfolioDashboard()
        dashboard.render()

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
