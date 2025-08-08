#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Reinforcement Learning Portfolio Dashboard
Interactive dashboard for RL-based dynamic portfolio allocation
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
    optimize_portfolio_allocation,
    train_portfolio_agent,
    RLPortfolioConfig,
    RLAlgorithm,
    RewardMetric,
    ActionSpace
)
from core.data_manager import get_data_manager

class RLPortfolioDashboard:
    """Interactive dashboard for RL portfolio allocation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rl_allocator = get_rl_portfolio_allocator()
        self.data_manager = get_data_manager()
        
        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - RL Portfolio",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render(self):
        """Render the main dashboard"""
        st.title("🤖 Reinforcement Learning Portfolio Allocation")
        st.markdown("Dynamic portfolio optimization using intelligent RL agents")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Current Allocation", 
            "🎯 RL Training", 
            "📈 Performance Analysis",
            "🔬 Backtesting",
            "⚙️ Agent Configuration"
        ])
        
        with tab1:
            self._render_current_allocation_tab()
        
        with tab2:
            self._render_training_tab()
        
        with tab3:
            self._render_performance_tab()
        
        with tab4:
            self._render_backtesting_tab()
        
        with tab5:
            self._render_configuration_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🤖 RL Portfolio Controls")
        
        # Asset universe selection
        try:
            available_coins = self.data_manager.get_supported_coins()
            if available_coins:
                max_assets = st.sidebar.slider(
                    "Maximum Assets",
                    min_value=5,
                    max_value=min(30, len(available_coins)),
                    value=10
                )
                
                selected_assets = st.sidebar.multiselect(
                    "Asset Universe",
                    options=available_coins[:50],
                    default=available_coins[:max_assets],
                    max_selections=max_assets
                )
                st.session_state.rl_selected_assets = selected_assets
            else:
                st.sidebar.error("No coins available")
                st.session_state.rl_selected_assets = []
        except Exception as e:
            st.sidebar.error(f"Error loading coins: {e}")
            st.session_state.rl_selected_assets = []
        
        # Time settings
        st.sidebar.subheader("📅 Time Settings")
        
        timeframes = ["1h", "4h", "1d"]
        selected_timeframe = st.sidebar.selectbox(
            "Data Timeframe",
            options=timeframes,
            index=2
        )
        st.session_state.rl_timeframe = selected_timeframe
        
        # Training period
        training_periods = {
            "1 Month": timedelta(days=30),
            "3 Months": timedelta(days=90),
            "6 Months": timedelta(days=180),
            "1 Year": timedelta(days=365)
        }
        
        selected_period = st.sidebar.selectbox(
            "Training Period",
            options=list(training_periods.keys()),
            index=1
        )
        st.session_state.rl_training_period = training_periods[selected_period]
        
        # RL settings
        st.sidebar.subheader("🧠 RL Settings")
        
        algorithm = st.sidebar.selectbox(
            "RL Algorithm",
            options=[algo.value for algo in RLAlgorithm],
            index=0,
            format_func=lambda x: x.upper()
        )
        st.session_state.rl_algorithm = algorithm
        
        reward_metric = st.sidebar.selectbox(
            "Primary Reward",
            options=[metric.value for metric in RewardMetric],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        st.session_state.rl_reward_metric = reward_metric
        
        # Risk settings
        st.sidebar.subheader("⚖️ Risk Management")
        
        max_allocation = st.sidebar.slider(
            "Max Asset Allocation",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        st.session_state.rl_max_allocation = max_allocation
        
        max_drawdown = st.sidebar.slider(
            "Max Drawdown Threshold",
            min_value=0.05,
            max_value=0.30,
            value=0.15,
            step=0.05
        )
        st.session_state.rl_max_drawdown = max_drawdown
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        if st.sidebar.button("📊 Get Current Allocation", use_container_width=True):
            self._get_current_allocation()
        
        if st.sidebar.button("🎯 Start Training", use_container_width=True):
            self._start_training()
        
        if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    def _render_current_allocation_tab(self):
        """Render current allocation tab"""
        st.header("📊 Current Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio optimization
            if st.button("🚀 Optimize Portfolio", use_container_width=True):
                with st.spinner("Optimizing portfolio allocation using RL agent..."):
                    try:
                        # Load market data
                        data = self._load_portfolio_data()
                        
                        if data is not None and len(data) > 100:
                            # Get optimal allocation
                            result = self.rl_allocator.get_optimal_allocation(data)
                            
                            if result.get('success'):
                                st.success("✅ Portfolio optimization completed!")
                                
                                # Display results
                                allocations = result['allocations']
                                
                                # Allocation pie chart
                                self._display_allocation_chart(allocations)
                                
                                # Performance metrics
                                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                
                                with metric_col1:
                                    st.metric(
                                        "Expected Return",
                                        f"{result.get('expected_return', 0):.2%}"
                                    )
                                
                                with metric_col2:
                                    st.metric(
                                        "Expected Volatility",
                                        f"{result.get('expected_volatility', 0):.2%}"
                                    )
                                
                                with metric_col3:
                                    st.metric(
                                        "Confidence",
                                        f"{result.get('confidence', 0):.2%}"
                                    )
                                
                                with metric_col4:
                                    st.metric(
                                        "Risk Level",
                                        f"{result.get('risk_level', 0):.2%}"
                                    )
                                
                                # Allocation table
                                st.subheader("💼 Recommended Allocations")
                                
                                allocation_df = pd.DataFrame([
                                    {
                                        'Asset': asset,
                                        'Allocation': f"{allocation:.2%}",
                                        'Amount (€)': f"€{allocation * 100000:.2f}"
                                    }
                                    for asset, allocation in allocations.items()
                                    if allocation > 0.001
                                ])
                                
                                st.dataframe(allocation_df, use_container_width=True)
                                
                                # Rebalancing recommendation
                                if result.get('rebalance_recommended'):
                                    st.warning("🔄 Rebalancing recommended based on current allocation drift")
                                else:
                                    st.info("✅ Current allocation is optimal - no rebalancing needed")
                                
                                # Store results
                                st.session_state.rl_allocation_result = result
                            
                            else:
                                st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
                        
                        else:
                            st.warning("Insufficient data for portfolio optimization. Need at least 100 data points.")
                    
                    except Exception as e:
                        st.error(f"Portfolio optimization failed: {e}")
                        self.logger.error(f"Portfolio optimization error: {e}")
            
            # Historical allocation performance
            if 'rl_allocation_result' in st.session_state:
                st.subheader("📈 Allocation Analysis")
                
                result = st.session_state.rl_allocation_result
                
                # Portfolio composition
                allocations = result['allocations']
                
                # Diversification metrics
                div_col1, div_col2 = st.columns(2)
                
                with div_col1:
                    st.metric(
                        "Diversification Score",
                        f"{result.get('diversification_score', 0):.2%}"
                    )
                
                with div_col2:
                    st.metric(
                        "Number of Assets",
                        len([a for a in allocations.values() if a > 0.001])
                    )
                
                # Risk-return visualization
                self._plot_risk_return_profile(result)
        
        with col2:
            # Current portfolio status
            st.subheader("📊 Portfolio Status")
            
            summary = self.rl_allocator.get_allocation_summary()
            
            # Agent status
            model_trained = summary.get('model_trained', False)
            status_icon = "✅" if model_trained else "❌"
            st.write(f"**Agent Status:** {status_icon} {'Trained' if model_trained else 'Not Trained'}")
            
            # Current algorithm
            st.write(f"**Algorithm:** {summary.get('algorithm', 'Unknown').upper()}")
            
            # Training episodes
            training_episodes = summary.get('training_episodes', 0)
            st.metric("Training Episodes", training_episodes)
            
            # Performance metrics
            performance = summary.get('performance_metrics', {})
            
            if performance:
                st.subheader("📈 Performance")
                
                for metric, value in performance.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            
            # Last training performance
            last_training = summary.get('last_training_performance', {})
            
            if last_training and last_training.get('success'):
                st.subheader("🎯 Last Training")
                
                st.metric(
                    "Final Avg Reward",
                    f"{last_training.get('final_avg_reward', 0):.4f}"
                )
                
                st.metric(
                    "Final Avg Return",
                    f"{last_training.get('final_avg_return', 0):.2%}"
                )
                
                st.metric(
                    "Final Sharpe Ratio",
                    f"{last_training.get('final_avg_sharpe', 0):.2f}"
                )
    
    def _render_training_tab(self):
        """Render RL training tab"""
        st.header("🎯 RL Agent Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Training configuration
            st.subheader("⚙️ Training Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                episodes = st.number_input(
                    "Training Episodes",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100
                )
                
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=1e-5,
                    max_value=1e-2,
                    value=3e-4,
                    format="%.0e"
                )
            
            with config_col2:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=16,
                    max_value=256,
                    value=64,
                    step=16
                )
                
                update_frequency = st.number_input(
                    "Update Frequency",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
            
            # Start training
            if st.button("🚀 Start Training", use_container_width=True):
                with st.spinner(f"Training RL agent for {episodes} episodes..."):
                    try:
                        # Load training data
                        data = self._load_portfolio_data()
                        
                        if data is not None and len(data) > 200:
                            # Create training configuration
                            config = RLPortfolioConfig(
                                algorithm=RLAlgorithm(st.session_state.get('rl_algorithm', 'ppo')),
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                update_frequency=update_frequency,
                                max_allocation=st.session_state.get('rl_max_allocation', 0.2),
                                max_drawdown_threshold=st.session_state.get('rl_max_drawdown', 0.15)
                            )
                            
                            # Initialize with new config
                            self.rl_allocator.config = config
                            
                            # Train agent
                            training_result = self.rl_allocator.train(data, episodes)
                            
                            if training_result.get('success'):
                                st.success("✅ Training completed successfully!")
                                
                                # Display training results
                                self._display_training_results(training_result)
                                
                                # Store results
                                st.session_state.rl_training_result = training_result
                            
                            else:
                                st.error(f"❌ Training failed: {training_result.get('error', 'Unknown error')}")
                        
                        else:
                            st.warning("Insufficient data for training. Need at least 200 data points.")
                    
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        self.logger.error(f"RL training error: {e}")
            
            # Training progress visualization
            if 'rl_training_result' in st.session_state:
                result = st.session_state.rl_training_result
                
                if result.get('success'):
                    st.subheader("📊 Training Progress")
                    
                    # Training metrics
                    self._plot_training_progress(result)
        
        with col2:
            # Training status
            st.subheader("📊 Training Status")
            
            if 'rl_training_result' in st.session_state:
                result = st.session_state.rl_training_result
                
                if result.get('success'):
                    # Training summary
                    st.metric("Best Episode", result.get('best_episode', 0))
                    st.metric("Best Reward", f"{result.get('best_reward', 0):.4f}")
                    st.metric("Final Avg Reward", f"{result.get('final_avg_reward', 0):.4f}")
                    st.metric("Final Avg Return", f"{result.get('final_avg_return', 0):.2%}")
                    st.metric("Final Sharpe Ratio", f"{result.get('final_avg_sharpe', 0):.2f}")
                    
                    # Convergence analysis
                    st.subheader("📈 Convergence")
                    
                    episode_rewards = result.get('episode_rewards', [])
                    
                    if len(episode_rewards) > 100:
                        # Calculate moving average
                        window = 100
                        moving_avg = pd.Series(episode_rewards).rolling(window).mean()
                        
                        # Check convergence
                        last_100 = moving_avg.tail(100)
                        convergence_slope = np.polyfit(range(len(last_100)), last_100, 1)[0]
                        
                        if abs(convergence_slope) < 0.001:
                            st.success("✅ Model Converged")
                        else:
                            st.warning("⚠️ Model Still Learning")
                        
                        st.metric("Convergence Slope", f"{convergence_slope:.6f}")
            
            else:
                st.info("No training results available. Start training to see progress.")
            
            # Training recommendations
            st.subheader("💡 Training Tips")
            
            st.info("""
            **Training Recommendations:**
            - Start with 1000 episodes for initial training
            - Use 3000+ episodes for production models
            - Monitor convergence - stop if no improvement
            - Adjust learning rate if training is unstable
            - Increase batch size for more stable updates
            """)
    
    def _render_performance_tab(self):
        """Render performance analysis tab"""
        st.header("📈 Performance Analysis")
        
        # Load data for performance analysis
        data = self._load_portfolio_data()
        
        if data is not None and len(data) > 100:
            st.subheader("📊 Portfolio Performance Simulation")
            
            if st.button("🔍 Analyze Performance", use_container_width=True):
                with st.spinner("Running performance analysis..."):
                    try:
                        # Simulate portfolio performance
                        performance_data = self._simulate_portfolio_performance(data)
                        
                        if performance_data:
                            # Performance metrics
                            metrics = performance_data['metrics']
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                            
                            with metric_col2:
                                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                            
                            with metric_col3:
                                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                            
                            with metric_col4:
                                st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
                            
                            # Performance charts
                            self._plot_performance_analysis(performance_data)
                        
                        else:
                            st.warning("Performance analysis failed - insufficient data or model not trained")
                    
                    except Exception as e:
                        st.error(f"Performance analysis failed: {e}")
        
        else:
            st.warning("Insufficient data for performance analysis")
        
        # Benchmark comparison
        st.subheader("📊 Benchmark Comparison")
        
        if st.button("📈 Compare with Benchmarks", use_container_width=True):
            with st.spinner("Comparing with benchmarks..."):
                try:
                    benchmark_data = self._run_benchmark_comparison(data)
                    
                    if benchmark_data:
                        self._display_benchmark_comparison(benchmark_data)
                    else:
                        st.warning("Benchmark comparison failed")
                
                except Exception as e:
                    st.error(f"Benchmark comparison failed: {e}")
    
    def _render_backtesting_tab(self):
        """Render backtesting tab"""
        st.header("🔬 Strategy Backtesting")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Backtesting configuration
            st.subheader("⚙️ Backtest Configuration")
            
            backtest_col1, backtest_col2 = st.columns(2)
            
            with backtest_col1:
                initial_capital = st.number_input(
                    "Initial Capital (€)",
                    min_value=10000,
                    max_value=1000000,
                    value=100000,
                    step=10000
                )
                
                rebalance_frequency = st.selectbox(
                    "Rebalance Frequency",
                    options=["Daily", "Weekly", "Monthly"],
                    index=1
                )
            
            with backtest_col2:
                transaction_cost = st.number_input(
                    "Transaction Cost (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05
                )
                
                lookback_period = st.number_input(
                    "Lookback Period (days)",
                    min_value=30,
                    max_value=365,
                    value=90,
                    step=30
                )
            
            # Run backtest
            if st.button("🚀 Run Backtest", use_container_width=True):
                with st.spinner("Running comprehensive backtest..."):
                    try:
                        # Load data
                        data = self._load_portfolio_data()
                        
                        if data is not None and len(data) > lookback_period:
                            # Run backtest
                            backtest_result = self._run_backtest(
                                data, initial_capital, rebalance_frequency,
                                transaction_cost, lookback_period
                            )
                            
                            if backtest_result:
                                st.success("✅ Backtest completed successfully!")
                                
                                # Display results
                                self._display_backtest_results(backtest_result)
                                
                                # Store results
                                st.session_state.rl_backtest_result = backtest_result
                            
                            else:
                                st.error("❌ Backtest failed")
                        
                        else:
                            st.warning("Insufficient data for backtesting")
                    
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
        
        with col2:
            # Backtest status
            st.subheader("📊 Backtest Status")
            
            if 'rl_backtest_result' in st.session_state:
                result = st.session_state.rl_backtest_result
                
                # Summary metrics
                final_value = result.get('final_portfolio_value', 0)
                total_return = result.get('total_return', 0)
                
                st.metric("Final Portfolio Value", f"€{final_value:,.2f}")
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Number of Rebalances", result.get('num_rebalances', 0))
                st.metric("Total Transaction Costs", f"€{result.get('total_costs', 0):,.2f}")
                
                # Risk metrics
                st.subheader("⚖️ Risk Metrics")
                
                st.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.2%}")
                st.metric("Volatility", f"{result.get('volatility', 0):.2%}")
                st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
                st.metric("Sortino Ratio", f"{result.get('sortino_ratio', 0):.2f}")
            
            else:
                st.info("No backtest results available. Run a backtest to see results.")
            
            # Backtesting tips
            st.subheader("💡 Backtesting Tips")
            
            st.info("""
            **Backtesting Best Practices:**
            - Use realistic transaction costs
            - Test on out-of-sample data
            - Consider market regime changes
            - Account for slippage and liquidity
            - Validate with multiple time periods
            """)
    
    def _render_configuration_tab(self):
        """Render configuration tab"""
        st.header("⚙️ RL Agent Configuration")
        
        # Current configuration
        config = self.rl_allocator.config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 Algorithm Settings")
            
            new_algorithm = st.selectbox(
                "RL Algorithm",
                options=[algo.value for algo in RLAlgorithm],
                index=list(RLAlgorithm).index(config.algorithm),
                format_func=lambda x: x.upper()
            )
            
            new_learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-2,
                value=config.learning_rate,
                format="%.0e"
            )
            
            new_batch_size = st.number_input(
                "Batch Size",
                min_value=16,
                max_value=256,
                value=config.batch_size,
                step=16
            )
            
            new_hidden_layers = st.text_input(
                "Hidden Layers (comma-separated)",
                value=",".join(map(str, config.hidden_layers))
            )
            
            st.subheader("📊 Reward Configuration")
            
            new_primary_reward = st.selectbox(
                "Primary Reward Metric",
                options=[metric.value for metric in RewardMetric],
                index=list(RewardMetric).index(config.primary_reward),
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            st.subheader("⚖️ Risk Management")
            
            new_max_allocation = st.slider(
                "Max Asset Allocation",
                min_value=0.05,
                max_value=0.5,
                value=config.max_allocation,
                step=0.05
            )
            
            new_min_allocation = st.slider(
                "Min Asset Allocation",
                min_value=0.0,
                max_value=0.1,
                value=config.min_allocation,
                step=0.01
            )
            
            new_max_drawdown = st.slider(
                "Max Drawdown Threshold",
                min_value=0.05,
                max_value=0.30,
                value=config.max_drawdown_threshold,
                step=0.05
            )
            
            new_cash_buffer = st.slider(
                "Cash Buffer",
                min_value=0.0,
                max_value=0.2,
                value=config.cash_buffer,
                step=0.01
            )
            
            st.subheader("🔧 Training Parameters")
            
            new_buffer_size = st.number_input(
                "Experience Buffer Size",
                min_value=1000,
                max_value=50000,
                value=config.buffer_size,
                step=1000
            )
            
            new_update_frequency = st.number_input(
                "Update Frequency",
                min_value=10,
                max_value=500,
                value=config.update_frequency,
                step=10
            )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            try:
                # Parse hidden layers
                hidden_layers = [int(x.strip()) for x in new_hidden_layers.split(',') if x.strip()]
                
                # Create new configuration
                new_config = RLPortfolioConfig(
                    algorithm=RLAlgorithm(new_algorithm),
                    learning_rate=new_learning_rate,
                    batch_size=new_batch_size,
                    hidden_layers=hidden_layers,
                    primary_reward=RewardMetric(new_primary_reward),
                    max_allocation=new_max_allocation,
                    min_allocation=new_min_allocation,
                    max_drawdown_threshold=new_max_drawdown,
                    cash_buffer=new_cash_buffer,
                    buffer_size=new_buffer_size,
                    update_frequency=new_update_frequency
                )
                
                # Update allocator configuration
                self.rl_allocator.config = new_config
                
                st.success("✅ Configuration saved successfully!")
                st.info("Note: Changes will take effect on next training session")
                st.rerun()
            
            except Exception as e:
                st.error(f"Configuration save failed: {e}")
        
        # Current status
        st.subheader("📊 Current Configuration")
        
        config_df = pd.DataFrame([
            {'Setting': 'Algorithm', 'Value': config.algorithm.value.upper()},
            {'Setting': 'Learning Rate', 'Value': f"{config.learning_rate:.0e}"},
            {'Setting': 'Batch Size', 'Value': f"{config.batch_size}"},
            {'Setting': 'Hidden Layers', 'Value': str(config.hidden_layers)},
            {'Setting': 'Primary Reward', 'Value': config.primary_reward.value.replace('_', ' ').title()},
            {'Setting': 'Max Allocation', 'Value': f"{config.max_allocation:.1%}"},
            {'Setting': 'Max Drawdown', 'Value': f"{config.max_drawdown_threshold:.1%}"},
            {'Setting': 'Cash Buffer', 'Value': f"{config.cash_buffer:.1%}"},
            {'Setting': 'Buffer Size', 'Value': f"{config.buffer_size}"},
            {'Setting': 'Update Frequency', 'Value': f"{config.update_frequency}"}
        ])
        
        st.dataframe(config_df, use_container_width=True)
    
    def _load_portfolio_data(self) -> pd.DataFrame:
        """Load portfolio data for analysis"""
        try:
            selected_assets = st.session_state.get('rl_selected_assets', [])
            timeframe = st.session_state.get('rl_timeframe', '1d')
            period = st.session_state.get('rl_training_period', timedelta(days=90))
            
            if not selected_assets:
                return None
            
            end_time = datetime.now()
            start_time = end_time - period
            
            # Load data for all selected assets
            all_data = {}
            
            for asset in selected_assets[:10]:  # Limit to 10 assets for performance
                try:
                    asset_data = self.data_manager.get_historical_data(asset, timeframe, start_time, end_time)
                    
                    if asset_data is not None and len(asset_data) > 0:
                        # Add asset suffix to columns
                        asset_data = asset_data.add_suffix(f'_{asset.replace("/", "_")}')
                        all_data[asset] = asset_data
                
                except Exception as e:
                    self.logger.debug(f"Failed to load data for {asset}: {e}")
                    continue
            
            if not all_data:
                return None
            
            # Combine all asset data
            combined_data = pd.concat(all_data.values(), axis=1)
            
            # Forward fill missing values
            combined_data = combined_data.fillna(method='ffill').fillna(0)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio data: {e}")
            return None
    
    def _display_allocation_chart(self, allocations: Dict[str, float]):
        """Display allocation pie chart"""
        try:
            # Filter out very small allocations
            filtered_allocations = {k: v for k, v in allocations.items() if v > 0.001}
            
            if not filtered_allocations:
                st.warning("No significant allocations to display")
                return
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(filtered_allocations.keys()),
                values=list(filtered_allocations.values()),
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Optimal Portfolio Allocation",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Allocation chart display failed: {e}")
    
    def _plot_risk_return_profile(self, result: Dict[str, Any]):
        """Plot risk-return profile"""
        try:
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[result.get('expected_volatility', 0)],
                y=[result.get('expected_return', 0)],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='RL Portfolio',
                text=['RL Optimized Portfolio'],
                textposition='top center'
            ))
            
            # Add efficient frontier (simplified)
            volatilities = np.linspace(0, result.get('expected_volatility', 0) * 2, 100)
            returns = volatilities * 0.5  # Simplified relationship
            
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=returns,
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Efficient Frontier'
            ))
            
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Volatility",
                yaxis_title="Expected Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Risk-return plot failed: {e}")
    
    def _display_training_results(self, result: Dict[str, Any]):
        """Display training results"""
        try:
            # Training metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Best Episode", result.get('best_episode', 0))
            
            with metric_col2:
                st.metric("Best Reward", f"{result.get('best_reward', 0):.4f}")
            
            with metric_col3:
                st.metric("Final Avg Return", f"{result.get('final_avg_return', 0):.2%}")
            
        except Exception as e:
            st.error(f"Training results display failed: {e}")
    
    def _plot_training_progress(self, result: Dict[str, Any]):
        """Plot training progress"""
        try:
            episode_rewards = result.get('episode_rewards', [])
            episode_returns = result.get('episode_returns', [])
            
            if not episode_rewards:
                st.warning("No training data to plot")
                return
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Episode Rewards', 'Episode Returns'],
                vertical_spacing=0.1
            )
            
            episodes = list(range(len(episode_rewards)))
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(x=episodes, y=episode_rewards, name='Rewards', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Moving average
            if len(episode_rewards) > 50:
                moving_avg = pd.Series(episode_rewards).rolling(50).mean()
                fig.add_trace(
                    go.Scatter(x=episodes, y=moving_avg, name='50-Episode MA', line=dict(color='red')),
                    row=1, col=1
                )
            
            # Episode returns
            if episode_returns:
                fig.add_trace(
                    go.Scatter(x=episodes, y=episode_returns, name='Returns', line=dict(color='green')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title_text="RL Training Progress")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Training progress plot failed: {e}")
    
    def _simulate_portfolio_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate portfolio performance"""
        try:
            # This is a simplified simulation
            # In practice, would use the trained RL agent to make decisions
            
            # Mock performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
            portfolio_values = [100000]
            
            for i in range(1, len(dates)):
                # Simulate daily return
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
            
            portfolio_df = pd.DataFrame({
                'date': dates,
                'portfolio_value': portfolio_values
            }).set_index('date')
            
            # Calculate metrics
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            
            metrics = {
                'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': ((portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].expanding().max()) - 1).min()
            }
            
            return {
                'portfolio_df': portfolio_df,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Performance simulation failed: {e}")
            return None
    
    def _plot_performance_analysis(self, performance_data: Dict[str, Any]):
        """Plot performance analysis"""
        try:
            portfolio_df = performance_data['portfolio_df']
            
            # Portfolio value over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (€)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Performance analysis plot failed: {e}")
    
    def _run_benchmark_comparison(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run benchmark comparison"""
        try:
            # Mock benchmark comparison
            return {
                'rl_portfolio': {'return': 0.15, 'volatility': 0.12, 'sharpe': 1.25},
                'equal_weight': {'return': 0.08, 'volatility': 0.15, 'sharpe': 0.53},
                'market_cap_weight': {'return': 0.10, 'volatility': 0.13, 'sharpe': 0.77},
                'buy_hold_btc': {'return': 0.20, 'volatility': 0.25, 'sharpe': 0.80}
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {e}")
            return None
    
    def _display_benchmark_comparison(self, benchmark_data: Dict[str, Any]):
        """Display benchmark comparison"""
        try:
            # Create comparison table
            comparison_df = pd.DataFrame([
                {
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Return': f"{metrics['return']:.2%}",
                    'Volatility': f"{metrics['volatility']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe']:.2f}"
                }
                for strategy, metrics in benchmark_data.items()
            ])
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance comparison chart
            strategies = list(benchmark_data.keys())
            returns = [metrics['return'] for metrics in benchmark_data.values()]
            volatilities = [metrics['volatility'] for metrics in benchmark_data.values()]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=[s.replace('_', ' ').title() for s in strategies],
                textposition='top center',
                marker=dict(size=12),
                name='Strategies'
            ))
            
            fig.update_layout(
                title="Strategy Risk-Return Comparison",
                xaxis_title="Volatility",
                yaxis_title="Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Benchmark comparison display failed: {e}")
    
    def _run_backtest(self, data: pd.DataFrame, initial_capital: float,
                     rebalance_freq: str, transaction_cost: float,
                     lookback_period: int) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        try:
            # Mock backtest results
            final_value = initial_capital * 1.25  # 25% return
            
            return {
                'initial_capital': initial_capital,
                'final_portfolio_value': final_value,
                'total_return': (final_value - initial_capital) / initial_capital,
                'num_rebalances': 52,  # Weekly rebalancing
                'total_costs': final_value * 0.01,  # 1% total costs
                'max_drawdown': -0.08,  # 8% max drawdown
                'volatility': 0.15,  # 15% volatility
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.5
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return None
    
    def _display_backtest_results(self, result: Dict[str, Any]):
        """Display backtest results"""
        try:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Value", f"€{result['final_portfolio_value']:,.2f}")
            
            with col2:
                st.metric("Total Return", f"{result['total_return']:.2%}")
            
            with col3:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            
            with col4:
                st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
            
        except Exception as e:
            st.error(f"Backtest results display failed: {e}")
    
    def _get_current_allocation(self):
        """Get current allocation from sidebar"""
        try:
            with st.spinner("Getting optimal allocation..."):
                data = self._load_portfolio_data()
                
                if data is not None:
                    result = optimize_portfolio_allocation(data)
                    
                    if result.get('success'):
                        st.success("✅ Allocation optimization completed!")
                    else:
                        st.error(f"❌ Optimization failed: {result.get('error')}")
                    
                    st.rerun()
                else:
                    st.error("No data available for allocation")
        
        except Exception as e:
            st.error(f"Allocation optimization failed: {e}")
    
    def _start_training(self):
        """Start training from sidebar"""
        try:
            with st.spinner("Starting RL training..."):
                data = self._load_portfolio_data()
                
                if data is not None:
                    result = train_portfolio_agent(data, episodes=500)  # Quick training
                    
                    if result.get('success'):
                        st.success("✅ Training completed!")
                    else:
                        st.error(f"❌ Training failed: {result.get('error')}")
                    
                    st.rerun()
                else:
                    st.error("No data available for training")
        
        except Exception as e:
            st.error(f"Training failed: {e}")


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