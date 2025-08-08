#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Causal Inference Dashboard
Interactive dashboard for causal analysis and counterfactual predictions
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

from core.causal_inference_engine import (
    get_causal_inference_engine,
    discover_market_causality,
    explain_price_movement,
    CausalMethod,
    InterventionType,
    CausalInferenceConfig
)
from core.data_manager import get_data_manager

class CausalInferenceDashboard:
    """Interactive dashboard for causal inference and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.causal_engine = get_causal_inference_engine()
        self.data_manager = get_data_manager()
        
        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - Causal Inference",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render(self):
        """Render the main dashboard"""
        st.title("🧠 Causal Inference & Counterfactual Analysis")
        st.markdown("Understand WHY market movements happen, not just correlations")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Causal Discovery", 
            "🎯 Movement Explanation", 
            "🔮 Counterfactual Predictions",
            "📊 Causal Graph Visualization",
            "⚙️ Configuration"
        ])
        
        with tab1:
            self._render_causal_discovery_tab()
        
        with tab2:
            self._render_movement_explanation_tab()
        
        with tab3:
            self._render_counterfactual_tab()
        
        with tab4:
            self._render_causal_graph_tab()
        
        with tab5:
            self._render_configuration_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🧠 Causal Analysis Controls")
        
        # Data selection
        try:
            available_coins = self.data_manager.get_supported_coins()
            if available_coins:
                selected_coin = st.sidebar.selectbox(
                    "Select Cryptocurrency",
                    options=available_coins[:50],
                    index=0
                )
                st.session_state.causal_selected_coin = selected_coin
            else:
                st.sidebar.error("No coins available")
                st.session_state.causal_selected_coin = "BTC/EUR"
        except Exception as e:
            st.sidebar.error(f"Error loading coins: {e}")
            st.session_state.causal_selected_coin = "BTC/EUR"
        
        # Timeframe selection
        timeframes = ["1h", "4h", "1d", "1w"]
        selected_timeframe = st.sidebar.selectbox(
            "Analysis Timeframe",
            options=timeframes,
            index=2
        )
        st.session_state.causal_timeframe = selected_timeframe
        
        # Analysis period
        periods = {
            "Last Week": timedelta(weeks=1),
            "Last Month": timedelta(days=30),
            "Last 3 Months": timedelta(days=90),
            "Last 6 Months": timedelta(days=180)
        }
        
        selected_period = st.sidebar.selectbox(
            "Analysis Period",
            options=list(periods.keys()),
            index=2
        )
        st.session_state.causal_period = periods[selected_period]
        
        # Causal methods
        st.sidebar.subheader("🔧 Analysis Methods")
        
        available_methods = [
            "Double Machine Learning",
            "Granger Causality",
            "Difference-in-Differences"
        ]
        
        selected_methods = st.sidebar.multiselect(
            "Causal Methods",
            options=available_methods,
            default=available_methods
        )
        st.session_state.causal_methods = selected_methods
        
        # Statistical settings
        st.sidebar.subheader("📊 Statistical Settings")
        
        significance_level = st.sidebar.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01
        )
        st.session_state.significance_level = significance_level
        
        min_effect_size = st.sidebar.slider(
            "Minimum Effect Size",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001
        )
        st.session_state.min_effect_size = min_effect_size
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        if st.sidebar.button("🔍 Discover Causal Effects", use_container_width=True):
            self._run_causal_discovery()
        
        if st.sidebar.button("📈 Explain Recent Movement", use_container_width=True):
            self._explain_recent_movement()
        
        if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    def _render_causal_discovery_tab(self):
        """Render causal discovery tab"""
        st.header("🔍 Causal Discovery")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load market data
            data = self._load_market_data()
            
            if data is not None and len(data) > 100:
                st.subheader("📊 Market Data Analysis")
                
                # Data summary
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Data Points", len(data))
                
                with metric_col2:
                    st.metric("Features", len(data.columns))
                
                with metric_col3:
                    latest_price = data['close'].iloc[-1] if 'close' in data.columns else 0
                    st.metric("Latest Price", f"€{latest_price:.4f}")
                
                with metric_col4:
                    volatility = data['close'].pct_change().std() * 100 if 'close' in data.columns else 0
                    st.metric("Volatility", f"{volatility:.2f}%")
                
                # Feature engineering for causal analysis
                enhanced_data = self._engineer_causal_features(data)
                
                if st.button("🚀 Run Causal Discovery", use_container_width=True):
                    with st.spinner("Discovering causal relationships..."):
                        try:
                            # Define treatment and outcome variables
                            treatment_vars = [
                                'volume_spike', 'price_momentum', 'volatility_change',
                                'trend_change', 'support_resistance_break'
                            ]
                            
                            outcome_vars = [
                                'price_change', 'future_return_1h', 'future_return_24h',
                                'volatility_future', 'volume_future'
                            ]
                            
                            # Add these variables to enhanced data
                            enhanced_data = self._add_treatment_outcome_vars(enhanced_data, treatment_vars, outcome_vars)
                            
                            # Run causal discovery
                            causal_effects = self.causal_engine.discover_causal_effects(
                                enhanced_data, treatment_vars, outcome_vars
                            )
                            
                            if causal_effects:
                                st.success(f"✅ Discovered {len(causal_effects)} significant causal effects!")
                                
                                # Display results
                                self._display_causal_effects(causal_effects)
                                
                            else:
                                st.warning("No significant causal effects discovered with current settings")
                        
                        except Exception as e:
                            st.error(f"Causal discovery failed: {e}")
                            self.logger.error(f"Causal discovery error: {e}")
            
            else:
                st.warning("Insufficient data for causal analysis. Need at least 100 data points.")
        
        with col2:
            # Causal discovery status
            st.subheader("📈 Discovery Status")
            
            summary = self.causal_engine.get_causal_summary()
            
            st.metric("Effects Discovered", summary.get('total_effects_discovered', 0))
            
            # Method breakdown
            st.subheader("🔧 Methods Used")
            methods_data = summary.get('effects_by_method', {})
            
            for method, count in methods_data.items():
                if count > 0:
                    st.write(f"**{method.replace('_', ' ').title()}:** {count}")
            
            # Strongest effects
            strongest_effects = summary.get('strongest_effects', [])
            if strongest_effects:
                st.subheader("💪 Strongest Effects")
                
                for i, effect in enumerate(strongest_effects[:5]):
                    with st.expander(f"Effect {i+1}: {effect['treatment']} → {effect['outcome']}"):
                        st.write(f"**Effect Size:** {effect['effect_size']:.4f}")
                        st.write(f"**P-value:** {effect['p_value']:.4f}")
                        st.write(f"**Method:** {effect['method'].replace('_', ' ').title()}")
    
    def _render_movement_explanation_tab(self):
        """Render movement explanation tab"""
        st.header("🎯 Price Movement Explanation")
        
        # Load data
        data = self._load_market_data()
        
        if data is not None and len(data) > 50:
            # Recent price movement analysis
            st.subheader("📈 Recent Price Analysis")
            
            recent_data = data.tail(20)
            
            if 'close' in recent_data.columns:
                # Price chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['close'],
                    mode='lines+markers',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Recent Price Movement",
                    xaxis_title="Time",
                    yaxis_title="Price (EUR)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate price change
                start_price = recent_data['close'].iloc[0]
                end_price = recent_data['close'].iloc[-1]
                price_change = (end_price - start_price) / start_price * 100
                
                st.metric("Price Change", f"{price_change:.2f}%")
            
            # Movement explanation
            if st.button("🔍 Explain Movement", use_container_width=True):
                with st.spinner("Analyzing causal factors..."):
                    try:
                        # Engineer features for explanation
                        enhanced_data = self._engineer_causal_features(data)
                        enhanced_data = self._add_treatment_outcome_vars(enhanced_data, [], ['price_change'])
                        
                        # Get explanation
                        explanation = self.causal_engine.explain_movement(enhanced_data, 'price_change')
                        
                        if explanation['confidence'] > 0.1:
                            st.success(f"✅ Movement explained with {explanation['confidence']:.1%} confidence")
                            
                            # Primary causes
                            primary_causes = explanation.get('primary_causes', [])
                            if primary_causes:
                                st.subheader("🎯 Primary Causes")
                                
                                for cause in primary_causes:
                                    with st.expander(f"📊 {cause['cause'].replace('_', ' ').title()}"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write(f"**Effect Size:** {cause['effect_size']:.4f}")
                                            st.write(f"**Contribution:** {cause['contribution_ratio']:.1%}")
                                            st.write(f"**Confidence:** {cause['confidence']:.1%}")
                                        
                                        with col2:
                                            st.write(f"**Method:** {cause['method'].replace('_', ' ').title()}")
                                            st.write(f"**Change:** {cause['treatment_change']:.4f}")
                                            st.write(f"**Lag:** {cause.get('lag', 0)} periods")
                                        
                                        st.write(f"**Mechanism:** {cause.get('mechanism', 'Unknown')}")
                            
                            # Contributing factors
                            contributing_factors = explanation.get('contributing_factors', [])
                            if contributing_factors:
                                st.subheader("📈 Contributing Factors")
                                
                                factors_df = pd.DataFrame([
                                    {
                                        'Factor': factor['cause'].replace('_', ' ').title(),
                                        'Contribution': f"{factor['contribution_ratio']:.1%}",
                                        'Confidence': f"{factor['confidence']:.1%}",
                                        'Effect Size': f"{factor['effect_size']:.4f}"
                                    }
                                    for factor in contributing_factors
                                ])
                                
                                st.dataframe(factors_df, use_container_width=True)
                        
                        else:
                            st.warning("Unable to explain movement with sufficient confidence")
                            st.info("This could indicate the movement was due to random factors or external events not captured in the data")
                    
                    except Exception as e:
                        st.error(f"Movement explanation failed: {e}")
        
        else:
            st.warning("Insufficient data for movement explanation")
    
    def _render_counterfactual_tab(self):
        """Render counterfactual predictions tab"""
        st.header("🔮 Counterfactual Predictions")
        st.markdown("Predict what would happen under different scenarios")
        
        # Load data
        data = self._load_market_data()
        
        if data is not None and len(data) > 50:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎛️ Intervention Settings")
                
                # Select intervention variable
                intervention_vars = [
                    'volume_spike', 'price_momentum', 'volatility_change',
                    'trend_change', 'support_resistance_break'
                ]
                
                selected_intervention = st.selectbox(
                    "Intervention Variable",
                    options=intervention_vars,
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                # Select outcome variable
                outcome_vars = [
                    'price_change', 'future_return_1h', 'future_return_24h'
                ]
                
                selected_outcome = st.selectbox(
                    "Outcome Variable",
                    options=outcome_vars,
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                # Intervention value
                intervention_value = st.slider(
                    "Intervention Value",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1
                )
                
                if st.button("🔮 Predict Counterfactual", use_container_width=True):
                    with st.spinner("Calculating counterfactual outcome..."):
                        try:
                            # Engineer features
                            enhanced_data = self._engineer_causal_features(data)
                            enhanced_data = self._add_treatment_outcome_vars(enhanced_data, [selected_intervention], [selected_outcome])
                            
                            # Predict counterfactual
                            counterfactual = self.causal_engine.predict_counterfactual(
                                enhanced_data, selected_intervention, selected_outcome, intervention_value
                            )
                            
                            if counterfactual is not None:
                                # Current value
                                current_value = enhanced_data[selected_outcome].iloc[-1] if selected_outcome in enhanced_data.columns else 0
                                
                                # Display results
                                st.success("✅ Counterfactual prediction completed")
                                
                                st.metric(
                                    "Predicted Outcome",
                                    f"{counterfactual:.4f}",
                                    delta=f"{counterfactual - current_value:.4f}"
                                )
                                
                                # Store results for visualization
                                st.session_state.counterfactual_result = {
                                    'intervention': selected_intervention,
                                    'outcome': selected_outcome,
                                    'intervention_value': intervention_value,
                                    'current_value': current_value,
                                    'predicted_value': counterfactual,
                                    'change': counterfactual - current_value
                                }
                            
                            else:
                                st.warning("No causal relationship found for this intervention")
                        
                        except Exception as e:
                            st.error(f"Counterfactual prediction failed: {e}")
            
            with col2:
                st.subheader("📊 Counterfactual Results")
                
                # Display stored results
                if 'counterfactual_result' in st.session_state:
                    result = st.session_state.counterfactual_result
                    
                    # Results table
                    results_df = pd.DataFrame([
                        {'Metric': 'Current Value', 'Value': f"{result['current_value']:.4f}"},
                        {'Metric': 'Predicted Value', 'Value': f"{result['predicted_value']:.4f}"},
                        {'Metric': 'Change', 'Value': f"{result['change']:.4f}"},
                        {'Metric': 'Percentage Change', 'Value': f"{(result['change'] / result['current_value'] * 100):.2f}%" if result['current_value'] != 0 else "N/A"}
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Current', 'Counterfactual'],
                        y=[result['current_value'], result['predicted_value']],
                        marker_color=['blue', 'orange'],
                        text=[f"{result['current_value']:.4f}", f"{result['predicted_value']:.4f}"],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"Counterfactual: {result['intervention'].replace('_', ' ').title()} → {result['outcome'].replace('_', ' ').title()}",
                        yaxis_title=result['outcome'].replace('_', ' ').title(),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("Run a counterfactual prediction to see results here")
        
        else:
            st.warning("Insufficient data for counterfactual analysis")
    
    def _render_causal_graph_tab(self):
        """Render causal graph visualization tab"""
        st.header("📊 Causal Graph Visualization")
        
        summary = self.causal_engine.get_causal_summary()
        
        if summary.get('total_effects_discovered', 0) > 0:
            st.subheader("🌐 Causal Network")
            
            # Get causal graph
            causal_graph = self.causal_engine.causal_graph
            
            if causal_graph and len(causal_graph.edges) > 0:
                # Create network visualization
                self._create_causal_network_plot(causal_graph)
                
                # Graph statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nodes", len(causal_graph.nodes))
                
                with col2:
                    st.metric("Causal Relationships", len(causal_graph.edges))
                
                with col3:
                    # Calculate network density
                    max_edges = len(causal_graph.nodes) * (len(causal_graph.nodes) - 1)
                    density = len(causal_graph.edges) / max_edges if max_edges > 0 else 0
                    st.metric("Network Density", f"{density:.2%}")
                
                # Edge details
                st.subheader("🔗 Causal Relationships")
                
                edges_df = pd.DataFrame([
                    {
                        'From': edge[0].replace('_', ' ').title(),
                        'To': edge[1].replace('_', ' ').title(),
                        'Strength': f"{edge[2]:.4f}",
                        'Direction': '→'
                    }
                    for edge in causal_graph.edges
                ])
                
                st.dataframe(edges_df, use_container_width=True)
            
            else:
                st.info("No causal graph available. Run causal discovery first.")
        
        else:
            st.info("No causal effects discovered yet. Use the Causal Discovery tab to find relationships.")
    
    def _render_configuration_tab(self):
        """Render configuration tab"""
        st.header("⚙️ Causal Inference Configuration")
        
        # Current configuration
        config = self.causal_engine.config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistical Settings")
            
            new_significance = st.slider(
                "Significance Level",
                min_value=0.01,
                max_value=0.10,
                value=config.significance_level,
                step=0.01
            )
            
            new_min_effect = st.slider(
                "Minimum Effect Size",
                min_value=0.001,
                max_value=0.05,
                value=config.min_effect_size,
                step=0.001
            )
            
            new_min_evidence = st.slider(
                "Minimum Evidence Strength",
                min_value=0.1,
                max_value=1.0,
                value=config.min_evidence_strength,
                step=0.1
            )
            
            st.subheader("📈 Data Requirements")
            
            new_min_samples = st.number_input(
                "Minimum Samples",
                min_value=50,
                max_value=1000,
                value=config.min_samples,
                step=50
            )
            
            new_lookback = st.number_input(
                "Lookback Periods",
                min_value=20,
                max_value=500,
                value=config.lookback_periods,
                step=10
            )
        
        with col2:
            st.subheader("🔧 Method Settings")
            
            # Available methods
            all_methods = [method.value for method in CausalMethod]
            current_methods = [method.value for method in config.enabled_methods]
            
            new_methods = st.multiselect(
                "Enabled Methods",
                options=all_methods,
                default=current_methods,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            st.subheader("💾 Model Settings")
            
            new_save_models = st.checkbox(
                "Save Models",
                value=config.save_models
            )
            
            new_n_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=config.n_folds
            )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            try:
                # Update configuration
                new_config = CausalInferenceConfig(
                    enabled_methods=[CausalMethod(method) for method in new_methods],
                    min_samples=new_min_samples,
                    lookback_periods=new_lookback,
                    significance_level=new_significance,
                    min_effect_size=new_min_effect,
                    min_evidence_strength=new_min_evidence,
                    save_models=new_save_models,
                    n_folds=new_n_folds
                )
                
                # Reinitialize engine with new config
                self.causal_engine.config = new_config
                
                st.success("✅ Configuration updated successfully!")
                st.rerun()
            
            except Exception as e:
                st.error(f"Configuration update failed: {e}")
        
        # Current status
        st.subheader("📊 Current Status")
        
        status_df = pd.DataFrame([
            {'Setting': 'Significance Level', 'Value': f"{config.significance_level}"},
            {'Setting': 'Min Effect Size', 'Value': f"{config.min_effect_size}"},
            {'Setting': 'Min Evidence Strength', 'Value': f"{config.min_evidence_strength}"},
            {'Setting': 'Min Samples', 'Value': f"{config.min_samples}"},
            {'Setting': 'Lookback Periods', 'Value': f"{config.lookback_periods}"},
            {'Setting': 'Enabled Methods', 'Value': f"{len(config.enabled_methods)}"},
            {'Setting': 'Save Models', 'Value': f"{config.save_models}"}
        ])
        
        st.dataframe(status_df, use_container_width=True)
    
    def _load_market_data(self) -> pd.DataFrame:
        """Load market data for analysis"""
        try:
            coin = st.session_state.get('causal_selected_coin', 'BTC/EUR')
            timeframe = st.session_state.get('causal_timeframe', '1d')
            period = st.session_state.get('causal_period', timedelta(days=90))
            
            end_time = datetime.now()
            start_time = end_time - period
            
            data = self.data_manager.get_historical_data(coin, timeframe, start_time, end_time)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return None
    
    def _engineer_causal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for causal analysis"""
        try:
            enhanced_data = data.copy()
            
            if 'close' in data.columns:
                # Price-based features
                enhanced_data['price_change'] = data['close'].pct_change()
                enhanced_data['price_momentum'] = data['close'].rolling(5).mean() / data['close'].rolling(20).mean() - 1
                enhanced_data['volatility'] = enhanced_data['price_change'].rolling(10).std()
                enhanced_data['volatility_change'] = enhanced_data['volatility'].pct_change()
                
                # Technical indicators
                enhanced_data['rsi'] = self._calculate_rsi(data['close'])
                enhanced_data['macd'] = self._calculate_macd(data['close'])
                
                # Support/resistance
                enhanced_data['support_resistance_break'] = self._detect_breakouts(data['close'])
                
                # Trend detection
                enhanced_data['trend_change'] = self._detect_trend_changes(data['close'])
            
            if 'volume' in data.columns:
                # Volume features
                enhanced_data['volume_change'] = data['volume'].pct_change()
                enhanced_data['volume_spike'] = (data['volume'] / data['volume'].rolling(20).mean() - 1).clip(-1, 3)
                enhanced_data['volume_trend'] = data['volume'].rolling(10).mean() / data['volume'].rolling(30).mean() - 1
            
            # Fill NaN values
            enhanced_data = enhanced_data.fillna(0)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return data
    
    def _add_treatment_outcome_vars(self, data: pd.DataFrame, treatments: List[str], outcomes: List[str]) -> pd.DataFrame:
        """Add treatment and outcome variables to data"""
        try:
            enhanced_data = data.copy()
            
            # Add future returns as outcomes
            if 'close' in data.columns:
                enhanced_data['future_return_1h'] = data['close'].shift(-1) / data['close'] - 1
                enhanced_data['future_return_24h'] = data['close'].shift(-24) / data['close'] - 1
                enhanced_data['volatility_future'] = enhanced_data['price_change'].shift(-5).rolling(5).std()
            
            if 'volume' in data.columns:
                enhanced_data['volume_future'] = data['volume'].shift(-1)
            
            # Ensure all required variables exist
            for var in treatments + outcomes:
                if var not in enhanced_data.columns:
                    enhanced_data[var] = 0.0
            
            return enhanced_data.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Treatment/outcome variable creation failed: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            return ema12 - ema26
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _detect_breakouts(self, prices: pd.Series) -> pd.Series:
        """Detect support/resistance breakouts"""
        try:
            # Simple breakout detection
            rolling_max = prices.rolling(20).max()
            rolling_min = prices.rolling(20).min()
            
            upper_break = (prices > rolling_max.shift(1)).astype(int)
            lower_break = (prices < rolling_min.shift(1)).astype(int) * -1
            
            return upper_break + lower_break
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _detect_trend_changes(self, prices: pd.Series) -> pd.Series:
        """Detect trend changes"""
        try:
            # Simple trend change detection using moving averages
            short_ma = prices.rolling(5).mean()
            long_ma = prices.rolling(20).mean()
            
            trend_signal = (short_ma > long_ma).astype(int)
            trend_change = trend_signal.diff().abs()
            
            return trend_change.fillna(0)
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _display_causal_effects(self, effects: List):
        """Display discovered causal effects"""
        try:
            st.subheader("🔍 Discovered Causal Effects")
            
            for i, effect in enumerate(effects[:10]):  # Show top 10
                with st.expander(f"Effect {i+1}: {effect.treatment} → {effect.outcome}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Effect Size:** {effect.effect_size:.4f}")
                        st.write(f"**P-value:** {effect.p_value:.4f}")
                        st.write(f"**Evidence Strength:** {effect.evidence_strength:.2%}")
                    
                    with col2:
                        st.write(f"**Method:** {effect.method.value.replace('_', ' ').title()}")
                        st.write(f"**Confidence Interval:** [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]")
                        st.write(f"**Temporal Lag:** {effect.temporal_lag} periods")
                    
                    if effect.mechanism:
                        st.write(f"**Mechanism:** {effect.mechanism}")
                    
                    # Visualize effect
                    self._visualize_causal_effect(effect)
        
        except Exception as e:
            st.error(f"Effect display failed: {e}")
    
    def _visualize_causal_effect(self, effect):
        """Visualize individual causal effect"""
        try:
            # Create effect size visualization
            fig = go.Figure()
            
            # Effect size with confidence interval
            fig.add_trace(go.Scatter(
                x=[effect.effect_size],
                y=[effect.treatment],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='Effect Size',
                error_x=dict(
                    type='data',
                    array=[effect.confidence_interval[1] - effect.effect_size],
                    arrayminus=[effect.effect_size - effect.confidence_interval[0]]
                )
            ))
            
            # Add zero line
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Effect")
            
            fig.update_layout(
                title=f"Causal Effect: {effect.treatment} → {effect.outcome}",
                xaxis_title="Effect Size",
                yaxis_title="Treatment",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Effect visualization failed: {e}")
    
    def _create_causal_network_plot(self, causal_graph):
        """Create causal network visualization"""
        try:
            # Create network plot using plotly
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in causal_graph.nodes:
                G.add_node(node)
            
            # Add edges
            for edge in causal_graph.edges:
                G.add_edge(edge[0], edge[1], weight=abs(edge[2]))
            
            # Calculate layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Extract coordinates
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            # Create edge traces
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create plot
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='lightgray'),
                hoverinfo='none',
                mode='lines',
                name='Causal Relationships'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[node.replace('_', ' ').title() for node in G.nodes()],
                textposition="middle center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='black')
                ),
                name='Variables'
            ))
            
            fig.update_layout(
                title="Causal Network Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Arrows show causal direction",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Network visualization failed: {e}")
            self.logger.error(f"Network plot error: {e}")
    
    def _run_causal_discovery(self):
        """Run causal discovery from sidebar"""
        try:
            with st.spinner("Running causal discovery..."):
                data = self._load_market_data()
                
                if data is not None and len(data) > 100:
                    enhanced_data = self._engineer_causal_features(data)
                    
                    treatment_vars = ['volume_spike', 'price_momentum', 'volatility_change']
                    outcome_vars = ['price_change', 'future_return_1h', 'future_return_24h']
                    
                    enhanced_data = self._add_treatment_outcome_vars(enhanced_data, treatment_vars, outcome_vars)
                    
                    effects = self.causal_engine.discover_causal_effects(
                        enhanced_data, treatment_vars, outcome_vars
                    )
                    
                    if effects:
                        st.success(f"✅ Discovered {len(effects)} causal effects!")
                    else:
                        st.warning("No significant effects found")
                    
                    st.rerun()
                else:
                    st.error("Insufficient data for causal discovery")
        
        except Exception as e:
            st.error(f"Causal discovery failed: {e}")
    
    def _explain_recent_movement(self):
        """Explain recent movement from sidebar"""
        try:
            with st.spinner("Explaining recent movement..."):
                data = self._load_market_data()
                
                if data is not None:
                    enhanced_data = self._engineer_causal_features(data)
                    enhanced_data = self._add_treatment_outcome_vars(enhanced_data, [], ['price_change'])
                    
                    explanation = self.causal_engine.explain_movement(enhanced_data, 'price_change')
                    
                    if explanation['confidence'] > 0.1:
                        st.success(f"✅ Movement explained with {explanation['confidence']:.1%} confidence")
                    else:
                        st.warning("Unable to explain recent movement")
                    
                    st.rerun()
                else:
                    st.error("No data available for explanation")
        
        except Exception as e:
            st.error(f"Movement explanation failed: {e}")


def main():
    """Main dashboard entry point"""
    try:
        dashboard = CausalInferenceDashboard()
        dashboard.render()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()