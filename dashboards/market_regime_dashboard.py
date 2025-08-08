#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Market Regime Detection Dashboard
Interactive dashboard for monitoring market regime detection and adaptive model switching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_regime_detector import (
    get_market_regime_detector,
    MarketRegime,
    DetectionMethod,
    detect_current_regime
)
from core.adaptive_model_switcher import (
    get_adaptive_model_switcher,
    ModelType,
    AdaptationStrategy,
    adapt_to_current_regime
)
from core.data_manager import get_data_manager

class MarketRegimeDashboard:
    """Interactive dashboard for market regime detection and adaptive switching"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.regime_detector = get_market_regime_detector()
        self.adaptive_switcher = get_adaptive_model_switcher()
        self.data_manager = get_data_manager()
        
        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - Market Regime Detection",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render(self):
        """Render the main dashboard"""
        st.title("🌍 Market Regime Detection & Adaptive Model Switching")
        st.markdown("Real-time regime detection with automatic model adaptation")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Regime Detection", 
            "🔄 Adaptive Switching", 
            "📊 Performance Analysis",
            "⚙️ Model Configuration",
            "📈 Real-time Monitoring"
        ])
        
        with tab1:
            self._render_regime_detection_tab()
        
        with tab2:
            self._render_adaptive_switching_tab()
        
        with tab3:
            self._render_performance_analysis_tab()
        
        with tab4:
            self._render_model_configuration_tab()
        
        with tab5:
            self._render_realtime_monitoring_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🌍 Regime Detection Controls")
        
        # Coin selection
        try:
            available_coins = self.data_manager.get_supported_coins()
            if available_coins:
                selected_coin = st.sidebar.selectbox(
                    "Select Cryptocurrency",
                    options=available_coins[:50],  # Limit for performance
                    index=0
                )
                st.session_state.selected_coin = selected_coin
            else:
                st.sidebar.error("No coins available")
                st.session_state.selected_coin = "BTC/EUR"
        except Exception as e:
            st.sidebar.error(f"Error loading coins: {e}")
            st.session_state.selected_coin = "BTC/EUR"
        
        # Timeframe selection
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        selected_timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=timeframes,
            index=timeframes.index("1h")
        )
        st.session_state.selected_timeframe = selected_timeframe
        
        # Data period
        data_periods = {
            "Last 6 Hours": timedelta(hours=6),
            "Last Day": timedelta(days=1),
            "Last 3 Days": timedelta(days=3),
            "Last Week": timedelta(weeks=1),
            "Last Month": timedelta(days=30)
        }
        
        selected_period = st.sidebar.selectbox(
            "Analysis Period",
            options=list(data_periods.keys()),
            index=1
        )
        st.session_state.data_period = data_periods[selected_period]
        
        # Detection method
        st.sidebar.subheader("🔧 Detection Settings")
        
        detection_methods = ["ENSEMBLE", "STATISTICAL", "CLUSTERING", "PCA_BASED"]
        selected_method = st.sidebar.selectbox(
            "Detection Method",
            options=detection_methods,
            index=0
        )
        st.session_state.detection_method = selected_method
        
        # Adaptation strategy
        adaptation_strategies = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE", "DYNAMIC"]
        selected_strategy = st.sidebar.selectbox(
            "Adaptation Strategy",
            options=adaptation_strategies,
            index=1
        )
        st.session_state.adaptation_strategy = selected_strategy
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
        
        if st.sidebar.button("🎯 Detect Current Regime", use_container_width=True):
            self._detect_current_regime()
        
        if st.sidebar.button("⚡ Force Model Adaptation", use_container_width=True):
            self._force_model_adaptation()
        
        if st.sidebar.button("🔍 Train Detection Models", use_container_width=True):
            self._train_detection_models()
    
    def _render_regime_detection_tab(self):
        """Render regime detection tab"""
        st.header("🔍 Market Regime Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load and analyze data
            data = self._load_market_data()
            
            if data is not None and len(data) > 50:
                st.subheader("📊 Current Market Analysis")
                
                # Basic market metrics
                latest_price = data['close'].iloc[-1] if 'close' in data.columns else 0
                price_change = data['close'].pct_change().iloc[-1] * 100 if 'close' in data.columns else 0
                volatility = data['close'].pct_change().std() * 100 if 'close' in data.columns else 0
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Current Price", f"€{latest_price:.4f}")
                
                with metric_col2:
                    st.metric("Price Change %", f"{price_change:.2f}%")
                
                with metric_col3:
                    st.metric("Volatility %", f"{volatility:.2f}%")
                
                # Regime detection
                if st.button("🎯 Analyze Current Regime", use_container_width=True):
                    with st.spinner("Analyzing market regime..."):
                        try:
                            # Train detector if needed
                            self.regime_detector.fit(data, 'close')
                            
                            # Detect current regime
                            regime_result = self.regime_detector.detect_regime(data, 'close')
                            
                            # Display results
                            st.success(f"Regime Detection Complete!")
                            
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.subheader("🎯 Detected Regime")
                                
                                # Regime display with confidence
                                confidence_color = "green" if regime_result.confidence > 0.7 else "orange" if regime_result.confidence > 0.4 else "red"
                                
                                st.markdown(f"""
                                **Regime:** {regime_result.regime.value.replace('_', ' ').title()}  
                                **Confidence:** <span style='color:{confidence_color}'>{regime_result.confidence:.1%}</span>  
                                **Method:** {regime_result.method.value.replace('_', ' ').title()}  
                                **Timestamp:** {regime_result.detection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                                """, unsafe_allow_html=True)
                                
                                # Supporting evidence
                                if regime_result.supporting_evidence:
                                    st.subheader("📋 Supporting Evidence")
                                    
                                    for key, value in regime_result.supporting_evidence.items():
                                        if isinstance(value, (int, float)):
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
                                        elif isinstance(value, list) and len(value) <= 5:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            
                            with result_col2:
                                # Regime probability distribution
                                if regime_result.regime_probability:
                                    st.subheader("📊 Regime Probabilities")
                                    
                                    regime_names = [r.value.replace('_', ' ').title() for r in regime_result.regime_probability.keys()]
                                    regime_probs = list(regime_result.regime_probability.values())
                                    
                                    fig = px.bar(
                                        x=regime_probs,
                                        y=regime_names,
                                        orientation='h',
                                        title="Regime Detection Confidence"
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                else:
                                    # Single regime confidence
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = regime_result.confidence * 100,
                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                        title = {'text': "Detection Confidence (%)"},
                                        gauge = {
                                            'axis': {'range': [None, 100]},
                                            'bar': {'color': confidence_color},
                                            'steps': [
                                                {'range': [0, 40], 'color': "lightgray"},
                                                {'range': [40, 70], 'color': "gray"},
                                                {'range': [70, 100], 'color': "lightgreen"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 90
                                            }
                                        }
                                    ))
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Time series visualization
                            self._plot_regime_analysis(data, regime_result)
                            
                        except Exception as e:
                            st.error(f"Regime detection failed: {e}")
                            self.logger.error(f"Regime detection error: {e}")
            
            else:
                st.warning("Insufficient data for regime analysis. Need at least 50 data points.")
        
        with col2:
            # Regime detection status
            st.subheader("📈 Detection Status")
            
            summary = self.regime_detector.get_regime_summary()
            
            # Current regime info
            current_regime = summary.get('current_regime', 'unknown')
            regime_confidence = summary.get('regime_confidence', 0)
            
            st.metric("Current Regime", current_regime.replace('_', ' ').title())
            st.metric("Confidence", f"{regime_confidence:.1%}")
            
            # Model status
            st.subheader("🔧 Model Status")
            model_status = summary.get('model_status', {})
            
            status_items = [
                ("Autoencoder", model_status.get('autoencoder_available', False)),
                ("Clustering", model_status.get('clustering_available', False)),
                ("PCA", model_status.get('pca_available', False)),
                ("Scaler", model_status.get('scaler_fitted', False))
            ]
            
            for name, available in status_items:
                status_icon = "✅" if available else "❌"
                st.write(f"{status_icon} {name}")
            
            # Recent transitions
            recent_transitions = summary.get('recent_transitions', [])
            
            if recent_transitions:
                st.subheader("🔄 Recent Transitions")
                
                for transition in recent_transitions[-3:]:  # Show last 3
                    from_regime = transition['from_regime'].replace('_', ' ').title()
                    to_regime = transition['to_regime'].replace('_', ' ').title()
                    confidence = transition['confidence']
                    
                    st.write(f"**{from_regime}** → **{to_regime}** ({confidence:.1%})")
            
            # Regime distribution
            regime_dist = summary.get('regime_distribution', {})
            
            if regime_dist:
                st.subheader("📊 Regime Distribution")
                
                dist_data = pd.DataFrame([
                    {'Regime': k.replace('_', ' ').title(), 'Frequency': v}
                    for k, v in regime_dist.items()
                ])
                
                fig = px.pie(
                    dist_data,
                    values='Frequency',
                    names='Regime',
                    title="Recent Regime Distribution"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_adaptive_switching_tab(self):
        """Render adaptive model switching tab"""
        st.header("🔄 Adaptive Model Switching")
        
        # Current adaptation status
        adaptation_summary = self.adaptive_switcher.get_adaptation_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_regime = adaptation_summary.get('current_regime', 'unknown')
            st.metric("Active Regime", current_regime.replace('_', ' ').title())
        
        with col2:
            current_model = adaptation_summary.get('current_model_type', 'unknown')
            st.metric("Current Model", current_model.replace('_', ' ').title())
        
        with col3:
            feature_count = adaptation_summary.get('current_features_count', 0)
            st.metric("Active Features", feature_count)
        
        with col4:
            daily_switches = adaptation_summary.get('daily_switch_count', 0)
            st.metric("Switches Today", daily_switches)
        
        # Adaptation controls
        st.subheader("🎮 Adaptation Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Load data for adaptation
            data = self._load_market_data()
            
            if st.button("⚡ Run Full Adaptation", use_container_width=True):
                if data is not None and len(data) > 50:
                    with st.spinner("Running adaptive model switching..."):
                        try:
                            # Add basic features to data
                            enhanced_data = self._enhance_data_for_adaptation(data)
                            
                            # Run adaptation
                            result = self.adaptive_switcher.adapt_to_regime(enhanced_data, 'close')
                            
                            if result.get('success'):
                                st.success("✅ Model adaptation successful!")
                                
                                adaptation_col1, adaptation_col2 = st.columns(2)
                                
                                with adaptation_col1:
                                    st.write(f"**Regime:** {result['regime'].replace('_', ' ').title()}")
                                    st.write(f"**Model Type:** {result['model_type'].replace('_', ' ').title()}")
                                    st.write(f"**Features:** {len(result['features'])}")
                                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                                
                                with adaptation_col2:
                                    st.write("**Trading Thresholds:**")
                                    thresholds = result['thresholds']
                                    for name, value in thresholds.items():
                                        st.write(f"- {name.title()}: {value:.1%}")
                                
                                # Show features
                                if result['features']:
                                    st.subheader("📋 Selected Features")
                                    features_df = pd.DataFrame({
                                        'Feature': result['features'][:10],  # Show top 10
                                        'Index': range(1, min(11, len(result['features']) + 1))
                                    })
                                    st.dataframe(features_df, use_container_width=True)
                                
                                st.rerun()
                            
                            else:
                                st.error(f"❌ Adaptation failed: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Adaptation error: {e}")
                
                else:
                    st.warning("Insufficient data for adaptation")
        
        with col2:
            # Signal generation
            if st.button("📊 Generate Trading Signals", use_container_width=True):
                if data is not None and len(data) > 20:
                    with st.spinner("Generating trading signals..."):
                        try:
                            enhanced_data = self._enhance_data_for_adaptation(data)
                            signals = self.adaptive_switcher.get_current_signals(enhanced_data, 'close')
                            
                            signal = signals.get('signal', 'HOLD')
                            confidence = signals.get('confidence', 0)
                            
                            # Signal display
                            signal_color = {
                                'BUY': 'green',
                                'SELL': 'red',
                                'HOLD': 'orange'
                            }.get(signal, 'gray')
                            
                            st.markdown(f"""
                            **Signal:** <span style='color:{signal_color}; font-size:24px; font-weight:bold'>{signal}</span>  
                            **Confidence:** {confidence:.1%}
                            """, unsafe_allow_html=True)
                            
                            # Additional signal info
                            if 'predicted_return' in signals:
                                st.write(f"**Predicted Return:** {signals['predicted_return']:.2%}")
                            
                            if 'current_price' in signals and 'predicted_price' in signals:
                                current = signals['current_price']
                                predicted = signals['predicted_price']
                                st.write(f"**Current Price:** €{current:.4f}")
                                st.write(f"**Predicted Price:** €{predicted:.4f}")
                        
                        except Exception as e:
                            st.error(f"Signal generation error: {e}")
                
                else:
                    st.warning("Insufficient data for signal generation")
        
        # Regime-specific model performance
        st.subheader("📊 Model Performance by Regime")
        
        regime_mappings = adaptation_summary.get('regime_mappings', {})
        
        if regime_mappings:
            performance_data = []
            
            for regime, mapping in regime_mappings.items():
                performance_data.append({
                    'Regime': regime.replace('_', ' ').title(),
                    'Model Type': mapping['model_type'].replace('_', ' ').title(),
                    'Features': mapping['feature_count'],
                    'Switches': mapping['switch_count'],
                    'Avg Performance': mapping['avg_performance']
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Performance visualization
            if len(performance_df) > 0:
                fig = px.bar(
                    performance_df,
                    x='Regime',
                    y='Avg Performance',
                    color='Model Type',
                    title="Average Performance by Regime and Model Type"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_analysis_tab(self):
        """Render performance analysis tab"""
        st.header("📊 Performance Analysis")
        
        # Load data for analysis
        data = self._load_market_data()
        
        if data is not None and len(data) > 100:
            # Performance evaluation
            if st.button("📈 Evaluate Current Performance", use_container_width=True):
                with st.spinner("Evaluating model performance..."):
                    try:
                        enhanced_data = self._enhance_data_for_adaptation(data)
                        metrics = self.adaptive_switcher.evaluate_current_performance(enhanced_data, 'close')
                        
                        if metrics:
                            st.success("✅ Performance evaluation complete!")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                            
                            with metric_col2:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                            
                            with metric_col3:
                                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            
                            with metric_col4:
                                st.metric("Directional Accuracy", f"{metrics.get('directional_accuracy', 0):.1%}")
                            
                            # Performance visualization
                            metric_names = list(metrics.keys())
                            metric_values = list(metrics.values())
                            
                            if len(metric_names) > 0:
                                fig = px.bar(
                                    x=metric_names,
                                    y=metric_values,
                                    title="Current Model Performance Metrics"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.warning("No performance metrics available. Model may not be trained.")
                    
                    except Exception as e:
                        st.error(f"Performance evaluation error: {e}")
        
        else:
            st.warning("Insufficient data for performance analysis. Need at least 100 data points.")
        
        # Historical performance trends
        st.subheader("📈 Performance Trends")
        
        # Generate sample performance data for demonstration
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1h')
        sample_performance = pd.DataFrame({
            'timestamp': dates,
            'r2_score': 0.5 + 0.3 * np.random.random(len(dates)),
            'directional_accuracy': 0.4 + 0.4 * np.random.random(len(dates)),
            'regime_changes': np.random.poisson(0.1, len(dates))
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Model Performance Over Time', 'Regime Stability'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_performance['timestamp'],
                y=sample_performance['r2_score'],
                name='R² Score',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_performance['timestamp'],
                y=sample_performance['directional_accuracy'],
                name='Directional Accuracy',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_performance['timestamp'],
                y=sample_performance['regime_changes'],
                name='Regime Changes',
                line=dict(color='red'),
                mode='markers'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Performance Analysis Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_configuration_tab(self):
        """Render model configuration tab"""
        st.header("⚙️ Model Configuration")
        
        # Configuration overview
        regime_mappings = self.adaptive_switcher.regime_models
        
        st.subheader("🔧 Regime-Model Mappings")
        
        # Display configuration for each regime
        for regime, mapping in regime_mappings.items():
            with st.expander(f"🌍 {regime.value.replace('_', ' ').title()} Configuration"):
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.write("**Primary Model:**")
                    st.write(f"- Type: {mapping.primary_model.model_type.value.replace('_', ' ').title()}")
                    st.write(f"- Features: {len(mapping.feature_set)}")
                    st.write(f"- Switch Count: {mapping.switch_count}")
                    
                    st.write("**Trading Thresholds:**")
                    for name, value in mapping.trading_thresholds.items():
                        st.write(f"- {name.title()}: {value:.1%}")
                
                with config_col2:
                    st.write("**Backup Models:**")
                    for backup in mapping.backup_models:
                        st.write(f"- {backup.model_type.value.replace('_', ' ').title()}")
                    
                    st.write("**Key Features:**")
                    for feature in mapping.feature_set[:5]:  # Show top 5
                        st.write(f"- {feature.replace('_', ' ').title()}")
                    
                    if len(mapping.feature_set) > 5:
                        st.write(f"... and {len(mapping.feature_set) - 5} more")
        
        # Global configuration
        st.subheader("🌐 Global Settings")
        
        config = self.adaptive_switcher.config
        
        setting_col1, setting_col2 = st.columns(2)
        
        with setting_col1:
            st.write("**Adaptation Settings:**")
            st.write(f"- Strategy: {config.adaptation_strategy.value.title()}")
            st.write(f"- Max Switches/Day: {config.max_switches_per_day}")
            st.write(f"- Min Performance Improvement: {config.min_performance_improvement:.1%}")
            st.write(f"- Model Switch Threshold: {config.model_switch_threshold:.1%}")
        
        with setting_col2:
            st.write("**Training Settings:**")
            st.write(f"- Retrain Frequency: {config.retrain_frequency_hours}h")
            st.write(f"- Min Training Samples: {config.min_training_samples}")
            st.write(f"- Validation Split: {config.validation_split:.1%}")
            st.write(f"- Evaluation Window: {config.performance_evaluation_window}")
    
    def _render_realtime_monitoring_tab(self):
        """Render real-time monitoring tab"""
        st.header("📈 Real-time Monitoring")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh (30 seconds)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Current status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Current Detection")
            
            data = self._load_market_data()
            if data is not None and len(data) > 50:
                try:
                    regime_result = detect_current_regime(data, 'close')
                    
                    st.metric(
                        "Detected Regime",
                        regime_result.regime.value.replace('_', ' ').title(),
                        f"{regime_result.confidence:.1%} confidence"
                    )
                    
                    st.write(f"**Method:** {regime_result.method.value.replace('_', ' ').title()}")
                    st.write(f"**Last Updated:** {regime_result.detection_timestamp.strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"Detection error: {e}")
            else:
                st.warning("Insufficient data")
        
        with col2:
            st.subheader("🔄 Current Adaptation")
            
            adaptation_summary = self.adaptive_switcher.get_adaptation_summary()
            
            st.metric(
                "Active Model",
                adaptation_summary.get('current_model_type', 'unknown').replace('_', ' ').title(),
                f"{adaptation_summary.get('current_features_count', 0)} features"
            )
            
            last_adaptation = adaptation_summary.get('last_adaptation_time')
            if last_adaptation:
                st.write(f"**Last Adaptation:** {last_adaptation}")
            
            st.write(f"**Switches Today:** {adaptation_summary.get('daily_switch_count', 0)}")
        
        with col3:
            st.subheader("📊 System Health")
            
            # Simple health indicators
            regime_summary = self.regime_detector.get_regime_summary()
            model_status = regime_summary.get('model_status', {})
            
            health_score = sum([
                model_status.get('scaler_fitted', False),
                adaptation_summary.get('current_features_count', 0) > 0,
                regime_summary.get('regime_confidence', 0) > 0.5
            ]) / 3
            
            health_color = "green" if health_score > 0.7 else "orange" if health_score > 0.3 else "red"
            
            st.metric(
                "System Health",
                f"{health_score:.1%}",
                "Operational" if health_score > 0.5 else "Issues Detected"
            )
            
            st.markdown(f"**Status:** <span style='color:{health_color}'>{'🟢 Healthy' if health_score > 0.7 else '🟡 Partial' if health_score > 0.3 else '🔴 Issues'}</span>", 
                       unsafe_allow_html=True)
        
        # Live monitoring chart
        st.subheader("📈 Live Market Monitoring")
        
        if data is not None and len(data) > 100:
            # Recent price and regime data
            recent_data = data.tail(100)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Price Movement', 'Volatility'],
                vertical_spacing=0.1
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['close'] if 'close' in recent_data.columns else recent_data.iloc[:, 0],
                    name='Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Volatility chart
            if 'close' in recent_data.columns:
                volatility = recent_data['close'].pct_change().rolling(10).std()
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=volatility,
                        name='Volatility',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=500, title_text="Real-time Market Data")
            st.plotly_chart(fig, use_container_width=True)
    
    def _load_market_data(self) -> pd.DataFrame:
        """Load market data for analysis"""
        try:
            coin = st.session_state.get('selected_coin', 'BTC/EUR')
            timeframe = st.session_state.get('selected_timeframe', '1h')
            period = st.session_state.get('data_period', timedelta(days=1))
            
            end_time = datetime.now()
            start_time = end_time - period
            
            data = self.data_manager.get_historical_data(coin, timeframe, start_time, end_time)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return None
    
    def _enhance_data_for_adaptation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with basic features for adaptation"""
        try:
            enhanced_data = data.copy()
            
            if 'close' in data.columns:
                # Add moving averages
                enhanced_data['sma_10'] = data['close'].rolling(10).mean()
                enhanced_data['sma_20'] = data['close'].rolling(20).mean()
                enhanced_data['ema_12'] = data['close'].ewm(span=12).mean()
                
                # Add returns and volatility
                enhanced_data['returns'] = data['close'].pct_change()
                enhanced_data['volatility'] = enhanced_data['returns'].rolling(10).std()
                
                # Add RSI-like indicator
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                enhanced_data['rsi'] = 100 - (100 / (1 + rs))
            
            if 'volume' in data.columns:
                enhanced_data['volume_sma'] = data['volume'].rolling(10).mean()
                enhanced_data['volume_ratio'] = data['volume'] / enhanced_data['volume_sma']
            
            # Fill NaN values
            enhanced_data = enhanced_data.fillna(0)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Data enhancement error: {e}")
            return data
    
    def _plot_regime_analysis(self, data: pd.DataFrame, regime_result):
        """Plot regime analysis visualization"""
        try:
            st.subheader("📊 Market Regime Analysis")
            
            if 'close' in data.columns and len(data) > 50:
                recent_data = data.tail(100)
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['Price Movement', 'Volatility', 'Volume'],
                    vertical_spacing=0.08
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data['close'],
                        name='Close Price',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Volatility
                volatility = recent_data['close'].pct_change().rolling(10).std()
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=volatility,
                        name='Volatility',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                # Volume
                if 'volume' in recent_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=recent_data.index,
                            y=recent_data['volume'],
                            name='Volume',
                            line=dict(color='green')
                        ),
                        row=3, col=1
                    )
                
                # Add regime detection annotation
                fig.add_annotation(
                    x=recent_data.index[-1],
                    y=recent_data['close'].iloc[-1],
                    text=f"Detected: {regime_result.regime.value.replace('_', ' ').title()}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )
                
                fig.update_layout(height=600, title_text="Market Analysis with Regime Detection")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Visualization error: {e}")
    
    def _detect_current_regime(self):
        """Detect current regime and update display"""
        try:
            data = self._load_market_data()
            
            if data is not None and len(data) > 50:
                with st.spinner("Detecting current market regime..."):
                    regime_result = detect_current_regime(data, 'close')
                    st.success(f"Current regime: {regime_result.regime.value.replace('_', ' ').title()}")
                    st.rerun()
            else:
                st.error("Insufficient data for regime detection")
        
        except Exception as e:
            st.error(f"Regime detection failed: {e}")
    
    def _force_model_adaptation(self):
        """Force model adaptation and update display"""
        try:
            data = self._load_market_data()
            
            if data is not None and len(data) > 50:
                with st.spinner("Forcing model adaptation..."):
                    enhanced_data = self._enhance_data_for_adaptation(data)
                    result = adapt_to_current_regime(enhanced_data, 'close')
                    
                    if result.get('success'):
                        st.success(f"Adaptation successful: {result['model_type']} model for {result['regime']} regime")
                    else:
                        st.error(f"Adaptation failed: {result.get('error', 'Unknown error')}")
                    
                    st.rerun()
            else:
                st.error("Insufficient data for adaptation")
        
        except Exception as e:
            st.error(f"Model adaptation failed: {e}")
    
    def _train_detection_models(self):
        """Train regime detection models"""
        try:
            data = self._load_market_data()
            
            if data is not None and len(data) > 100:
                with st.spinner("Training regime detection models..."):
                    self.regime_detector.fit(data, 'close')
                    st.success("✅ Detection models trained successfully!")
                    st.rerun()
            else:
                st.error("Insufficient data for model training. Need at least 100 data points.")
        
        except Exception as e:
            st.error(f"Model training failed: {e}")


def main():
    """Main dashboard entry point"""
    try:
        dashboard = MarketRegimeDashboard()
        dashboard.render()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()