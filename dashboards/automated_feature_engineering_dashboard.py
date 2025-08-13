#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Automated Feature Engineering Dashboard
Interactive dashboard for monitoring and controlling automated feature engineering
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

from core.automated_feature_engineering import (
    get_automated_feature_engineer,
    FeatureEngineeringConfig,
    engineer_features_for_coin,
)
from core.feature_discovery_engine import (
    get_feature_discovery_engine,
    DiscoveryConfig,
    AdaptationTrigger,
)
from core.shap_regime_analyzer import get_shap_regime_analyzer, MarketRegime
from core.live_feature_adaptation import get_live_adaptation_engine, AdaptationTrigger
from core.data_manager import get_data_manager


class AutomatedFeatureEngineeringDashboard:
    """Interactive dashboard for automated feature engineering system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.feature_engineer = get_automated_feature_engineer()
        self.discovery_engine = get_feature_discovery_engine()
        self.shap_analyzer = get_shap_regime_analyzer()
        self.adaptation_engine = get_live_adaptation_engine()
        self.data_manager = get_data_manager()

        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - Automated Feature Engineering",
            page_icon="🔧",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render(self):
        """Render the main dashboard"""
        st.title("🔧 Automated Feature Engineering & Discovery")
        st.markdown("Advanced feature engineering with SHAP analysis and live adaptation")

        # Sidebar controls
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🎯 Feature Engineering",
                "🔍 Feature Discovery",
                "📊 SHAP Analysis",
                "⚡ Live Adaptation",
                "📈 Performance Monitor",
            ]
        )

        with tab1:
            self._render_feature_engineering_tab()

        with tab2:
            self._render_feature_discovery_tab()

        with tab3:
            self._render_shap_analysis_tab()

        with tab4:
            self._render_live_adaptation_tab()

        with tab5:
            self._render_performance_monitor_tab()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🔧 Feature Engineering Controls")

        # Coin selection
        try:
            available_coins = self.data_manager.get_supported_coins()
            if available_coins:
                selected_coin = st.sidebar.selectbox(
                    "Select Cryptocurrency", options=available_coins, index=0
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
            "Timeframe", options=timeframes, index=timeframes.index("1h")
        st.session_state.selected_timeframe = selected_timeframe

        # Data period
        data_periods = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last Day": timedelta(days=1),
            "Last Week": timedelta(weeks=1),
            "Last Month": timedelta(days=30),
        }

        selected_period = st.sidebar.selectbox(
            "Data Period", options=list(data_periods.keys()), index=2
        )
        st.session_state.data_period = data_periods[selected_period]

        # Feature engineering settings
        st.sidebar.subheader("⚙️ Engineering Settings")

        max_features = st.sidebar.slider(
            "Max Features per Iteration", min_value=10, max_value=200, value=50, step=10
        )

        correlation_threshold = st.sidebar.slider(
            "Correlation Threshold", min_value=0.5, max_value=0.99, value=0.95, step=0.01
        )

        # Update config
        st.session_state.fe_config = FeatureEngineeringConfig(
            max_features_per_iteration=max_features, correlation_threshold=correlation_threshold
        )

        # Action buttons
        st.sidebar.subheader("🚀 Actions")

        if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        if st.sidebar.button("🎯 Run Feature Engineering", use_container_width=True):
            self._run_feature_engineering()

        if st.sidebar.button("🔍 Discover New Features", use_container_width=True):
            self._run_feature_discovery()

        if st.sidebar.button("⚡ Force Adaptation", use_container_width=True):
            self._force_adaptation()

    def _render_feature_engineering_tab(self):
        """Render feature engineering tab"""
        st.header("🎯 Automated Feature Engineering")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Load and display data
            data = self._load_market_data()

            if data is not None and len(data) > 0:
                st.subheader("📊 Market Data Overview")

                # Basic stats
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    st.metric("Data Points", len(data))

                with stats_col2:
                    if "close" in data.columns:
                        price_change = data["close"].pct_change().iloc[-1] * 100
                        st.metric("Price Change %", f"{price_change:.2f}%")

                with stats_col3:
                    if "volume" in data.columns:
                        avg_volume = data["volume"].mean()
                        st.metric("Avg Volume", f"{avg_volume:,.0f}")

                with stats_col4:
                    if "close" in data.columns:
                        volatility = data["close"].pct_change().std() * 100
                        st.metric("Volatility %", f"{volatility:.2f}%")

                # Feature engineering results
                if st.button("🚀 Generate Features", use_container_width=True):
                    with st.spinner("Generating features..."):
                        try:
                            engineered_data = self._engineer_features(data)

                            if engineered_data is not None:
                                st.success(f"✅ Generated {len(engineered_data.columns)} features")

                                # Display feature summary
                                st.subheader("📋 Generated Features")

                                # Feature types breakdown
                                feature_types = self._analyze_feature_types(engineered_data)

                                fig = px.pie(
                                    values=list(feature_types.values()),
                                    names=list(feature_types.keys()),
                                    title="Feature Types Distribution",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Top features by importance
                                importance_summary = (
                                    self.feature_engineer.get_feature_importance_summary()

                                if "global_importance_scores" in importance_summary:
                                    top_features = dict(
                                        list(
                                            importance_summary["global_importance_scores"].items()[:10]
                                    )

                                    if top_features:
                                        fig = px.bar(
                                            x=list(top_features.values()),
                                            y=list(top_features.keys()),
                                            orientation="h",
                                            title="Top 10 Features by Importance",
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)

                                # Feature correlation heatmap
                                if len(engineered_data.columns) <= 20:  # Limit for visualization
                                    corr_matrix = engineered_data.select_dtypes(
                                        include=[np.number]
                                    ).corr()

                                    fig = px.imshow(
                                        corr_matrix,
                                        title="Feature Correlation Matrix",
                                        color_continuous_scale="RdBu",
                                        aspect="auto",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                # Show sample data
                                st.subheader("📋 Sample Engineered Data")
                                st.dataframe(engineered_data.head(10), use_container_width=True)

                        except Exception as e:
                            st.error(f"Feature engineering failed: {e}")
                            self.logger.error(f"Feature engineering error: {e}")

            else:
                st.warning("No market data available. Please check your connection.")

        with col2:
            # Feature engineering status
            st.subheader("⚙️ Engineering Status")

            # Get current status
            importance_summary = self.feature_engineer.get_feature_importance_summary()

            st.metric(
                "Total Features Generated", importance_summary.get("total_features_generated", 0)

            if importance_summary.get("feature_types_distribution"):
                st.write("**Feature Types:**")
                for ftype, count in importance_summary["feature_types_distribution"].items():
                    st.write(f"- {ftype}: {count}")

            # Configuration display
            st.subheader("🔧 Current Configuration")
            config = st.session_state.get("fe_config", FeatureEngineeringConfig())

            st.write(f"**Max Features:** {config.max_features_per_iteration}")
            st.write(f"**Correlation Threshold:** {config.correlation_threshold}")
            st.write(f"**Cross Feature Depth:** {config.cross_feature_max_depth}")

            # Performance info
            if importance_summary.get("regime_specific_sets"):
                st.subheader("🌍 Regime-Specific Sets")
                for regime, count in importance_summary["regime_specific_sets"].items():
                    st.write(f"- {regime}: {count} features")

    def _render_feature_discovery_tab(self):
        """Render feature discovery tab"""
        st.header("🔍 Automated Feature Discovery")

        # Discovery status
        discovery_summary = self.discovery_engine.get_discovery_summary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Discovery Mode", discovery_summary.get("discovery_mode", "Unknown"))

        with col2:
            st.metric("Active Features", discovery_summary.get("active_features", 0))

        with col3:
            st.metric("Candidates", discovery_summary.get("total_candidates", 0))

        with col4:
            st.metric("Discovery Iteration", discovery_summary.get("discovery_iteration", 0))

        # Discovery controls
        st.subheader("🎮 Discovery Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🌟 Exploratory Discovery", use_container_width=True):
                self._run_discovery_mode("exploratory")

        with col2:
            if st.button("🎯 Exploitative Discovery", use_container_width=True):
                self._run_discovery_mode("exploitative")

        with col3:
            if st.button("📈 Performance-Driven", use_container_width=True):
                self._run_discovery_mode("performance")

        # Feature discovery results
        if discovery_summary.get("top_performing_features"):
            st.subheader("🏆 Top Performing Features")

            top_features = discovery_summary["top_performing_features"][:10]

            feature_performance = pd.DataFrame(
                {"Feature": top_features, "Rank": range(1, len(top_features) + 1)}
            )

            st.dataframe(feature_performance, use_container_width=True)

        # Regime feature mapping
        if discovery_summary.get("regime_feature_mapping"):
            st.subheader("🌍 Features by Market Regime")

            regime_data = []
            for regime, count in discovery_summary["regime_feature_mapping"].items():
                regime_data.append({"Regime": regime, "Feature Count": count})

            if regime_data:
                regime_df = pd.DataFrame(regime_data)

                fig = px.bar(
                    regime_df, x="Regime", y="Feature Count", title="Feature Count by Market Regime"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Declining features alert
        if discovery_summary.get("declining_features"):
            st.subheader("⚠️ Declining Features")
            declining = discovery_summary["declining_features"]

            if declining:
                st.warning(f"Found {len(declining)} features with declining performance")
                for feature in declining[:5]:  # Show top 5
                    st.write(f"- {feature}")
            else:
                st.success("No declining features detected")

    def _render_shap_analysis_tab(self):
        """Render SHAP analysis tab"""
        st.header("📊 SHAP Feature Analysis")

        # Load data for analysis
        data = self._load_market_data()

        if data is not None and len(data) > 10:
            # Add some basic features for SHAP analysis
            enhanced_data = data.copy()

            if "close" in data.columns:
                enhanced_data["price_ma_5"] = data["close"].rolling(5).mean()
                enhanced_data["price_ma_20"] = data["close"].rolling(20).mean()
                enhanced_data["price_change"] = data["close"].pct_change()
                enhanced_data["volatility"] = data["close"].rolling(10).std()

            if "volume" in data.columns:
                enhanced_data["volume_ma"] = data["volume"].rolling(10).mean()
                enhanced_data["volume_ratio"] = data["volume"] / enhanced_data["volume_ma"]

            # Remove NaN values
            enhanced_data = enhanced_data.fillna(0)

            if st.button("🧠 Run SHAP Analysis", use_container_width=True):
                with st.spinner("Running SHAP analysis..."):
                    try:
                        # Perform SHAP analysis
                        target_column = (
                            "close"
                            if "close" in enhanced_data.columns
                            else enhanced_data.columns[0]
                        )

                        regime_results = self.shap_analyzer.analyze_regime_specific_importance(
                            enhanced_data, target_column
                        )

                        if regime_results:
                            st.success(
                                f"✅ SHAP analysis completed for {len(regime_results)} regimes"
                            )

                            # Display results by regime
                            for regime, analysis in regime_results.items():
                                st.subheader(f"📊 {regime.value} Market Regime")

                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    # Feature importance chart
                                    importance_data = analysis.feature_importance

                                    if importance_data:
                                        top_features = dict(
                                            list(
                                                sorted(
                                                    importance_data.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True,
                                                )[:10]
                                        )

                                        fig = px.bar(
                                            x=list(top_features.values()),
                                            y=list(top_features.keys()),
                                            orientation="h",
                                            title=f"Feature Importance - {regime.value}",
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)

                                with col2:
                                    # Analysis metrics
                                    st.metric("Sample Count", analysis.sample_count)
                                    st.metric("Confidence", f"{analysis.confidence_score:.2f}")

                                    if analysis.model_performance:
                                        st.write("**Model Performance:**")
                                        for metric, value in analysis.model_performance.items():
                                            st.write(f"- {metric}: {value:.4f}")

                        else:
                            st.warning(
                                "No SHAP results available. This might be due to insufficient data or missing SHAP library."
                            )

                    except Exception as e:
                        st.error(f"SHAP analysis failed: {e}")
                        self.logger.error(f"SHAP analysis error: {e}")

            # Feature dominance across regimes
            st.subheader("🌍 Feature Dominance Across Regimes")

            try:
                dominance_map = self.shap_analyzer.get_regime_feature_dominance()

                if dominance_map:
                    # Create heatmap of feature dominance
                    regimes = list(dominance_map.keys())
                    all_features = set()

                    for regime_features in dominance_map.values():
                        all_features.update(regime_features.keys())

                    # Limit features for visualization
                    all_features = list(all_features)[:15]

                    dominance_matrix = []
                    for feature in all_features:
                        row = []
                        for regime in regimes:
                            weight = dominance_map.get(regime, {}).get(feature, 0)
                            row.append(weight)
                        dominance_matrix.append(row)

                    if dominance_matrix:
                        fig = px.imshow(
                            dominance_matrix,
                            x=regimes,
                            y=all_features,
                            title="Feature Dominance by Market Regime",
                            color_continuous_scale="Viridis",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not display feature dominance: {e}")

        else:
            st.warning("Insufficient data for SHAP analysis. Need at least 10 data points.")

    def _render_live_adaptation_tab(self):
        """Render live adaptation tab"""
        st.header("⚡ Live Feature Adaptation")

        # Adaptation status
        adaptation_summary = self.adaptation_engine.get_adaptation_summary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Features", adaptation_summary.get("current_features_count", 0))

        with col2:
            current_regime = adaptation_summary.get("current_regime", "Unknown")
            st.metric("Current Regime", current_regime)

        with col3:
            performance = adaptation_summary.get("current_performance", 0)
            st.metric("Performance Score", f"{performance:.3f}")

        with col4:
            adaptations_today = adaptation_summary.get("adaptations_today", 0)
            st.metric("Adaptations Today", adaptations_today)

        # Adaptation triggers
        st.subheader("🎮 Adaptation Triggers")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📈 Regime Change", use_container_width=True):
                self._trigger_adaptation("REGIME_CHANGE")

        with col2:
            if st.button("📉 Performance Decline", use_container_width=True):
                self._trigger_adaptation("PERFORMANCE_DECLINE")

        with col3:
            if st.button("🆕 New Discovery", use_container_width=True):
                self._trigger_adaptation("NEW_FEATURE_DISCOVERY")

        # Adaptation history
        if adaptation_summary.get("total_adaptations", 0) > 0:
            st.subheader("📜 Adaptation History")

            trigger_counts = adaptation_summary.get("adaptation_triggers", {})

            if trigger_counts:
                fig = px.pie(
                    values=list(trigger_counts.values()),
                    names=list(trigger_counts.keys()),
                    title="Adaptation Triggers Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Regime stability
        regime_stability = adaptation_summary.get("regime_stability", 0)
        st.subheader("🌍 Market Regime Stability")

        stability_color = (
            "green" if regime_stability > 5 else "orange" if regime_stability > 2 else "red"
        )
        st.markdown(
            f"**Stability Score:** <span style='color:{stability_color}'>{regime_stability}</span>",
            unsafe_allow_html=True,
        )

        # Feature performance cache info
        cache_size = adaptation_summary.get("feature_performance_cache_size", 0)
        st.metric("Cached Feature Performance", cache_size)

        # Rollback options
        if adaptation_summary.get("rollback_enabled", False):
            snapshots = adaptation_summary.get("snapshots_available", 0)

            if snapshots > 0:
                st.subheader("🔄 Rollback Options")
                st.info(f"{snapshots} snapshots available for rollback")

                if st.button("⏪ Rollback Last Adaptation", use_container_width=True):
                    self._perform_rollback()

    def _render_performance_monitor_tab(self):
        """Render performance monitoring tab"""
        st.header("📈 Performance Monitoring")

        # System performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Feature Engineering")
            fe_summary = self.feature_engineer.get_feature_importance_summary()

            st.metric("Total Features", fe_summary.get("total_features_generated", 0))
            st.metric("Feature Types", len(fe_summary.get("feature_types_distribution", {})))

            if fe_summary.get("attention_pruner_available"):
                st.success("✅ Attention Pruner Active")
            else:
                st.warning("⚠️ Attention Pruner Inactive")

        with col2:
            st.subheader("🔍 Feature Discovery")
            discovery_summary = self.discovery_engine.get_discovery_summary()

            st.metric("Active Features", discovery_summary.get("active_features", 0))
            st.metric("Discovery Iteration", discovery_summary.get("discovery_iteration", 0))
            st.metric(
                "Avg Performance", f"{discovery_summary.get('avg_feature_performance', 0):.3f}"
            )

        with col3:
            st.subheader("⚡ Live Adaptation")
            adaptation_summary = self.adaptation_engine.get_adaptation_summary()

            st.metric("Total Adaptations", adaptation_summary.get("total_adaptations", 0))
            st.metric("Adaptations Today", adaptation_summary.get("adaptations_today", 0))

            if adaptation_summary.get("last_adaptation"):
                last_adaptation = adaptation_summary["last_adaptation"]
                st.write(f"**Last:** {last_adaptation}")

        # Performance trends (placeholder for future implementation)
        st.subheader("📊 Performance Trends")

        # Generate sample performance data for demonstration
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="1h"
        )
        performance_data = pd.DataFrame(
            {
                "timestamp": dates,
                "feature_count": np.random.randint(20, 100, len(dates)),
                "adaptation_score": np.random.random(len(dates)),
                "discovery_rate": np.random.random(len(dates)),
            }
        )

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=["Feature Count", "Adaptation Score", "Discovery Rate"],
            vertical_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamp"],
                y=performance_data["feature_count"],
                name="Features",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamp"],
                y=performance_data["adaptation_score"],
                name="Adaptation",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamp"],
                y=performance_data["discovery_rate"],
                name="Discovery",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(height=600, title_text="System Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # System health indicators
        st.subheader("🔋 System Health")

        health_col1, health_col2, health_col3, health_col4 = st.columns(4)

        with health_col1:
            # Feature engineering health
            fe_health = (
                "🟢 Healthy" if fe_summary.get("total_features_generated", 0) > 0 else "🔴 Issues"
            )
            st.markdown(f"**Feature Engineering:** {fe_health}")

        with health_col2:
            # Discovery health
            discovery_health = (
                "🟢 Active" if discovery_summary.get("active_features", 0) > 0 else "🟡 Idle"
            )
            st.markdown(f"**Feature Discovery:** {discovery_health}")

        with health_col3:
            # Adaptation health
            adaptation_health = (
                "🟢 Ready" if adaptation_summary.get("rollback_enabled", False) else "🟡 Limited"
            )
            st.markdown(f"**Live Adaptation:** {adaptation_health}")

        with health_col4:
            # Overall health
            all_healthy = all(
                [
                    fe_summary.get("total_features_generated", 0) > 0,
                    discovery_summary.get("discovery_iteration", 0) > 0,
                    adaptation_summary.get("feature_performance_cache_size", 0) > 0,
                ]
            )
            overall_health = "🟢 Optimal" if all_healthy else "🟡 Partial"
            st.markdown(f"**Overall System:** {overall_health}")

    def _load_market_data(self) -> pd.DataFrame:
        """Load market data for selected coin and timeframe"""
        try:
            coin = st.session_state.get("selected_coin", "BTC/EUR")
            timeframe = st.session_state.get("selected_timeframe", "1h")
            period = st.session_state.get("data_period", timedelta(days=1))

            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - period

            # Get data from data manager
            data = self.data_manager.get_historical_data(coin, timeframe, start_time, end_time)

            return data

        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return None

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for the given data"""
        try:
            target_column = "close" if "close" in data.columns else data.columns[0]

            # Use the automated feature engineering
            engineered_data = engineer_features_for_coin(data, target_column)

            return engineered_data

        except Exception as e:
            self.logger.error(f"Feature engineering error: {e}")
            return None

    def _analyze_feature_types(self, data: pd.DataFrame) -> dict:
        """Analyze feature types in the data"""
        feature_types = {
            "Original": 0,
            "Rolling": 0,
            "EMA": 0,
            "Cross": 0,
            "Polynomial": 0,
            "Technical": 0,
            "Other": 0,
        }

        for col in data.columns:
            if any(x in col.lower() for x in ["rolling", "roll"]):
                feature_types["Rolling"] += 1
            elif "ema" in col.lower():
                feature_types["EMA"] += 1
            elif any(x in col for x in ["_x_", "_div_", "_diff_"]):
                feature_types["Cross"] += 1
            elif "poly" in col.lower():
                feature_types["Polynomial"] += 1
            elif any(x in col.lower() for x in ["rsi", "bollinger", "macd"]):
                feature_types["Technical"] += 1
            elif col in ["open", "high", "low", "close", "volume"]:
                feature_types["Original"] += 1
            else:
                feature_types["Other"] += 1

        return feature_types

    def _run_feature_engineering(self):
        """Run feature engineering process"""
        try:
            data = self._load_market_data()

            if data is not None:
                with st.spinner("Running feature engineering..."):
                    engineered_data = self._engineer_features(data)

                    if engineered_data is not None:
                        st.success(
                            f"✅ Feature engineering completed: {len(engineered_data.columns)} features"
                        )
                        st.rerun()
                    else:
                        st.error("Feature engineering failed")
            else:
                st.error("No data available for feature engineering")

        except Exception as e:
            st.error(f"Feature engineering error: {e}")

    def _run_feature_discovery(self):
        """Run feature discovery process"""
        try:
            data = self._load_market_data()

            if data is not None:
                with st.spinner("Running feature discovery..."):
                    target_column = "close" if "close" in data.columns else data.columns[0]

                    candidates = self.discovery_engine.discover_features(data, target_column)

                    st.success(f"✅ Discovery completed: {len(candidates)} candidates found")
                    st.rerun()
            else:
                st.error("No data available for feature discovery")

        except Exception as e:
            st.error(f"Feature discovery error: {e}")

    def _force_adaptation(self):
        """Force feature adaptation"""
        try:
            data = self._load_market_data()

            if data is not None:
                with st.spinner("Forcing feature adaptation..."):
                    target_column = "close" if "close" in data.columns else data.columns[0]

                    adapted_features = self.adaptation_engine.force_adaptation(
                        data, target_column, AdaptationTrigger.USER_REQUESTED
                    )

                    st.success(
                        f"✅ Adaptation completed: {len(adapted_features)} features selected"
                    )
                    st.rerun()
            else:
                st.error("No data available for adaptation")

        except Exception as e:
            st.error(f"Feature adaptation error: {e}")

    def _run_discovery_mode(self, mode: str):
        """Run discovery in specific mode"""
        try:
            # This would set the discovery mode and run discovery
            st.info(f"Setting discovery mode to: {mode}")
            self._run_feature_discovery()

        except Exception as e:
            st.error(f"Discovery mode error: {e}")

    def _trigger_adaptation(self, trigger_type: str):
        """Trigger feature adaptation with specific trigger"""
        try:
            data = self._load_market_data()

            if data is not None:
                trigger_map = {
                    "REGIME_CHANGE": AdaptationTrigger.REGIME_CHANGE,
                    "PERFORMANCE_DECLINE": AdaptationTrigger.PERFORMANCE_DECLINE,
                    "NEW_FEATURE_DISCOVERY": AdaptationTrigger.NEW_FEATURE_DISCOVERY,
                }

                trigger = trigger_map.get(trigger_type, AdaptationTrigger.USER_REQUESTED)

                with st.spinner(f"Triggering adaptation: {trigger_type}..."):
                    target_column = "close" if "close" in data.columns else data.columns[0]

                    adapted_features = self.adaptation_engine.adapt_features(
                        data, target_column, trigger
                    )

                    st.success(f"✅ Adaptation triggered: {len(adapted_features)} features")
                    st.rerun()
            else:
                st.error("No data available for adaptation")

        except Exception as e:
            st.error(f"Adaptation trigger error: {e}")

    def _perform_rollback(self):
        """Perform feature adaptation rollback"""
        try:
            with st.spinner("Performing rollback..."):
                success = self.adaptation_engine.rollback_adaptation()

                if success:
                    st.success("✅ Rollback completed successfully")
                    st.rerun()
                else:
                    st.error("❌ Rollback failed")

        except Exception as e:
            st.error(f"Rollback error: {e}")


def main():
    """Main dashboard entry point"""
    try:
        dashboard = AutomatedFeatureEngineeringDashboard()
        dashboard.render()

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
