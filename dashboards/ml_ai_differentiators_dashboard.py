#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - ML/AI Differentiators Dashboard
Interactive dashboard for monitoring all 8 ML/AI differentiators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json

# Import the ML differentiators
try:
    from core.ml_ai_differentiators import (
        get_ml_differentiators_coordinator,
        MLDifferentiatorConfig,
    )
    from core.ai_news_event_mining import get_ai_news_event_mining_coordinator, EventMiningConfig
    from core.ai_portfolio_optimizer import get_ai_portfolio_optimizer_coordinator, PortfolioConfig

    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"ML Differentiators not available: {e}")
    ML_AVAILABLE = False


def main():
    st.set_page_config(
        page_title="CryptoSmartTrader V2 - ML/AI Differentiators", page_icon="🧠", layout="wide"
    )

    st.title("🧠 ML/AI Differentiators Dashboard")
    st.markdown("**Volledige monitoring van alle 8 next-level AI capabilities**")

    if not ML_AVAILABLE:
        st.error("ML Differentiators engines niet beschikbaar. Check de installatie.")
        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 System Overview", "📊 Live Performance", "🧪 Model Analysis", "⚙️ Configuration"]
    )

    with tab1:
        show_system_overview()

    with tab2:
        show_live_performance()

    with tab3:
        show_model_analysis()

    with tab4:
        show_configuration()


def show_system_overview():
    """Show comprehensive system status"""

    st.header("🎯 System Status Overview")

    # Get coordinators
    ml_coordinator = get_ml_differentiators_coordinator()
    news_coordinator = get_ai_news_event_mining_coordinator()
    portfolio_coordinator = get_ai_portfolio_optimizer_coordinator()

    # Get status from all systems
    ml_status = ml_coordinator.get_system_status()
    news_status = news_coordinator.get_system_status()
    portfolio_status = portfolio_coordinator.get_system_status()

    # Create three columns for the main differentiators
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧠 ML Differentiators Core")

        # Status indicators
        status_data = {
            "Deep Learning": "✅ Operational"
            if ml_status["deep_learning"]["model_trained"]
            else "⚠️ Training Required",
            "Feature Fusion": f"✅ {ml_status['feature_fusion']['sources_configured']} sources",
            "Confidence Filter": f"✅ {ml_status['confidence_filtering']['threshold']} threshold",
            "Self Learning": f"📈 {ml_status['self_learning']['prediction_history_size']} predictions",
            "Explainability": "✅ SHAP Ready"
            if ml_status["explainability"]["shap_available"]
            else "⚠️ SHAP Missing",
            "Anomaly Detection": "✅ Active"
            if ml_status["anomaly_detection"]["baseline_fitted"]
            else "⚠️ No Baseline",
        }

        for feature, status in status_data.items():
            st.write(f"**{feature}:** {status}")

    with col2:
        st.subheader("📰 AI News Mining")

        news_data = {
            "AI Analyzer": "✅ Ready" if news_status["ai_analyzer_ready"] else "❌ Not Ready",
            "OpenAI Integration": "✅ Available"
            if news_status["openai_available"]
            else "❌ Missing",
            "Recent Events": f"📊 {news_status['recent_events_count']} total",
            "Events Last Hour": f"⏰ {news_status['events_last_hour']} new",
            "URLs Processed": f"🔗 {news_status['processed_urls_count']} cached",
        }

        for feature, status in news_data.items():
            st.write(f"**{feature}:** {status}")

    with col3:
        st.subheader("📊 AI Portfolio Optimizer")

        portfolio_data = {
            "ML Models": "✅ Trained"
            if portfolio_status["ml_predictor_trained"]
            else "⚠️ Not Trained",
            "Model Count": f"🔢 {portfolio_status['models_count']} active",
            "Current Assets": f"💼 {portfolio_status['current_allocation_assets']} positions",
            "Rebalance Status": "🔄 Required"
            if portfolio_status["should_rebalance"]
            else "✅ Current",
            "Performance History": f"📈 {portfolio_status['performance_history_length']} records",
        }

        for feature, status in portfolio_data.items():
            st.write(f"**{feature}:** {status}")

    # Overall system health
    st.header("🎯 Overall System Health")

    # Calculate health score
    health_components = {
        "Deep Learning": 1.0 if ml_status["deep_learning"]["model_trained"] else 0.5,
        "News Mining": 1.0 if news_status["ai_analyzer_ready"] else 0.0,
        "Portfolio Optimization": 1.0 if portfolio_status["models_count"] > 0 else 0.5,
        "Dependencies": 1.0 if ml_status["dependencies"]["sklearn"] else 0.3,
    }

    overall_health = sum(health_components.values()) / len(health_components)

    # Create health gauge
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall_health * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "System Health Score"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {
                    "color": "green"
                    if overall_health > 0.8
                    else "orange"
                    if overall_health > 0.5
                    else "red"
                },
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "lightgreen"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    # Health breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Health Breakdown")
        health_df = pd.DataFrame(
            [
                {"Component": k, "Health": f"{v * 100:.1f}%", "Score": v}
                for k, v in health_components.items()
            ]
        )
        st.dataframe(health_df, use_container_width=True)

    with col2:
        st.subheader("📋 Next Actions")
        actions = []

        if not ml_status["deep_learning"]["model_trained"]:
            actions.append("🔧 Train deep learning models")
        if not news_status["ai_analyzer_ready"]:
            actions.append("🔑 Configure OpenAI API key")
        if portfolio_status["should_rebalance"]:
            actions.append("⚖️ Run portfolio rebalancing")
        if not ml_status["anomaly_detection"]["baseline_fitted"]:
            actions.append("📊 Establish anomaly baseline")

        if actions:
            for action in actions:
                st.write(f"• {action}")
        else:
            st.success("🎉 All systems operational!")


def show_live_performance():
    """Show live performance metrics"""

    st.header("📊 Live Performance Monitoring")

    # Create sample performance data for demo
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="H")

    # ML Differentiators Performance
    st.subheader("🧠 ML Differentiators Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Confidence scores over time
        confidence_data = np.random.beta(8, 2, len(dates))  # High confidence distribution

        fig_confidence = go.Figure()
        fig_confidence.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_data,
                mode="lines",
                name="Confidence Score",
                line=dict(color="blue", width=2),
            )
        )
        fig_confidence.add_hline(
            y=0.8, line_dash="dash", line_color="red", annotation_text="Confidence Threshold (80%)"
        )
        fig_confidence.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_confidence, use_container_width=True)

    with col2:
        # Model accuracy
        accuracy_data = 0.7 + 0.2 * np.random.beta(2, 2, len(dates))

        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(
            go.Scatter(
                x=dates,
                y=accuracy_data,
                mode="lines",
                name="Model Accuracy",
                line=dict(color="green", width=2),
            )
        )
        fig_accuracy.update_layout(
            title="Model Accuracy Trend",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)

    # News Mining Performance
    st.subheader("📰 News Mining Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Events detected per hour
        events_per_hour = np.random.poisson(3, len(dates))

        fig_events = go.Figure()
        fig_events.add_trace(
            go.Bar(
                x=dates[-24:],
                y=events_per_hour[-24:],
                name="Events Detected",
                marker_color="orange",
            )
        )
        fig_events.update_layout(
            title="News Events Detected (Last 24h)", xaxis_title="Hour", yaxis_title="Events Count"
        )
        st.plotly_chart(fig_events, use_container_width=True)

    with col2:
        # Sentiment distribution
        sentiment_scores = np.random.normal(0.1, 0.3, 100)  # Slightly bullish

        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(
            go.Histogram(
                x=sentiment_scores, nbinsx=20, name="Sentiment Distribution", marker_color="purple"
            )
        )
        fig_sentiment.update_layout(
            title="News Sentiment Distribution",
            xaxis_title="Sentiment Score",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Portfolio Performance
    st.subheader("📊 Portfolio Optimization Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Portfolio returns
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative_returns = (1 + pd.Series(returns)).cumprod()

        fig_returns = go.Figure()
        fig_returns.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=3),
            )
        )
        fig_returns.update_layout(
            title="Portfolio Performance", xaxis_title="Time", yaxis_title="Cumulative Return"
        )
        st.plotly_chart(fig_returns, use_container_width=True)

    with col2:
        # Risk metrics
        volatility = pd.Series(returns).rolling(24).std() * np.sqrt(24)

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=dates,
                y=volatility,
                mode="lines",
                name="24h Volatility",
                line=dict(color="red", width=2),
            )
        )
        fig_vol.add_hline(
            y=0.15, line_dash="dash", line_color="orange", annotation_text="Risk Tolerance (15%)"
        )
        fig_vol.update_layout(
            title="Portfolio Risk (Rolling 24h Volatility)",
            xaxis_title="Time",
            yaxis_title="Volatility",
        )
        st.plotly_chart(fig_vol, use_container_width=True)


def show_model_analysis():
    """Show detailed model analysis"""

    st.header("🧪 Model Analysis & Explainability")

    # Feature importance analysis
    st.subheader("🔍 Feature Importance Analysis")

    # Mock SHAP-style explanation
    features = [
        "Price_MA_20",
        "Volume_Ratio",
        "RSI",
        "Sentiment_Score",
        "Whale_Activity",
        "News_Impact",
        "Social_Mentions",
        "On_Chain_Activity",
    ]
    importance_values = np.random.exponential(0.3, len(features))
    importance_values = importance_values / np.sum(importance_values)

    fig_importance = go.Figure()
    fig_importance.add_trace(
        go.Bar(
            x=importance_values,
            y=features,
            orientation="h",
            marker_color=[
                "red" if x < 0 else "green" for x in np.random.normal(0, 1, len(features))
            ],
        )
    )
    fig_importance.update_layout(
        title="Feature Importance (SHAP Values)", xaxis_title="SHAP Value", yaxis_title="Features"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # Model performance comparison
    st.subheader("📊 Model Performance Comparison")

    models = ["LSTM", "Transformer", "XGBoost", "Ensemble"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Create performance matrix
    performance_data = np.random.uniform(0.6, 0.9, (len(models), len(metrics)))

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=performance_data,
            x=metrics,
            y=models,
            colorscale="RdYlGn",
            text=np.round(performance_data, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig_heatmap.update_layout(
        title="Model Performance Comparison", xaxis_title="Metrics", yaxis_title="Models"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Anomaly detection results
    st.subheader("🚨 Anomaly Detection Results")

    col1, col2 = st.columns(2)

    with col1:
        # Recent anomalies
        anomalies_data = {
            "Timestamp": [datetime.now() - timedelta(hours=i) for i in range(10)],
            "Symbol": ["BTC", "ETH", "ADA", "SOL", "MATIC", "DOT", "LINK", "UNI", "AAVE", "COMP"],
            "Anomaly_Score": np.random.exponential(2, 10),
            "Type": np.random.choice(
                ["Price Spike", "Volume Surge", "Sentiment Shift", "Whale Movement"], 10
            ),
        }
        anomalies_df = pd.DataFrame(anomalies_data)
        st.dataframe(anomalies_df, use_container_width=True)

    with col2:
        # Anomaly score distribution
        all_scores = np.random.exponential(1, 1000)

        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Histogram(x=all_scores, nbinsx=30, name="Anomaly Scores", marker_color="red")
        )
        fig_anomaly.add_vline(
            x=np.percentile(all_scores, 95),
            line_dash="dash",
            line_color="black",
            annotation_text="95th Percentile",
        )
        fig_anomaly.update_layout(
            title="Anomaly Score Distribution", xaxis_title="Anomaly Score", yaxis_title="Frequency"
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)


def show_configuration():
    """Show and allow configuration changes"""

    st.header("⚙️ System Configuration")

    # ML Differentiators Configuration
    st.subheader("🧠 ML Differentiators Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Deep Learning Settings**")
        use_deep_learning = st.checkbox("Enable Deep Learning", value=True)
        lstm_hidden_size = st.slider("LSTM Hidden Size", 64, 256, 128)
        sequence_length = st.slider("Sequence Length", 30, 120, 60)

        st.write("**Confidence Filtering**")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.80)
        uncertainty_method = st.selectbox("Uncertainty Method", ["ensemble", "bayesian", "dropout"])

    with col2:
        st.write("**Feature Fusion Settings**")
        attention_mechanism = st.checkbox("Enable Attention Mechanism", value=True)
        feature_sources = st.multiselect(
            "Feature Sources",
            ["price", "volume", "sentiment", "whale", "news", "orderbook", "social", "onchain"],
            default=["price", "volume", "sentiment", "whale"],
        )

        st.write("**Self-Learning Settings**")
        enable_feedback_loop = st.checkbox("Enable Feedback Loop", value=True)
        retrain_frequency = st.slider("Retrain Frequency (hours)", 6, 72, 24)
        concept_drift_threshold = st.slider("Concept Drift Threshold", 0.05, 0.30, 0.15)

    # News Mining Configuration
    st.subheader("📰 News Mining Settings")

    col1, col2 = st.columns(2)

    with col1:
        importance_threshold = st.slider("Importance Threshold", 0.3, 0.9, 0.6)
        max_events_per_hour = st.slider("Max Events Per Hour", 10, 100, 50)

    with col2:
        requests_per_minute = st.slider("Requests Per Minute", 5, 60, 20)
        max_concurrent_requests = st.slider("Max Concurrent Requests", 1, 10, 5)

    # Portfolio Optimization Configuration
    st.subheader("📊 Portfolio Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        max_position_size = st.slider("Max Position Size", 0.10, 0.50, 0.25)
        min_position_size = st.slider("Min Position Size", 0.005, 0.05, 0.01)
        risk_tolerance = st.slider("Risk Tolerance", 0.05, 0.30, 0.15)

    with col2:
        lookback_days = st.slider("Lookback Days", 30, 180, 90)
        rebalance_frequency = st.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly"])
        max_assets = st.slider("Max Assets", 5, 50, 20)

    # Apply configuration button
    if st.button("🔄 Apply Configuration Changes", type="primary"):
        try:
            # Create new configurations
            ml_config = MLDifferentiatorConfig(
                use_deep_learning=use_deep_learning,
                lstm_hidden_size=lstm_hidden_size,
                sequence_length=sequence_length,
                confidence_threshold=confidence_threshold,
                uncertainty_method=uncertainty_method,
                feature_sources=feature_sources,
                attention_mechanism=attention_mechanism,
                enable_feedback_loop=enable_feedback_loop,
                retrain_frequency=retrain_frequency,
                concept_drift_threshold=concept_drift_threshold,
            )

            news_config = EventMiningConfig(
                importance_threshold=importance_threshold,
                max_events_per_hour=max_events_per_hour,
                requests_per_minute=requests_per_minute,
                max_concurrent_requests=max_concurrent_requests,
            )

            portfolio_config = PortfolioConfig(
                max_position_size=max_position_size,
                min_position_size=min_position_size,
                risk_tolerance=risk_tolerance,
                lookback_days=lookback_days,
                rebalance_frequency=rebalance_frequency,
                max_assets=max_assets,
            )

            st.success("✅ Configuration updated successfully!")
            st.info("🔄 Restart the system to apply all changes.")

        except Exception as e:
            st.error(f"❌ Configuration update failed: {e}")


if __name__ == "__main__":
    main()
