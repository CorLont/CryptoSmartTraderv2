#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Causal Inference Dashboard
Interactive dashboard for causality discovery and counterfactual analysis
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
    analyze_causality,
    test_granger_causality,
    predict_counterfactuals,
    discover_market_causality,
)


class CausalInferenceDashboard:
    """Interactive dashboard for causal inference and analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize causal inference engine
        self.causal_engine = get_causal_inference_engine()

        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - Causal Inference",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render(self):
        """Render the main dashboard"""
        st.title("🧠 Causal Inference & Counterfactual Analysis")
        st.markdown("Discover WHY market movements happen using advanced causal inference")

        # Sidebar controls
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 Market Causality",
                "🔍 Granger Causality",
                "🔮 Counterfactual Analysis",
                "💡 Price Explanations",
                "📊 Analysis Results",
            ]
        )

        with tab1:
            self._render_market_causality_tab()

        with tab2:
            self._render_granger_causality_tab()

        with tab3:
            self._render_counterfactual_tab()

        with tab4:
            self._render_price_explanation_tab()

        with tab5:
            self._render_results_tab()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🧠 Causal Analysis Controls")

        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            [
                "Market Causality Discovery",
                "Custom Causal Analysis",
                "Granger Testing",
                "Counterfactual Scenarios",
            ],
        )

        st.sidebar.markdown("---")

        # Data generation for demo
        st.sidebar.subheader("📊 Demo Data")

        if st.sidebar.button("🎲 Generate Demo Market Data", use_container_width=True):
            demo_data = self._generate_demo_data()
            st.session_state["demo_market_data"] = demo_data
            st.sidebar.success("Demo data generated!")

        # Quick analysis
        st.sidebar.subheader("🚀 Quick Analysis")

        if st.sidebar.button("🔍 Discover Market Causality", use_container_width=True):
            if "demo_market_data" in st.session_state:
                with st.spinner("Analyzing market causality..."):
                    results = discover_market_causality(st.session_state["demo_market_data"])
                    st.session_state["causality_results"] = results
                st.sidebar.success("Analysis completed!")
            else:
                st.sidebar.warning("Generate demo data first!")

        if st.sidebar.button("📈 Test Granger Causality", use_container_width=True):
            if "demo_market_data" in st.session_state:
                with st.spinner("Testing Granger causality..."):
                    data = st.session_state["demo_market_data"]
                    variables = [
                        col
                        for col in data.columns
                        if "price" in col.lower() or "volume" in col.lower()
                    ][:4]
                    results = test_granger_causality(data, variables)
                    st.session_state["granger_results"] = results
                st.sidebar.success("Granger analysis completed!")
            else:
                st.sidebar.warning("Generate demo data first!")

        if st.sidebar.button("🔮 Generate Counterfactuals", use_container_width=True):
            if "demo_market_data" in st.session_state:
                with st.spinner("Generating counterfactual scenarios..."):
                    data = st.session_state["demo_market_data"]
                    price_cols = [col for col in data.columns if "price" in col.lower()]
                    if price_cols:
                        outcome = price_cols[0]
                        interventions = [col for col in data.columns if col != outcome][:3]
                        results = predict_counterfactuals(data, outcome, interventions)
                        st.session_state["counterfactual_results"] = results
                st.sidebar.success("Counterfactual analysis completed!")
            else:
                st.sidebar.warning("Generate demo data first!")

        # Settings
        st.sidebar.subheader("⚙️ Analysis Settings")

        significance_level = st.sidebar.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            format="%.2f",
        )

        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            format="%.2f",
        )

    def _render_market_causality_tab(self):
        """Render market causality discovery tab"""
        st.header("📈 Market Causality Discovery")
        st.markdown(
            "Discover causal relationships in cryptocurrency markets using Double Machine Learning"
        )

        # Check if analysis results are available
        if "causality_results" not in st.session_state:
            st.info("🔍 Run 'Discover Market Causality' from the sidebar to see results")

            # Show what the analysis would discover
            st.subheader("🎯 What We Can Discover")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **🔍 Causal Effects:**
                - Does volume cause price changes?
                - How does sentiment affect market movements?
                - What drives whale activity?
                - Cross-coin causal relationships
                """)

            with col2:
                st.markdown("""
                **📊 Methods Used:**
                - Double Machine Learning (DML)
                - Causal inference with confounders
                - Statistical significance testing
                - Effect size estimation
                """)

            return

        # Display results
        results = st.session_state["causality_results"]

        # Summary metrics
        summary = results.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Effects", summary.get("total_causal_effects", 0))

        with col2:
            st.metric("Significant Effects", summary.get("significant_effects", 0))

        with col3:
            st.metric("Granger Relations", summary.get("significant_granger", 0))

        with col4:
            st.metric("Counterfactuals", summary.get("high_confidence_scenarios", 0))

        # Causal effects visualization
        st.subheader("🎯 Discovered Causal Effects")

        causal_effects = results.get("causal_effects", [])

        if causal_effects:
            # Create causal effects dataframe
            effects_data = []
            for effect in causal_effects:
                effects_data.append(
                    {
                        "Treatment": effect["treatment"],
                        "Outcome": effect["outcome"],
                        "Effect Size": effect["effect_size"],
                        "P-Value": effect["p_value"],
                        "Significant": "✅" if effect["significance"] else "❌",
                        "Method": effect["method"],
                        "Sample Size": effect["sample_size"],
                    }
                )

            effects_df = pd.DataFrame(effects_data)
            st.dataframe(effects_df, use_container_width=True)

            # Visualization of effect sizes
            fig = go.Figure()

            significant_effects = [e for e in causal_effects if e["significance"]]

            if significant_effects:
                treatments = [e["treatment"] for e in significant_effects]
                outcomes = [e["outcome"] for e in significant_effects]
                effect_sizes = [e["effect_size"] for e in significant_effects]

                fig.add_trace(
                    go.Bar(
                        x=[f"{t} → {o}" for t, o in zip(treatments, outcomes)],
                        y=effect_sizes,
                        marker_color=["green" if e > 0 else "red" for e in effect_sizes],
                        text=[f"{e:.4f}" for e in effect_sizes],
                        textposition="auto",
                    )
                )

                fig.update_layout(
                    title="Significant Causal Effects",
                    xaxis_title="Treatment → Outcome",
                    yaxis_title="Effect Size",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant causal effects discovered in this analysis.")
        else:
            st.info("No causal effects analyzed yet.")

    def _render_granger_causality_tab(self):
        """Render Granger causality analysis tab"""
        st.header("🔍 Granger Causality Analysis")
        st.markdown("Test temporal causal relationships between time series")

        # Check if results are available
        if "granger_results" not in st.session_state:
            st.info("🔍 Run 'Test Granger Causality' from the sidebar to see results")

            # Explain Granger causality
            st.subheader("📚 What is Granger Causality?")

            st.markdown("""
            **Granger Causality** tests whether one time series can predict another:
            
            - **X Granger-causes Y** if past values of X improve prediction of Y
            - Based on temporal precedence and predictive power
            - Tests statistical causality, not true causation
            - Useful for understanding market dynamics and lead-lag relationships
            
            **Applications in Crypto Trading:**
            - Does Bitcoin price lead altcoin prices?
            - Can volume predict price movements?
            - Do sentiment indicators precede market moves?
            """)

            return

        # Display results
        results = st.session_state["granger_results"]

        if not results:
            st.warning("No Granger causality results available.")
            return

        # Summary statistics
        total_tests = len(results)
        significant_tests = len([r for r in results if r.is_causal])
        bidirectional = len([r for r in results if r.direction == "bidirectional"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Tests", total_tests)

        with col2:
            st.metric(
                "Significant",
                significant_tests,
                delta=f"{significant_tests / total_tests:.1%}" if total_tests > 0 else "0%",
            )

        with col3:
            st.metric("Bidirectional", bidirectional)

        # Results table
        st.subheader("📋 Granger Causality Results")

        # Convert results to dataframe
        granger_data = []
        for result in results:
            granger_data.append(
                {
                    "Cause": result.cause,
                    "Effect": result.effect,
                    "F-Statistic": f"{result.f_statistic:.4f}",
                    "P-Value": f"{result.p_value:.4f}",
                    "Is Causal": "✅" if result.is_causal else "❌",
                    "Direction": result.direction.title(),
                    "Optimal Lag": result.lag_order,
                    "AIC Score": f"{result.aic_score:.2f}",
                }
            )

        granger_df = pd.DataFrame(granger_data)
        st.dataframe(granger_df, use_container_width=True)

        # Visualization
        st.subheader("🕸️ Causality Network")

        # Create network visualization
        significant_results = [r for r in results if r.is_causal]

        if significant_results:
            self._plot_causality_network(significant_results)
        else:
            st.info("No significant Granger causal relationships found.")

        # Statistical distribution
        st.subheader("📊 Statistical Distribution")

        p_values = [r.p_value for r in results]
        f_statistics = [r.f_statistic for r in results]

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["P-Value Distribution", "F-Statistic Distribution"]
        )

        fig.add_trace(go.Histogram(x=p_values, nbinsx=20, name="P-Values"), row=1, col=1)

        fig.add_trace(go.Histogram(x=f_statistics, nbinsx=20, name="F-Statistics"), row=1, col=2)

        fig.add_vline(
            x=0.05, line_dash="dash", line_color="red", annotation_text="α=0.05", row=1, col=1
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_counterfactual_tab(self):
        """Render counterfactual analysis tab"""
        st.header("🔮 Counterfactual Analysis")
        st.markdown("Explore 'what-if' scenarios and intervention effects")

        # Check if results are available
        if "counterfactual_results" not in st.session_state:
            st.info("🔍 Run 'Generate Counterfactuals' from the sidebar to see results")

            # Explain counterfactual analysis
            st.subheader("🤔 What are Counterfactuals?")

            st.markdown("""
            **Counterfactual Analysis** answers "What if?" questions:
            
            - **What if** volume increased by 50%?
            - **What if** sentiment was more positive?
            - **What if** whale activity decreased?
            
            **Key Benefits:**
            - Understand intervention effects before acting
            - Quantify potential outcomes of trading decisions
            - Risk assessment for different scenarios
            - Strategic planning and optimization
            """)

            # Demo scenario
            st.subheader("💡 Example Scenarios")

            scenarios = [
                "Volume increases by 100% → Price impact?",
                "Sentiment score improves to 0.8 → Market response?",
                "Whale activity drops by 50% → Volatility change?",
                "RSI reaches oversold (30) → Recovery probability?",
            ]

            for scenario in scenarios:
                st.info(f"🔮 {scenario}")

            return

        # Display results
        results = st.session_state["counterfactual_results"]

        if not results:
            st.warning("No counterfactual results available.")
            return

        # Summary metrics
        total_scenarios = len(results)
        high_confidence = len([r for r in results if r.confidence > 0.7])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Scenarios", total_scenarios)

        with col2:
            st.metric(
                "High Confidence",
                high_confidence,
                delta=f"{high_confidence / total_scenarios:.1%}" if total_scenarios > 0 else "0%",
            )

        with col3:
            avg_confidence = np.mean([r.confidence for r in results])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

        # Scenarios table
        st.subheader("📋 Counterfactual Scenarios")

        # Convert results to dataframe
        cf_data = []
        for result in results:
            cf_data.append(
                {
                    "Scenario ID": result.scenario_id,
                    "Treatment": result.treatment_variable,
                    "Original Value": f"{result.original_value:.4f}",
                    "Counterfactual Value": f"{result.counterfactual_value:.4f}",
                    "Predicted Outcome": f"{result.predicted_outcome:.4f}",
                    "Confidence": f"{result.confidence:.1%}",
                    "Explanation": result.explanation[:80] + "..."
                    if len(result.explanation) > 80
                    else result.explanation,
                }
            )

        cf_df = pd.DataFrame(cf_data)
        st.dataframe(cf_df, use_container_width=True)

        # Visualization
        st.subheader("📊 Counterfactual Effects")

        # Group by treatment variable
        treatment_effects = {}
        for result in results:
            if result.treatment_variable not in treatment_effects:
                treatment_effects[result.treatment_variable] = []

            effect = result.predicted_outcome - result.original_value
            treatment_effects[result.treatment_variable].append(
                {
                    "effect": effect,
                    "confidence": result.confidence,
                    "cf_value": result.counterfactual_value,
                    "original": result.original_value,
                }
            )

        # Plot effects by treatment
        fig = go.Figure()

        for treatment, effects in treatment_effects.items():
            cf_values = [e["cf_value"] for e in effects]
            effect_sizes = [e["effect"] for e in effects]
            confidences = [e["confidence"] for e in effects]

            fig.add_trace(
                go.Scatter(
                    x=cf_values,
                    y=effect_sizes,
                    mode="markers",
                    name=treatment,
                    marker=dict(
                        size=[c * 20 + 5 for c in confidences],  # Size by confidence
                        opacity=0.6,
                    ),
                    text=[f"Confidence: {c:.1%}" for c in confidences],
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Intervention Value: %{x:.4f}<br>"
                    + "Effect Size: %{y:.4f}<br>"
                    + "%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Counterfactual Effects by Treatment Variable",
            xaxis_title="Intervention Value",
            yaxis_title="Predicted Effect Size",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # High-confidence scenarios
        st.subheader("⭐ High-Confidence Scenarios")

        high_conf_scenarios = [r for r in results if r.confidence > 0.7]

        if high_conf_scenarios:
            for scenario in high_conf_scenarios[:5]:  # Show top 5
                with st.expander(
                    f"🎯 {scenario.scenario_id} (Confidence: {scenario.confidence:.1%})"
                ):
                    st.write(f"**Treatment:** {scenario.treatment_variable}")
                    st.write(f"**Original Value:** {scenario.original_value:.4f}")
                    st.write(f"**Counterfactual Value:** {scenario.counterfactual_value:.4f}")
                    st.write(f"**Predicted Outcome:** {scenario.predicted_outcome:.4f}")
                    st.write(f"**Explanation:** {scenario.explanation}")
        else:
            st.info("No high-confidence scenarios found.")

    def _render_price_explanation_tab(self):
        """Render price movement explanation tab"""
        st.header("💡 Price Movement Explanations")
        st.markdown("Understand WHY significant price movements happened")

        # Demo explanation
        st.subheader("🔍 Example: Explaining BTC Price Movement")

        # Simulated explanation
        st.markdown("""
        **Date:** 2025-08-08 12:00 UTC  
        **Movement:** BTC/EUR +5.2% increase  
        **Timeframe:** 1-hour candle
        """)

        # Causal factors
        st.subheader("🎯 Identified Causal Factors")

        factors = [
            {
                "factor": "Volume Surge",
                "contribution": 0.68,
                "description": "Trading volume increased 340% above average",
                "confidence": 0.89,
            },
            {
                "factor": "Positive Sentiment",
                "contribution": 0.42,
                "description": "Social sentiment improved to 0.78 (from 0.52)",
                "confidence": 0.76,
            },
            {
                "factor": "Whale Activity",
                "contribution": 0.31,
                "description": "Large buy orders detected (>$50M)",
                "confidence": 0.82,
            },
            {
                "factor": "Technical Breakout",
                "contribution": 0.25,
                "description": "Price broke above resistance at €42,500",
                "confidence": 0.71,
            },
        ]

        for factor in factors:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**{factor['factor']}**")
                st.write(factor["description"])

            with col2:
                st.metric("Contribution", f"{factor['contribution']:.2f}")

            with col3:
                confidence_color = (
                    "green"
                    if factor["confidence"] > 0.8
                    else "orange"
                    if factor["confidence"] > 0.6
                    else "red"
                )
                st.markdown(f"**Confidence:** :{confidence_color}[{factor['confidence']:.1%}]")

        # Visualization
        st.subheader("📊 Factor Contribution Analysis")

        factor_names = [f["factor"] for f in factors]
        contributions = [f["contribution"] for f in factors]
        confidences = [f["confidence"] for f in factors]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=factor_names,
                y=contributions,
                marker=dict(
                    color=confidences,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Confidence"),
                ),
                text=[f"{c:.2f}" for c in contributions],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Causal Factor Contributions to Price Movement",
            xaxis_title="Factors",
            yaxis_title="Contribution Score",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Counterfactual scenarios
        st.subheader("🔮 What If Scenarios")

        st.markdown("**What would have happened if these factors were different?**")

        scenarios = [
            "If volume stayed normal → Price would have increased only +1.8%",
            "If sentiment remained negative → Price would have decreased -0.5%",
            "If no whale activity → Price change would be +2.3%",
            "If technical resistance held → Price would be flat (0.0%)",
        ]

        for scenario in scenarios:
            st.info(f"🔮 {scenario}")

    def _render_results_tab(self):
        """Render comprehensive analysis results tab"""
        st.header("📊 Comprehensive Analysis Results")
        st.markdown("Summary of all causal inference analyses")

        # Get analysis summary
        try:
            summary = self.causal_engine.get_analysis_summary()

            # Overall metrics
            st.subheader("📈 Analysis Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_effects = summary["causal_effects"]["total"]
                st.metric("Causal Effects", total_effects)

            with col2:
                significant_effects = summary["causal_effects"]["significant"]
                st.metric(
                    "Significant Effects",
                    significant_effects,
                    delta=f"{significant_effects / total_effects:.1%}"
                    if total_effects > 0
                    else "0%",
                )

            with col3:
                granger_tests = summary["granger_causality"]["total_tests"]
                st.metric("Granger Tests", granger_tests)

            with col4:
                cf_scenarios = summary["counterfactual_predictions"]["total_scenarios"]
                st.metric("Counterfactuals", cf_scenarios)

            # Recent analyses
            st.subheader("🕒 Recent Analyses")

            # Causal effects
            recent_effects = summary["causal_effects"].get("recent", [])
            if recent_effects:
                st.write("**Recent Causal Effects:**")
                for effect in recent_effects:
                    significance = "✅" if effect["significance"] else "❌"
                    st.write(
                        f"- {effect['treatment']} → {effect['outcome']}: {effect['effect_size']:.4f} {significance}"
                    )

            # Granger results
            recent_granger = summary["granger_causality"].get("recent", [])
            if recent_granger:
                st.write("**Recent Granger Tests:**")
                for result in recent_granger:
                    causality = "✅" if result["is_causal"] else "❌"
                    st.write(
                        f"- {result['cause']} → {result['effect']}: {result['direction']} {causality}"
                    )

            # Counterfactuals
            recent_cf = summary["counterfactual_predictions"].get("recent", [])
            if recent_cf:
                st.write("**Recent Counterfactuals:**")
                for cf in recent_cf:
                    st.write(f"- {cf['treatment_variable']}: {cf['confidence']:.1%} confidence")

            # Analysis capabilities
            st.subheader("🎯 Analysis Capabilities")

            capabilities = [
                "🔍 **Double Machine Learning:** Robust causal effect estimation with confounders",
                "📈 **Granger Causality:** Temporal causality testing for time series",
                "🔮 **Counterfactual Prediction:** What-if scenario analysis",
                "💡 **Price Explanations:** Understanding why movements happen",
                "🕸️ **Causal Networks:** Visualizing market relationships",
                "📊 **Statistical Testing:** Rigorous significance testing",
                "⚡ **Real-time Analysis:** Continuous causality monitoring",
                "🎯 **Market Regime Aware:** Context-dependent causality",
            ]

            for capability in capabilities:
                st.markdown(capability)

        except Exception as e:
            st.error(f"Error loading analysis results: {e}")

            # Fallback content
            st.info(
                "No analysis results available yet. Run analyses from the sidebar to see comprehensive results."
            )

    def _generate_demo_data(self) -> pd.DataFrame:
        """Generate demo cryptocurrency market data"""
        np.random.seed(42)

        # Generate 1000 time points
        n_points = 1000
        dates = pd.date_range(start="2024-01-01", periods=n_points, freq="H")

        # Generate correlated time series
        # Base price movements
        btc_returns = np.random.normal(0, 0.02, n_points)
        btc_returns = np.cumsum(btc_returns)
        btc_price = 40000 + btc_returns * 1000

        # Correlated altcoin
        eth_returns = 0.7 * btc_returns + np.random.normal(0, 0.015, n_points)
        eth_price = 2500 + eth_returns * 100

        # Volume with some causality
        volume_btc = 1000 + 50 * np.abs(btc_returns) + np.random.normal(0, 100, n_points)
        volume_eth = 800 + 40 * np.abs(eth_returns) + np.random.normal(0, 80, n_points)

        # Sentiment (lagged effect on price)
        sentiment = np.zeros(n_points)
        for i in range(1, n_points):
            sentiment[i] = (
                0.5 + 0.3 * sentiment[i - 1] + 0.1 * btc_returns[i - 1] + np.random.normal(0, 0.1)
            )
        sentiment = np.clip(sentiment, 0, 1)

        # Whale activity (causes volume spikes)
        whale_activity = np.random.exponential(0.1, n_points)
        volume_btc += whale_activity * 200
        volume_eth += whale_activity * 150

        # Technical indicators
        rsi_btc = 50 + 30 * np.sin(np.arange(n_points) * 0.1) + np.random.normal(0, 5, n_points)
        rsi_btc = np.clip(rsi_btc, 0, 100)

        rsi_eth = 50 + 25 * np.sin(np.arange(n_points) * 0.12) + np.random.normal(0, 5, n_points)
        rsi_eth = np.clip(rsi_eth, 0, 100)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "btc_price": btc_price,
                "eth_price": eth_price,
                "btc_volume": volume_btc,
                "eth_volume": volume_eth,
                "sentiment_score": sentiment,
                "whale_activity": whale_activity,
                "btc_rsi": rsi_btc,
                "eth_rsi": rsi_eth,
            }
        )

        # Add price changes
        data["btc_price_change"] = data["btc_price"].pct_change()
        data["eth_price_change"] = data["eth_price"].pct_change()
        data["btc_volume_change"] = data["btc_volume"].pct_change()
        data["eth_volume_change"] = data["eth_volume"].pct_change()

        # Drop NaN values
        data = data.dropna()

        return data

    def _plot_causality_network(self, granger_results):
        """Plot network visualization of Granger causality relationships"""
        import networkx as nx

        # Create network graph
        G = nx.DiGraph()

        # Add edges for significant relationships
        for result in granger_results:
            if result.is_causal:
                # Edge weight based on F-statistic
                weight = min(result.f_statistic / 10, 5)  # Normalize weight
                G.add_edge(result.cause, result.effect, weight=weight, f_stat=result.f_statistic)

                if result.direction == "bidirectional":
                    G.add_edge(
                        result.effect, result.cause, weight=weight, f_stat=result.f_statistic
                    )

        if len(G.edges()) == 0:
            st.info("No significant relationships to visualize.")
            return

        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Extract edge information
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]["weight"])

        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="lightblue"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            # Node info with degree
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_info.append(f"{node}<br>In: {in_degree}, Out: {out_degree}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            hoverinfo="text",
            marker=dict(size=30, color="lightcoral", line=dict(width=2, color="darkred")),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Granger Causality Network",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Arrows show causal direction",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)


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
