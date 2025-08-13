"""
CryptoSmartTrader V2 - Advanced Analytics Dashboard
Geavanceerde functies dashboard voor perfecte analyse van snelle groeiers
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AdvancedAnalyticsDashboard:
    """Geavanceerde analytics dashboard voor alle nieuwe functies"""

    def __init__(self, container):
        self.container = container

    def render(self):
        """Render advanced analytics dashboard"""
        st.set_page_config(
            page_title="Advanced Analytics - CryptoSmartTrader V2", page_icon="🧠", layout="wide"
        )

        st.title("🧠 Advanced Analytics Engine")
        st.markdown("**Geavanceerde functies voor perfecte analyse van snelle groeiers**")

        # Create tabs for different advanced features
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🔗 On-Chain Analytics",
                "📚 Order Book Analysis",
                "📰 News & Events",
                "🔍 Explainable AI",
                "⚡ Performance Optimizer",
            ]
        )

        with tab1:
            self._render_onchain_analytics()

        with tab2:
            self._render_orderbook_analysis()

        with tab3:
            self._render_news_events()

        with tab4:
            self._render_explainable_ai()

        with tab5:
            self._render_performance_optimizer()

    def _render_onchain_analytics(self):
        """Render on-chain analytics section"""
        st.header("🔗 On-Chain Analytics")
        st.markdown(
            "**Beyond basic whale detection: Smart money tracking, holder analysis, DEX analytics**"
        )

        # Control panel
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_coin = st.selectbox(
                "Select Coin for Analysis", ["BTC", "ETH", "ADA", "SOL", "MATIC"]
            )

        with col2:
            if st.button("🔄 Run On-Chain Analysis", type="primary"):
                st.rerun()

        with col3:
            analysis_type = st.selectbox(
                "Analysis Type", ["All", "Smart Money", "Holder Distribution", "DEX Activity"]
            )

        try:
            # Get advanced analytics engine
            advanced_analytics = self.container.advanced_analytics()

            # Get cached analysis data
            analysis_data = advanced_analytics.get_advanced_analysis(selected_coin)

            if analysis_data:
                # Display on-chain signals
                onchain_signals = analysis_data.get("onchain_signals", [])

                if onchain_signals:
                    st.subheader("📊 On-Chain Signals Detected")

                    signal_data = []
                    for signal in onchain_signals:
                        signal_data.append(
                            {
                                "Signal Type": signal.signal_type.replace("_", " ").title(),
                                "Strength": f"{signal.strength:.1%}",
                                "Timestamp": signal.timestamp.strftime("%H:%M:%S"),
                                "Details": str(signal.metadata),
                            }
                        )

                    if signal_data:
                        df_signals = pd.DataFrame(signal_data)
                        st.dataframe(df_signals, use_container_width=True)

                    # Visualize signal strengths
                    signal_types = [s.signal_type for s in onchain_signals]
                    signal_strengths = [s.strength for s in onchain_signals]

                    if signal_types:
                        fig_signals = px.bar(
                            x=signal_types,
                            y=signal_strengths,
                            title=f"On-Chain Signal Strengths for {selected_coin}",
                            labels={"x": "Signal Type", "y": "Strength"},
                        )
                        st.plotly_chart(fig_signals, use_container_width=True)
                else:
                    st.info("No on-chain signals detected for this coin yet")
            else:
                st.warning(
                    "No analysis data available. The advanced analytics engine may still be collecting data."
                )

        except Exception as e:
            st.error(f"On-chain analysis error: {e}")

        # Smart Money Tracker
        with st.expander("💎 Smart Money Flow Analysis"):
            st.markdown("""
            **Smart Money Features:**
            - Track known VC and institutional addresses
            - Detect accumulation patterns from smart money wallets
            - Monitor cross-chain flows and bridge activities
            - Analyze wallet clustering and behavior patterns
            """)

            # Simulate smart money data
            smart_money_data = pd.DataFrame(
                {
                    "Address Type": [
                        "Venture Capital",
                        "Institutional",
                        "Whale Wallet",
                        "Bridge Contract",
                    ],
                    "Net Flow (24h)": ["+$2.3M", "+$1.8M", "-$850K", "+$5.2M"],
                    "Transaction Count": [23, 12, 8, 156],
                    "Confidence": ["High", "High", "Medium", "High"],
                }
            )

            st.dataframe(smart_money_data, use_container_width=True)

        # Holder Distribution Analysis
        with st.expander("👥 Holder Distribution Dynamics"):
            st.markdown("""
            **Holder Analysis Features:**
            - Track whale accumulation/distribution patterns
            - Monitor new large holders entering/exiting
            - Analyze wallet age and holding patterns
            - Detect coordinated movements
            """)

            # Create holder distribution chart
            holders = ["Top 1%", "Top 5%", "Top 10%", "Top 25%", "Others"]
            distribution = [25, 35, 45, 65, 100]

            fig_distribution = px.pie(
                values=distribution, names=holders, title=f"{selected_coin} Holder Distribution"
            )
            st.plotly_chart(fig_distribution, use_container_width=True)

    def _render_orderbook_analysis(self):
        """Render order book analysis section"""
        st.header("📚 Order Book & Depth Analysis")
        st.markdown("**Real-time detection of spoofing, thin liquidity, and large walls**")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_coin = st.selectbox(
                "Select Coin", ["BTC", "ETH", "ADA", "SOL"], key="orderbook_coin"
            )

        with col2:
            exchange = st.selectbox(
                "Exchange", ["kraken", "binance", "kucoin"], key="orderbook_exchange"
            )

        with col3:
            if st.button("📊 Analyze Order Book"):
                st.rerun()

        try:
            # Get order book analyzer
            advanced_analytics = self.container.advanced_analytics()
            analysis_data = advanced_analytics.get_advanced_analysis(selected_coin)

            if analysis_data:
                orderbook_anomalies = analysis_data.get("orderbook_anomalies", [])

                if orderbook_anomalies:
                    st.subheader("🚨 Order Book Anomalies Detected")

                    for anomaly in orderbook_anomalies:
                        severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                            "high"
                            if anomaly.severity > 0.7
                            else "medium"
                            if anomaly.severity > 0.4
                            else "low",
                            "🟢",
                        )

                        st.write(
                            f"{severity_color} **{anomaly.anomaly_type.replace('_', ' ').title()}** - Severity: {anomaly.severity:.1%}"
                        )
                        st.json(anomaly.details)
                else:
                    st.success("No order book anomalies detected - healthy market structure")
            else:
                st.info("Order book analysis data not available")

        except Exception as e:
            st.error(f"Order book analysis error: {e}")

        # Order Book Visualization
        with st.expander("📈 Order Book Visualization"):
            # Simulate order book data
            bid_prices = np.linspace(45000, 45500, 20)
            bid_sizes = np.random.exponential(2, 20)
            ask_prices = np.linspace(45500, 46000, 20)
            ask_sizes = np.random.exponential(2, 20)

            fig_orderbook = go.Figure()

            # Bids (green)
            fig_orderbook.add_trace(
                go.Bar(
                    x=bid_sizes,
                    y=bid_prices,
                    orientation="h",
                    name="Bids",
                    marker_color="green",
                    opacity=0.7,
                )
            )

            # Asks (red)
            fig_orderbook.add_trace(
                go.Bar(
                    x=-ask_sizes,  # Negative for left side
                    y=ask_prices,
                    orientation="h",
                    name="Asks",
                    marker_color="red",
                    opacity=0.7,
                )
            )

            fig_orderbook.update_layout(
                title=f"{selected_coin} Order Book Depth",
                xaxis_title="Size",
                yaxis_title="Price",
                barmode="overlay",
            )

            st.plotly_chart(fig_orderbook, use_container_width=True)

        # Liquidity Analysis
        with st.expander("💧 Liquidity & Slippage Analysis"):
            st.markdown("""
            **Liquidity Features:**
            - Real-time slippage calculations for different order sizes
            - Depth analysis and liquidity risk assessment
            - Market impact predictions
            - Optimal execution strategies
            """)

            # Slippage simulation
            order_sizes = [1000, 5000, 10000, 25000, 50000]
            slippage = [0.1, 0.3, 0.8, 2.1, 4.5]

            fig_slippage = px.line(
                x=order_sizes,
                y=slippage,
                title="Estimated Slippage by Order Size",
                labels={"x": "Order Size ($)", "y": "Slippage (%)"},
            )
            st.plotly_chart(fig_slippage, use_container_width=True)

    def _render_news_events(self):
        """Render news and events section"""
        st.header("📰 News & Event Tracking")
        st.markdown("**Automated scraping and impact analysis from multiple sources**")

        # Controls
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_coin = st.selectbox(
                "Select Coin", ["BTC", "ETH", "ADA", "SOL"], key="news_coin"
            )

        with col2:
            if st.button("📡 Fetch Latest News"):
                st.rerun()

        try:
            # Get news tracker
            advanced_analytics = self.container.advanced_analytics()
            analysis_data = advanced_analytics.get_advanced_analysis(selected_coin)

            if analysis_data:
                news_events = analysis_data.get("news_events", [])

                if news_events:
                    st.subheader("📈 Recent News Events")

                    for event in news_events[:5]:  # Show top 5
                        # Determine sentiment color
                        sentiment_color = (
                            "🟢"
                            if event.sentiment > 0.3
                            else "🔴"
                            if event.sentiment < -0.3
                            else "🟡"
                        )

                        with st.container():
                            st.write(f"{sentiment_color} **{event.title}**")

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"Source: {event.source}")
                            with col2:
                                st.write(f"Impact: {event.impact_score:.1%}")
                            with col3:
                                st.write(f"Sentiment: {event.sentiment:.2f}")

                            if event.content:
                                with st.expander("Read More"):
                                    st.write(event.content[:200] + "...")

                            st.markdown("---")
                else:
                    st.info("No recent news events found for this coin")
            else:
                st.info("News analysis data not available")

        except Exception as e:
            st.error(f"News tracking error: {e}")

        # News Sources Configuration
        with st.expander("📡 News Sources & Configuration"):
            st.markdown("""
            **Tracked News Sources:**
            - CoinDesk API
            - CoinTelegraph RSS
            - CryptoPanic API
            - Reddit Crypto Communities
            - Twitter/X Crypto Influencers
            - Official Project Announcements
            - GitHub Repository Activities
            """)

            # News impact analysis
            sources = ["CoinDesk", "CoinTelegraph", "CryptoPanic", "Reddit", "Twitter"]
            impact_scores = [0.8, 0.7, 0.6, 0.5, 0.4]

            fig_sources = px.bar(
                x=sources,
                y=impact_scores,
                title="News Source Impact Weights",
                labels={"x": "Source", "y": "Impact Weight"},
            )
            st.plotly_chart(fig_sources, use_container_width=True)

    def _render_explainable_ai(self):
        """Render explainable AI section"""
        st.header("🔍 Explainable AI")
        st.markdown("**SHAP-based feature importance and prediction explanations**")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_coin = st.selectbox("Select Coin", ["BTC", "ETH", "ADA", "SOL"], key="ai_coin")

        with col2:
            horizon = st.selectbox("Time Horizon", ["1H", "24H", "7D", "30D"], key="ai_horizon")

        with col3:
            if st.button("🧠 Explain Prediction"):
                st.rerun()

        try:
            # Get prediction explainer
            explainer = self.container.prediction_explainer()

            # Get sample features for demonstration
            sample_features = {
                "price_change_24h": 0.15,
                "volume_spike": 2.3,
                "sentiment_score": 0.7,
                "whale_activity": 0.8,
                "rsi_14": 65,
                "macd_signal": 0.4,
                "momentum_strength": 0.6,
                "news_sentiment": 0.5,
                "onchain_activity": 0.75,
            }

            prediction = 1.25  # 25% predicted increase

            # Get explanation
            explanation = explainer.explain_prediction(
                coin=selected_coin, prediction=prediction, horizon=horizon, features=sample_features
            )

            # Display explanation
            st.subheader("🎯 Prediction Explanation")

            # Prediction summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Change", f"{((prediction - 1) * 100):+.1f}%")

            with col2:
                st.metric("Confidence", f"{explanation['confidence']:.1%}")

            with col3:
                pred_range = explanation["prediction_range"]
                uncertainty = pred_range["uncertainty"]
                st.metric("Uncertainty", f"±{uncertainty:.1%}")

            # Feature importance
            st.subheader("📊 Feature Importance")

            feature_importance = explanation["feature_importance"]
            if feature_importance:
                # Create feature importance chart
                features = [f["description"] for f in feature_importance[:8]]  # Top 8
                importances = [f["importance"] for f in feature_importance[:8]]
                directions = [f["direction"] for f in feature_importance[:8]]

                colors = ["green" if d == "positive" else "red" for d in directions]

                fig_importance = px.bar(
                    x=importances,
                    y=features,
                    orientation="h",
                    title="Top Feature Importance",
                    color=colors,
                    color_discrete_map={"green": "lightgreen", "red": "lightcoral"},
                )
                fig_importance.update_layout(showlegend=False)
                st.plotly_chart(fig_importance, use_container_width=True)

            # Human explanation
            st.subheader("💬 AI Explanation")
            st.write(explanation["human_explanation"])

            # Key factors
            if explanation["key_factors"]:
                st.subheader("🔑 Key Factors")
                for factor in explanation["key_factors"]:
                    st.write(f"• {factor}")

            # Risk factors
            if explanation["risk_factors"]:
                st.subheader("⚠️ Risk Factors")
                for risk in explanation["risk_factors"]:
                    st.write(f"• {risk}")

            # Scenario analysis
            if explanation["scenario_analysis"]:
                st.subheader("🎲 What-If Scenarios")

                scenario_data = []
                for scenario_name, scenario in explanation["scenario_analysis"].items():
                    scenario_data.append(
                        {
                            "Scenario": scenario["description"],
                            "Impact": f"{((scenario['impact'] - 1) * 100):+.1f}%",
                            "Probability": f"{scenario['probability']:.1%}",
                        }
                    )

                if scenario_data:
                    df_scenarios = pd.DataFrame(scenario_data)
                    st.dataframe(df_scenarios, use_container_width=True)

        except Exception as e:
            st.error(f"Explainable AI error: {e}")

            # Show fallback explanation
            st.write("**Simplified Explanation Available:**")
            st.write(
                f"The model predicts a {((prediction - 1) * 100):+.1f}% change for {selected_coin} over {horizon}."
            )

    def _render_performance_optimizer(self):
        """Render performance optimizer section"""
        st.header("⚡ Performance Optimizer")
        st.markdown("**Real-time monitoring with automatic optimization**")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.selectbox(
                "Optimization Focus",
                ["All Systems", "Memory", "CPU", "Cache", "API", "ML"],
                key="opt_focus",
            )

        with col2:
            if st.button("🔄 Refresh Metrics"):
                st.rerun()

        with col3:
            if st.button("🛠️ Auto-Optimize", type="primary"):
                try:
                    optimizer = self.container.performance_optimizer()
                    result = optimizer.apply_automatic_optimizations()

                    if result["status"] == "success":
                        st.success(f"Applied {len(result['optimizations'])} optimizations")
                        for opt in result["optimizations"]:
                            st.write(f"✅ {opt['description']}")
                    else:
                        st.warning("No optimizations applied")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

        try:
            # Get performance monitor
            optimizer = self.container.performance_optimizer()
            performance_summary = optimizer.monitor.get_performance_summary()

            if performance_summary.get("status") != "error":
                # Performance overview
                st.subheader("📊 System Performance Overview")

                col1, col2, col3, col4, col5 = st.columns(5)

                scores = performance_summary.get("scores", {})

                with col1:
                    cpu_score = scores.get("cpu", 0)
                    st.metric(
                        "CPU Health",
                        f"{cpu_score:.0f}%",
                        delta=f"{cpu_score - 75:.0f}" if cpu_score != 75 else None,
                    )

                with col2:
                    memory_score = scores.get("memory", 0)
                    st.metric(
                        "Memory Health",
                        f"{memory_score:.0f}%",
                        delta=f"{memory_score - 75:.0f}" if memory_score != 75 else None,
                    )

                with col3:
                    cache_score = scores.get("cache", 0)
                    st.metric(
                        "Cache Efficiency",
                        f"{cache_score:.0f}%",
                        delta=f"{cache_score - 80:.0f}" if cache_score != 80 else None,
                    )

                with col4:
                    api_score = scores.get("api", 0)
                    st.metric(
                        "API Performance",
                        f"{api_score:.0f}%",
                        delta=f"{api_score - 70:.0f}" if api_score != 70 else None,
                    )

                with col5:
                    overall_score = performance_summary.get("overall_score", 0)
                    status = performance_summary.get("status", "unknown")
                    st.metric("Overall Score", f"{overall_score:.0f}%", delta=status.title())

                # Performance trends
                st.subheader("📈 Performance Trends")

                # Create sample performance data
                hours = list(range(24))
                cpu_trend = [75 + 10 * np.sin(h / 4) + np.random.normal(0, 3) for h in hours]
                memory_trend = [60 + 15 * np.sin(h / 3 + 1) + np.random.normal(0, 4) for h in hours]

                fig_trends = go.Figure()

                fig_trends.add_trace(
                    go.Scatter(
                        x=hours,
                        y=cpu_trend,
                        mode="lines+markers",
                        name="CPU Usage (%)",
                        line=dict(color="red"),
                    )
                )

                fig_trends.add_trace(
                    go.Scatter(
                        x=hours,
                        y=memory_trend,
                        mode="lines+markers",
                        name="Memory Usage (%)",
                        line=dict(color="blue"),
                    )
                )

                fig_trends.update_layout(
                    title="24-Hour Performance Trends",
                    xaxis_title="Hours Ago",
                    yaxis_title="Usage (%)",
                )

                st.plotly_chart(fig_trends, use_container_width=True)

                # Optimization recommendations
                st.subheader("💡 Optimization Recommendations")

                recommendations = optimizer.get_optimization_recommendations()
                if recommendations:
                    for rec in recommendations[:5]:  # Top 5
                        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                            rec.priority, "🟢"
                        )
                        st.write(f"{priority_color} **{rec.description}**")
                        st.write(f"📋 Implementation: {rec.implementation}")
                        st.write(f"📈 Expected improvement: {rec.expected_improvement:.1f}%")
                        st.write(f"⏱️ Effort: {rec.estimated_effort}")
                        st.markdown("---")
                else:
                    st.success("🎉 System is running optimally - no recommendations needed")

                # Active alerts
                active_alerts = performance_summary.get("active_alerts", 0)
                if active_alerts > 0:
                    st.subheader("🚨 Active Performance Alerts")
                    st.warning(f"{active_alerts} high-priority alerts require attention")

            else:
                st.error("Performance monitoring not available")

        except Exception as e:
            st.error(f"Performance monitoring error: {e}")


# Main function for standalone usage
def main():
    """Main dashboard function"""
    try:
        # Import container
        from containers import ApplicationContainer

        # Initialize container
        container = ApplicationContainer()
        container.wire(modules=[__name__])

        # Initialize dashboard
        dashboard = AdvancedAnalyticsDashboard(container)
        dashboard.render()

    except Exception as e:
        st.error(f"Dashboard initialization failed: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
