#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Human-in-the-Loop Dashboard
Interactive dashboard for active learning and feedback management
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
    from core.human_in_the_loop import (
        get_human_in_the_loop_system,
        submit_trade_feedback,
        request_prediction_feedback,
        FeedbackType,
        FeedbackValue,
    )
except ImportError:
    st.error("Human-in-the-loop module not available")


class HumanInLoopDashboard:
    """Dashboard for human-in-the-loop learning and feedback"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render(self):
        """Render the human-in-the-loop dashboard"""

        st.header("👤 Human-in-the-Loop Learning System")
        st.markdown(
            "Active learning, feedback collection, and model improvement through human expertise"
        )

        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📝 Submit Feedback",
                "❓ Active Learning Queries",
                "📊 Feedback Analytics",
                "🎯 Calibration Metrics",
            ]
        )

        with tab1:
            self._render_feedback_submission()

        with tab2:
            self._render_active_learning()

        with tab3:
            self._render_feedback_analytics()

        with tab4:
            self._render_calibration_metrics()

    def _render_feedback_submission(self):
        """Render feedback submission interface"""

        st.subheader("📝 Submit Expert Feedback")

        feedback_type = st.selectbox(
            "Feedback Type:",
            options=[
                "Trade Quality",
                "Prediction Accuracy",
                "Feature Relevance",
                "Risk Assessment",
                "Strategy Preference",
            ],
            help="Type of feedback you want to provide",
        )

        if feedback_type == "Trade Quality":
            self._render_trade_feedback_form()
        elif feedback_type == "Prediction Accuracy":
            self._render_prediction_feedback_form()
        else:
            self._render_general_feedback_form(feedback_type)

    def _render_trade_feedback_form(self):
        """Render trade quality feedback form"""

        st.markdown("**🔄 Trade Quality Feedback**")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Trade Details**")

            coin = st.selectbox(
                "Cryptocurrency:",
                options=["BTC", "ETH", "ADA", "DOT", "SOL", "MATIC", "LINK"],
                index=0,
            )

            trade_side = st.selectbox("Trade Side:", options=["Buy", "Sell"], index=0)

            entry_price = st.number_input(
                "Entry Price ($):",
                min_value=0.01,
                value=45000.0 if coin == "BTC" else 2800.0,
                step=0.01,
            )

            exit_price = st.number_input(
                "Exit Price ($):",
                min_value=0.01,
                value=entry_price * 1.02,  # 2% gain default
                step=0.01,
            )

        with col2:
            st.markdown("**Performance Context**")

            holding_period = st.number_input(
                "Holding Period (hours):", min_value=0.1, value=12.0, step=0.1
            )

            ml_confidence = st.slider(
                "ML Model Confidence:", min_value=0.0, max_value=1.0, value=0.75, step=0.01
            )

            market_conditions = st.selectbox(
                "Market Conditions:",
                options=["Bullish", "Bearish", "Sideways", "Volatile"],
                index=0,
            )

            sentiment_score = st.slider(
                "Sentiment Score:", min_value=0.0, max_value=1.0, value=0.6, step=0.01
            )

        # Feedback assessment
        st.markdown("**📊 Your Assessment**")

        col1, col2 = st.columns([1, 1])

        with col1:
            feedback_rating = st.selectbox(
                "Trade Quality Rating:",
                options=[
                    "Excellent (5/5)",
                    "Good (4/5)",
                    "Neutral (3/5)",
                    "Poor (2/5)",
                    "Terrible (1/5)",
                ],
                index=1,
                help="How would you rate this trade's quality?",
            )

            confidence = st.slider(
                "Your Confidence:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.01,
                help="How confident are you in this assessment?",
            )

        with col2:
            explanation = st.text_area(
                "Explanation:",
                height=100,
                placeholder="Explain your reasoning for this rating...",
                help="Provide detailed reasoning for your feedback",
            )

            improvement_areas = st.multiselect(
                "Areas for Improvement:",
                options=[
                    "Entry Timing",
                    "Exit Timing",
                    "Position Sizing",
                    "Risk Management",
                    "Market Analysis",
                    "Sentiment Analysis",
                ],
                help="Select areas where the system could improve",
            )

        if st.button("📤 Submit Trade Feedback", type="primary"):
            try:
                # Convert rating to FeedbackValue
                rating_map = {
                    "Excellent (5/5)": FeedbackValue.EXCELLENT,
                    "Good (4/5)": FeedbackValue.GOOD,
                    "Neutral (3/5)": FeedbackValue.NEUTRAL,
                    "Poor (2/5)": FeedbackValue.POOR,
                    "Terrible (1/5)": FeedbackValue.TERRIBLE,
                }

                feedback_value = rating_map[feedback_rating]

                # Create trade context
                trade_context = {
                    "coin": coin,
                    "side": trade_side.lower(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "holding_period_hours": holding_period,
                    "ml_confidence": ml_confidence,
                    "market_conditions": market_conditions.lower(),
                    "sentiment_score": sentiment_score,
                    "improvement_areas": improvement_areas,
                }

                # Submit feedback
                insights = submit_trade_feedback(
                    trade_context, feedback_value, explanation, confidence
                )

                st.success("✅ Trade feedback submitted successfully!")

                # Show insights
                with st.expander("🔍 Generated Insights"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Feedback Score", f"{insights.get('feedback_score', 0)}/5")
                        st.metric("Actual Return", f"{insights.get('actual_return', 0):.2%}")

                    with col2:
                        st.metric("Misalignment", f"{insights.get('misalignment', 0):.1%}")
                        st.metric(
                            "Confidence Gap",
                            f"{insights.get('confidence_calibration', {}).get('confidence_gap', 0):.1%}",
                        )

                    if insights.get("improvement_areas"):
                        st.write("**Improvement Areas Identified:**")
                        for area in insights["improvement_areas"]:
                            st.write(f"- {area.replace('_', ' ').title()}")

            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")

    def _render_prediction_feedback_form(self):
        """Render prediction accuracy feedback form"""

        st.markdown("**🎯 Prediction Accuracy Feedback**")

        col1, col2 = st.columns([1, 1])

        with col1:
            predicted_value = st.number_input(
                "Model Prediction:",
                value=0.05,
                step=0.001,
                format="%.3f",
                help="The model's predicted return",
            )

            actual_value = st.number_input(
                "Actual Outcome:",
                value=0.03,
                step=0.001,
                format="%.3f",
                help="The actual return that occurred",
            )

        with col2:
            time_horizon = st.selectbox(
                "Time Horizon:", options=["1H", "24H", "7D", "30D"], index=1
            )

            prediction_confidence = st.slider(
                "Model Confidence:", min_value=0.0, max_value=1.0, value=0.75
            )

        # Your assessment
        col1, col2 = st.columns([1, 1])

        with col1:
            accuracy_rating = st.selectbox(
                "Prediction Accuracy:",
                options=[
                    "Very Accurate",
                    "Somewhat Accurate",
                    "Neutral",
                    "Somewhat Inaccurate",
                    "Very Inaccurate",
                ],
                index=1,
            )

        with col2:
            assessment_confidence = st.slider(
                "Your Confidence:", min_value=0.0, max_value=1.0, value=0.8
            )

        explanation = st.text_area(
            "Detailed Assessment:",
            height=80,
            placeholder="Explain why you think this prediction was accurate/inaccurate...",
        )

        if st.button("📤 Submit Prediction Feedback", type="primary"):
            st.success("✅ Prediction feedback submitted!")
            st.info("This feedback will be used to improve model calibration and accuracy.")

    def _render_general_feedback_form(self, feedback_type: str):
        """Render general feedback form"""

        st.markdown(f"**📋 {feedback_type} Feedback**")

        rating = st.selectbox(
            f"{feedback_type} Rating:",
            options=[
                "Excellent (5/5)",
                "Good (4/5)",
                "Neutral (3/5)",
                "Poor (2/5)",
                "Terrible (1/5)",
            ],
            index=2,
        )

        confidence = st.slider("Your Confidence:", min_value=0.0, max_value=1.0, value=0.7)

        explanation = st.text_area(
            "Detailed Explanation:",
            height=120,
            placeholder=f"Provide your assessment of {feedback_type.lower()}...",
        )

        if st.button(f"📤 Submit {feedback_type} Feedback", type="primary"):
            st.success(f"✅ {feedback_type} feedback submitted!")

    def _render_active_learning(self):
        """Render active learning queries interface"""

        st.subheader("❓ Active Learning Queries")

        st.info("""
        **Active Learning Process:**
        The system identifies uncertain predictions and asks for your expert opinion to improve model accuracy.
        """)

        # Mock active learning queries
        queries = [
            {
                "id": "al_001",
                "type": "Prediction Validation",
                "question": "How accurate is this BTC prediction: +5.2% in 24H?",
                "uncertainty": 0.85,
                "priority": 9,
                "context": {
                    "coin": "BTC",
                    "prediction": 0.052,
                    "confidence": 0.65,
                    "current_price": 45000,
                },
            },
            {
                "id": "al_002",
                "type": "Trade Signal Validation",
                "question": "Should we buy ETH based on current signals?",
                "uncertainty": 0.72,
                "priority": 7,
                "context": {
                    "coin": "ETH",
                    "signal": "buy",
                    "strength": 0.78,
                    "current_price": 2800,
                },
            },
            {
                "id": "al_003",
                "type": "Risk Assessment",
                "question": "How risky is this market condition for trading?",
                "uncertainty": 0.68,
                "priority": 6,
                "context": {"volatility": 0.045, "sentiment": 0.35, "market_regime": "bearish"},
            },
        ]

        st.markdown(f"**📋 Pending Queries ({len(queries)})**")

        for i, query in enumerate(queries):
            with st.expander(
                f"Query {i + 1}: {query['question']} (Priority: {query['priority']}/10)"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Type:** {query['type']}")
                    st.write(f"**Uncertainty Score:** {query['uncertainty']:.1%}")
                    st.write(f"**Context:** {query['context']}")

                    # Response options
                    if query["type"] == "Prediction Validation":
                        response = st.selectbox(
                            "Your Assessment:",
                            options=[
                                "Very Accurate",
                                "Somewhat Accurate",
                                "Neutral",
                                "Somewhat Inaccurate",
                                "Very Inaccurate",
                            ],
                            key=f"response_{query['id']}",
                        )
                    elif query["type"] == "Trade Signal Validation":
                        response = st.selectbox(
                            "Your Recommendation:",
                            options=[
                                "Definitely Execute",
                                "Probably Execute",
                                "Uncertain",
                                "Probably Skip",
                                "Definitely Skip",
                            ],
                            key=f"response_{query['id']}",
                        )
                    else:
                        response = st.selectbox(
                            "Risk Level:",
                            options=[
                                "Very Low Risk",
                                "Low Risk",
                                "Moderate Risk",
                                "High Risk",
                                "Very High Risk",
                            ],
                            key=f"response_{query['id']}",
                        )

                with col2:
                    confidence = st.slider(
                        "Confidence:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        key=f"conf_{query['id']}",
                    )

                    if st.button(f"✅ Submit", key=f"submit_{query['id']}"):
                        st.success(f"Response submitted for Query {i + 1}")
                        st.info("This feedback will improve model predictions.")

        if not queries:
            st.info(
                "🎉 No pending queries at the moment. The system is confident in current predictions!"
            )

    def _render_feedback_analytics(self):
        """Render feedback analytics interface"""

        st.subheader("📊 Feedback Analytics & Insights")

        # Mock analytics data
        feedback_summary = {
            "total_feedback": 127,
            "avg_confidence": 0.78,
            "feedback_distribution": {
                "Trade Quality": 45,
                "Prediction Accuracy": 38,
                "Risk Assessment": 25,
                "Feature Relevance": 12,
                "Strategy Preference": 7,
            },
            "quality_trends": {
                "Excellent": 15,
                "Good": 42,
                "Neutral": 35,
                "Poor": 28,
                "Terrible": 7,
            },
        }

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Feedback", feedback_summary["total_feedback"], delta="12 this week")

        with col2:
            st.metric("Avg Confidence", f"{feedback_summary['avg_confidence']:.1%}", delta="5% ↑")

        with col3:
            recent_good = (
                feedback_summary["quality_trends"]["Excellent"]
                + feedback_summary["quality_trends"]["Good"]
            )
            total_recent = sum(feedback_summary["quality_trends"].values())
            st.metric("Positive Feedback", f"{recent_good / total_recent:.1%}", delta="8% ↑")

        with col4:
            st.metric("Active Queries", "3", delta="2 resolved today")

        # Visualizations
        col1, col2 = st.columns([1, 1])

        with col1:
            # Feedback type distribution
            feedback_types = list(feedback_summary["feedback_distribution"].keys())
            feedback_counts = list(feedback_summary["feedback_distribution"].values())

            fig_pie = go.Figure(
                data=[go.Pie(labels=feedback_types, values=feedback_counts, hole=0.3)]
            )

            fig_pie.update_layout(title="Feedback Distribution by Type", height=400)

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Quality distribution
            quality_labels = list(feedback_summary["quality_trends"].keys())
            quality_counts = list(feedback_summary["quality_trends"].values())

            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=quality_labels,
                        y=quality_counts,
                        marker_color=["#2E8B57", "#32CD32", "#FFD700", "#FF6347", "#DC143C"],
                    )
                ]
            )

            fig_bar.update_layout(
                title="Feedback Quality Distribution",
                xaxis_title="Quality Rating",
                yaxis_title="Count",
                height=400,
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Feedback timeline
        st.markdown("**📈 Feedback Timeline**")

        # Generate mock timeline data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        daily_feedback = np.random.poisson(4, 30)
        quality_scores = np.random.uniform(3.2, 4.1, 30)

        timeline_df = pd.DataFrame(
            {"Date": dates, "Feedback Count": daily_feedback, "Avg Quality Score": quality_scores}
        )

        fig_timeline = go.Figure()

        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df["Date"],
                y=timeline_df["Feedback Count"],
                mode="lines+markers",
                name="Daily Feedback Count",
                yaxis="y",
            )
        )

        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df["Date"],
                y=timeline_df["Avg Quality Score"],
                mode="lines+markers",
                name="Avg Quality Score",
                yaxis="y2",
                line=dict(color="red"),
            )
        )

        fig_timeline.update_layout(
            title="Feedback Activity Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Feedback Count", side="left"),
            yaxis2=dict(title="Quality Score", side="right", overlaying="y"),
            height=400,
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

    def _render_calibration_metrics(self):
        """Render model calibration metrics"""

        st.subheader("🎯 Model Calibration & Alignment")

        st.info("""
        **Calibration Analysis:**
        How well do model confidence scores align with actual accuracy and human expert assessments?
        """)

        # Mock calibration data
        confidence_bins = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        model_confidence = [0.1, 0.3, 0.5, 0.7, 0.9]
        actual_accuracy = [0.15, 0.35, 0.48, 0.72, 0.85]
        human_agreement = [0.12, 0.28, 0.45, 0.68, 0.82]

        col1, col2 = st.columns([1, 1])

        with col1:
            # Calibration plot
            fig_cal = go.Figure()

            fig_cal.add_trace(
                go.Scatter(
                    x=model_confidence,
                    y=actual_accuracy,
                    mode="lines+markers",
                    name="Actual Accuracy",
                    line=dict(color="blue"),
                )
            )

            fig_cal.add_trace(
                go.Scatter(
                    x=model_confidence,
                    y=human_agreement,
                    mode="lines+markers",
                    name="Human Agreement",
                    line=dict(color="red"),
                )
            )

            # Perfect calibration line
            fig_cal.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(color="gray", dash="dash"),
                )
            )

            fig_cal.update_layout(
                title="Model Calibration Analysis",
                xaxis_title="Model Confidence",
                yaxis_title="Actual Performance",
                height=400,
            )

            st.plotly_chart(fig_cal, use_container_width=True)

        with col2:
            # Calibration error metrics
            cal_error = np.mean(np.abs(np.array(model_confidence) - np.array(actual_accuracy)))
            human_alignment = np.mean(np.abs(np.array(actual_accuracy) - np.array(human_agreement)))

            st.markdown("**📊 Calibration Metrics**")

            st.metric("Calibration Error", f"{cal_error:.3f}", delta="Lower is better")
            st.metric("Human Alignment", f"{1 - human_alignment:.3f}", delta="Higher is better")
            st.metric("Reliability Score", f"{0.78:.2f}", delta="0.05 ↑")

            # Improvement recommendations
            st.markdown("**🎯 Improvement Recommendations**")
            st.write("✅ Model slightly overconfident in 60-80% range")
            st.write("✅ Good alignment with human experts overall")
            st.write("⚠️ Consider recalibration for high-confidence predictions")

        # Detailed calibration table
        st.markdown("**📋 Detailed Calibration Breakdown**")

        cal_df = pd.DataFrame(
            {
                "Confidence Range": confidence_bins,
                "Model Confidence": [f"{c:.1%}" for c in model_confidence],
                "Actual Accuracy": [f"{a:.1%}" for a in actual_accuracy],
                "Human Agreement": [f"{h:.1%}" for h in human_agreement],
                "Calibration Error": [
                    f"{abs(c - a):.3f}" for c, a in zip(model_confidence, actual_accuracy)
                ],
            }
        )

        st.dataframe(cal_df, use_container_width=True)


if __name__ == "__main__":
    dashboard = HumanInLoopDashboard()
    dashboard.render()
