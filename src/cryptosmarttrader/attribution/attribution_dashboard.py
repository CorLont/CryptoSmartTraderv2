"""
Attribution Dashboard for CryptoSmartTrader
Real-time PnL attribution visualization and optimization insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from .return_attribution import (
    ReturnAttributionAnalyzer,
    AttributionPeriod,
    ReturnAttributionReport,
)
from ..parity.execution_simulator import ExecutionResult, OrderSide, OrderType


class AttributionDashboard:
    """
    Interactive dashboard for return attribution analysis.

    Features:
    - Real-time PnL breakdown visualization
    - Attribution component analysis
    - Execution optimization insights
    - Cost reduction recommendations
    - Performance benchmarking
    """

    def __init__(self):
        self.analyzer = ReturnAttributionAnalyzer()
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state."""
        if "attribution_data" not in st.session_state:
            st.session_state.attribution_data = []
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.utcnow()

    def render_dashboard(self):
        """Render the main attribution dashboard."""
        st.title("🎯 Return Attribution Dashboard")
        st.markdown("**PnL Decomposition: Alpha • Fees • Slippage • Timing • Sizing**")

        # Sidebar controls
        with st.sidebar:
            st.header("📊 Analysis Controls")

            period = st.selectbox(
                "Attribution Period",
                options=[p.value for p in AttributionPeriod],
                index=1,  # Default to daily
            )

            refresh_data = st.button("🔄 Refresh Data", type="primary")

            if refresh_data:
                self.refresh_attribution_data()

            # Demo data toggle
            use_demo_data = st.checkbox("📈 Use Demo Data", value=True)

            if use_demo_data:
                st.info("Using simulated trading data for demonstration")

        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)

        # Generate or load data
        if use_demo_data:
            report = self.generate_demo_attribution_report(AttributionPeriod(period))
        else:
            report = self.load_latest_attribution_report()

        if report:
            # Key metrics row
            with col1:
                st.metric(
                    "Total Return",
                    f"{report.total_return_bps:.1f} bps",
                    delta=f"{report.excess_return_bps:.1f} vs benchmark",
                )

            with col2:
                st.metric(
                    "Alpha Contribution",
                    f"{report.alpha_component.contribution_bps:.1f} bps",
                    delta=f"{report.alpha_component.contribution_pct:.1f}%",
                )

            with col3:
                st.metric(
                    "Execution Quality",
                    f"{report.execution_quality_score:.1%}",
                    delta="Good" if report.execution_quality_score > 0.8 else "Needs Improvement",
                )

            with col4:
                st.metric(
                    "Attribution Confidence",
                    f"{report.attribution_confidence:.1%}",
                    delta=f"{report.explained_variance_pct:.1f}% explained",
                )

            # Main visualization tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "🏆 Attribution Breakdown",
                    "💰 Cost Analysis",
                    "⚡ Execution Quality",
                    "💡 Optimization",
                ]
            )

            with tab1:
                self.render_attribution_breakdown(report)

            with tab2:
                self.render_cost_analysis(report)

            with tab3:
                self.render_execution_quality(report)

            with tab4:
                self.render_optimization_insights(report)

        else:
            st.warning("⚠️ No attribution data available. Generate some trading data first.")

            if st.button("🎲 Generate Demo Report"):
                demo_report = self.generate_demo_attribution_report(AttributionPeriod.DAILY)
                st.success("Demo attribution report generated!")
                st.rerun()

    def render_attribution_breakdown(self, report: ReturnAttributionReport):
        """Render attribution breakdown visualization."""
        st.subheader("🏆 PnL Attribution Breakdown")

        # Waterfall chart
        components = [
            ("Alpha", report.alpha_component.contribution_bps),
            ("Fees", report.fees_component.contribution_bps),
            ("Slippage", report.slippage_component.contribution_bps),
            ("Timing", report.timing_component.contribution_bps),
            ("Sizing", report.sizing_component.contribution_bps),
            ("Market Impact", report.market_impact_component.contribution_bps),
        ]

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = self.create_waterfall_chart(components, report.total_return_bps)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Component summary table
            st.markdown("**📊 Component Summary**")

            df_components = pd.DataFrame(
                [
                    {
                        "Component": comp[0],
                        "Contribution (bps)": f"{comp[1]:.1f}",
                        "% of Total": f"{comp[1] / report.total_return_bps * 100:.1f}%"
                        if report.total_return_bps != 0
                        else "0.0%",
                    }
                    for comp in components
                ]
            )

            st.dataframe(df_components, use_container_width=True)

        # Attribution confidence indicators
        st.markdown("**🎯 Attribution Confidence**")

        confidence_data = [
            ("Alpha", report.alpha_component.confidence),
            ("Fees", report.fees_component.confidence),
            ("Slippage", report.slippage_component.confidence),
            ("Timing", report.timing_component.confidence),
            ("Sizing", report.sizing_component.confidence),
            ("Market Impact", report.market_impact_component.confidence),
        ]

        fig_confidence = px.bar(
            x=[c[1] for c in confidence_data],
            y=[c[0] for c in confidence_data],
            orientation="h",
            title="Attribution Confidence by Component",
            labels={"x": "Confidence", "y": "Component"},
            color=[c[1] for c in confidence_data],
            color_continuous_scale="RdYlGn",
        )
        fig_confidence.update_layout(height=300)
        st.plotly_chart(fig_confidence, use_container_width=True)

    def render_cost_analysis(self, report: ReturnAttributionReport):
        """Render cost analysis section."""
        st.subheader("💰 Cost Analysis & Optimization")

        # Cost breakdown pie chart
        cost_components = {
            "Fees": abs(report.fees_component.contribution_bps),
            "Slippage": abs(report.slippage_component.contribution_bps),
            "Timing": abs(report.timing_component.contribution_bps),
            "Market Impact": abs(report.market_impact_component.contribution_bps),
        }

        # Filter out zero components
        cost_components = {k: v for k, v in cost_components.items() if v > 0}

        col1, col2 = st.columns([1, 1])

        with col1:
            if cost_components:
                fig_pie = px.pie(
                    values=list(cost_components.values()),
                    names=list(cost_components.keys()),
                    title="Cost Breakdown",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No significant costs detected")

        with col2:
            # Cost metrics
            total_costs = sum(cost_components.values())

            st.metric("Total Execution Costs", f"{total_costs:.1f} bps")

            if total_costs > 0:
                largest_cost = max(cost_components.items(), key=lambda x: x[1])
                st.metric(
                    "Largest Cost Component",
                    largest_cost[0],
                    f"{largest_cost[1]:.1f} bps ({largest_cost[1] / total_costs * 100:.1f}%)",
                )

        # Detailed cost analysis
        st.markdown("**🔍 Detailed Cost Analysis**")

        # Fees breakdown
        if report.fees_component.contribution_bps < -1:
            fees_meta = report.fees_component.metadata

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maker Orders", fees_meta.get("maker_orders", 0))
            with col2:
                st.metric("Taker Orders", fees_meta.get("taker_orders", 0))
            with col3:
                st.metric("Maker Ratio", f"{fees_meta.get('maker_ratio', 0):.1%}")

            if fees_meta.get("maker_ratio", 0) < 0.7:
                st.warning(
                    "⚠️ Low maker ratio detected. Consider using more limit orders to reduce fees."
                )

        # Slippage analysis
        if report.slippage_component.contribution_bps < -1:
            slippage_meta = report.slippage_component.metadata

            st.markdown("**Slippage Analysis**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Slippage", f"{slippage_meta.get('avg_slippage_bps', 0):.1f} bps")
            with col2:
                st.metric("Max Slippage", f"{slippage_meta.get('max_slippage_bps', 0):.1f} bps")
            with col3:
                st.metric("High Slippage Orders", slippage_meta.get("high_slippage_orders", 0))

    def render_execution_quality(self, report: ReturnAttributionReport):
        """Render execution quality analysis."""
        st.subheader("⚡ Execution Quality Analysis")

        # Overall quality score
        quality_score = report.execution_quality_score

        col1, col2 = st.columns([1, 2])

        with col1:
            # Quality gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=quality_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Execution Quality Score"},
                    delta={"reference": 80},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Quality factors
            st.markdown("**📊 Quality Factors**")

            # Extract timing metadata
            timing_meta = report.timing_component.metadata

            quality_factors = pd.DataFrame(
                [
                    {
                        "Factor": "Average Latency",
                        "Value": f"{timing_meta.get('avg_latency_ms', 0):.0f} ms",
                        "Score": f"{timing_meta.get('timing_score', 0):.1%}",
                    },
                    {
                        "Factor": "Slippage Control",
                        "Value": f"{report.slippage_component.metadata.get('avg_slippage_bps', 0):.1f} bps",
                        "Score": f"{max(0, 1 - report.slippage_component.metadata.get('avg_slippage_bps', 0) / 50):.1%}",
                    },
                    {
                        "Factor": "Fee Efficiency",
                        "Value": f"{report.fees_component.metadata.get('maker_ratio', 0):.1%} maker",
                        "Score": f"{report.fees_component.metadata.get('maker_ratio', 0):.1%}",
                    },
                ]
            )

            st.dataframe(quality_factors, use_container_width=True)

        # Execution timeline (if we had timestamp data)
        st.markdown("**⏱️ Timing Analysis**")

        timing_stats = {
            "Fast Executions (<100ms)": timing_meta.get("timing_efficiency", 0),
            "Average Latency": f"{timing_meta.get('avg_latency_ms', 0):.0f} ms",
            "Slow Executions (>1s)": timing_meta.get("slow_executions", 0),
            "Timing Score": timing_meta.get("timing_score", 0),
        }

        for stat, value in timing_stats.items():
            if isinstance(value, float) and 0 <= value <= 1:
                st.metric(stat, f"{value:.1%}")
            else:
                st.metric(stat, str(value))

    def render_optimization_insights(self, report: ReturnAttributionReport):
        """Render optimization insights and recommendations."""
        st.subheader("💡 Optimization Insights")

        # Top opportunities
        st.markdown("**🎯 Top Optimization Opportunities**")

        if report.optimization_opportunities:
            for i, opportunity in enumerate(report.optimization_opportunities, 1):
                st.markdown(f"{i}. {opportunity}")
        else:
            st.success("🎉 No major optimization opportunities detected!")

        # Execution improvements
        st.markdown("**⚡ Execution Improvements**")

        if report.execution_improvements:
            for i, improvement in enumerate(report.execution_improvements, 1):
                st.markdown(f"{i}. {improvement}")
        else:
            st.success("✅ Execution quality is optimal!")

        # Cost reduction suggestions
        st.markdown("**💰 Cost Reduction Suggestions**")

        if report.cost_reduction_suggestions:
            for i, suggestion in enumerate(report.cost_reduction_suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        else:
            st.success("💎 Cost structure is efficient!")

        # Priority matrix
        st.markdown("**📊 Optimization Priority Matrix**")

        # Create priority scores based on impact and ease
        opportunities_data = []

        # Analyze each component for optimization potential
        components = [
            ("Fees", report.fees_component),
            ("Slippage", report.slippage_component),
            ("Timing", report.timing_component),
            ("Market Impact", report.market_impact_component),
        ]

        for comp_name, comp in components:
            if comp.contribution_bps < -2:  # Significant negative impact
                impact = abs(comp.contribution_bps)
                ease = 1.0 - comp.confidence  # Lower confidence = easier to improve

                opportunities_data.append(
                    {
                        "Component": comp_name,
                        "Impact (bps)": impact,
                        "Ease of Improvement": ease,
                        "Priority Score": impact * ease,
                    }
                )

        if opportunities_data:
            df_opportunities = pd.DataFrame(opportunities_data)

            fig_scatter = px.scatter(
                df_opportunities,
                x="Ease of Improvement",
                y="Impact (bps)",
                size="Priority Score",
                text="Component",
                title="Optimization Priority Matrix",
                labels={
                    "Ease of Improvement": "Ease of Improvement →",
                    "Impact (bps)": "Impact (bps) →",
                },
            )
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Action plan
        st.markdown("**📋 Recommended Action Plan**")

        action_plan = self.generate_action_plan(report)
        for i, action in enumerate(action_plan, 1):
            st.markdown(f"**Step {i}:** {action}")

    def create_waterfall_chart(self, components: List[tuple], total_return: float) -> go.Figure:
        """Create waterfall chart for attribution breakdown."""

        # Prepare data for waterfall
        x_labels = ["Starting"] + [comp[0] for comp in components] + ["Total Return"]
        y_values = [0] + [comp[1] for comp in components] + [0]

        # Calculate cumulative values for positioning
        cumulative = [0]
        running_total = 0

        for comp_value in [comp[1] for comp in components]:
            running_total += comp_value
            cumulative.append(running_total)

        cumulative.append(total_return)

        # Create waterfall chart
        fig = go.Figure()

        # Starting bar
        fig.add_trace(go.Bar(x=["Starting"], y=[0], name="Starting", marker_color="blue"))

        # Component bars
        colors = ["green" if val > 0 else "red" for val in [comp[1] for comp in components]]

        for i, (comp_name, comp_value) in enumerate(components):
            fig.add_trace(
                go.Bar(
                    x=[comp_name],
                    y=[comp_value],
                    base=cumulative[i],
                    name=comp_name,
                    marker_color=colors[i],
                    text=f"{comp_value:.1f} bps",
                    textposition="outside",
                )
            )

        # Total bar
        fig.add_trace(
            go.Bar(
                x=["Total Return"],
                y=[total_return],
                name="Total Return",
                marker_color="navy",
                text=f"{total_return:.1f} bps",
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Return Attribution Waterfall",
            yaxis_title="Contribution (bps)",
            showlegend=False,
            height=400,
        )

        return fig

    def generate_demo_attribution_report(
        self, period: AttributionPeriod
    ) -> ReturnAttributionReport:
        """Generate demo attribution report with realistic data."""

        # Generate realistic execution results
        execution_results = []

        for i in range(20):  # 20 demo trades
            exec_result = ExecutionResult(
                order_id=f"demo_{i:03d}",
                symbol="BTC/USD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT if i % 3 == 0 else OrderType.MARKET,
                order_quantity=np.random.uniform(0.05, 0.5),
                executed_quantity=np.random.uniform(0.05, 0.5),
                avg_fill_price=50000 + np.random.normal(0, 1000),
                total_fees=np.random.uniform(5, 25),
                slippage_bps=np.random.uniform(2, 30),
                latency_ms=np.random.uniform(50, 800),
                timestamp=datetime.utcnow() - timedelta(minutes=np.random.randint(0, 1440)),
            )
            execution_results.append(exec_result)

        # Generate portfolio and benchmark returns
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 24))  # Hourly returns
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.018, 24))  # Slightly lower

        # Run attribution analysis
        report = self.analyzer.analyze_attribution(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            execution_results=execution_results,
            period=period,
        )

        return report

    def load_latest_attribution_report(self) -> Optional[ReturnAttributionReport]:
        """Load latest attribution report from storage."""
        reports_dir = Path("data/attribution_reports")

        if not reports_dir.exists():
            return None

        report_files = list(reports_dir.glob("attribution_report_*.json"))

        if not report_files:
            return None

        # Load most recent report
        latest_file = max(report_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, "r") as f:
                report_data = json.load(f)

            # TODO: Implement proper deserialization
            # For now, return None to use demo data
            return None

        except Exception:
            return None

    def refresh_attribution_data(self):
        """Refresh attribution data from latest trading activity."""
        st.session_state.last_refresh = datetime.utcnow()
        st.success("🔄 Attribution data refreshed!")

    def generate_action_plan(self, report: ReturnAttributionReport) -> List[str]:
        """Generate prioritized action plan."""
        actions = []

        # Prioritize by impact
        cost_components = [
            ("Fees", abs(report.fees_component.contribution_bps)),
            ("Slippage", abs(report.slippage_component.contribution_bps)),
            ("Timing", abs(report.timing_component.contribution_bps)),
            ("Market Impact", abs(report.market_impact_component.contribution_bps)),
        ]

        cost_components.sort(key=lambda x: x[1], reverse=True)

        for comp_name, impact in cost_components:
            if impact > 5:  # Significant impact
                if comp_name == "Fees":
                    actions.append("Optimize maker/taker ratio by using more limit orders")
                elif comp_name == "Slippage":
                    actions.append("Improve order routing and reduce order sizes")
                elif comp_name == "Timing":
                    actions.append("Optimize execution latency and order timing")
                elif comp_name == "Market Impact":
                    actions.append("Break up large orders using TWAP or iceberg strategies")

        # Alpha improvement
        if report.alpha_component.contribution_bps < 10:
            actions.append("Review and enhance alpha generation models")

        if not actions:
            actions.append("Continue current execution strategy - performance is optimal")

        return actions


# Streamlit app entry point
def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Return Attribution Dashboard", page_icon="🎯", layout="wide")

    dashboard = AttributionDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
