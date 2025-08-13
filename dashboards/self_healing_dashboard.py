#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Self-Healing Dashboard
Interactive dashboard for monitoring and controlling the self-healing system
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

from core.self_healing_system import (
    get_self_healing_system,
    SystemComponent,
    AlertLevel,
    DisableReason,
    report_performance,
    report_black_swan,
    report_data_gap,
    report_security_alert,
    disable_component,
    enable_component,
    get_system_health,
)


class SelfHealingDashboard:
    """Interactive dashboard for self-healing system monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize self-healing system
        self.healing_system = get_self_healing_system()

        # Page config
        st.set_page_config(
            page_title="CryptoSmartTrader - Self-Healing",
            page_icon="🔧",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render(self):
        """Render the main dashboard"""
        st.title("🔧 Self-Healing & Auto-Disabling System")
        st.markdown("Autonomous system protection against performance degradation and anomalies")

        # Sidebar controls
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 System Overview",
                "🚨 Alerts & Incidents",
                "📈 Performance Monitoring",
                "🔧 Component Control",
                "⚙️ Configuration",
            ]
        )

        with tab1:
            self._render_system_overview_tab()

        with tab2:
            self._render_alerts_tab()

        with tab3:
            self._render_performance_tab()

        with tab4:
            self._render_component_control_tab()

        with tab5:
            self._render_configuration_tab()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🔧 Self-Healing Controls")

        # System status
        status = self.healing_system.get_system_status()

        monitoring_status = "🟢 Active" if status.get("monitoring_active") else "🔴 Inactive"
        st.sidebar.metric("Monitoring Status", monitoring_status)

        # Quick stats
        st.sidebar.metric("Total Components", status.get("total_components", 0))
        st.sidebar.metric("Enabled", len(status.get("enabled_components", [])))
        st.sidebar.metric("Disabled", len(status.get("disabled_components", [])))
        st.sidebar.metric("Total Alerts", status.get("total_alerts", 0))

        st.sidebar.markdown("---")

        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")

        if st.sidebar.button("▶️ Start Monitoring", use_container_width=True):
            self.healing_system.start_monitoring()
            st.rerun()

        if st.sidebar.button("⏹️ Stop Monitoring", use_container_width=True):
            self.healing_system.stop_monitoring()
            st.rerun()

        if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

        # Emergency actions
        st.sidebar.subheader("🚨 Emergency Actions")

        if st.sidebar.button("🛑 Emergency Stop All", use_container_width=True):
            for component in SystemComponent:
                disable_component(component, DisableReason.MANUAL_OVERRIDE)
            st.sidebar.error("Emergency stop activated!")
            st.rerun()

        if st.sidebar.button("🟢 Enable All Components", use_container_width=True):
            for component in SystemComponent:
                enable_component(component)
            st.sidebar.success("All components enabled!")
            st.rerun()

        # Test scenarios
        st.sidebar.subheader("🧪 Test Scenarios")

        if st.sidebar.button("📉 Simulate Performance Drop", use_container_width=True):
            report_performance(SystemComponent.TRADING_ENGINE, "accuracy", 0.3, 0.6)
            st.sidebar.warning("Performance drop simulated!")

        if st.sidebar.button("🌪️ Simulate Black Swan", use_container_width=True):
            report_black_swan("market_crash", 0.9, "Simulated market crash event")
            st.sidebar.error("Black swan event simulated!")

        if st.sidebar.button("🔒 Simulate Security Alert", use_container_width=True):
            report_security_alert("unusual_activity", AlertLevel.HIGH, "Simulated security breach")
            st.sidebar.error("Security alert simulated!")

    def _render_system_overview_tab(self):
        """Render system overview tab"""
        st.header("📊 System Overview")

        # Get current status
        status = self.healing_system.get_system_status()

        # Overall health metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            monitoring_status = "Active" if status.get("monitoring_active") else "Inactive"
            delta = "🟢" if status.get("monitoring_active") else "🔴"
            st.metric("Monitoring", monitoring_status, delta=delta)

        with col2:
            enabled_count = len(status.get("enabled_components", []))
            total_count = status.get("total_components", 0)
            health_pct = (enabled_count / total_count * 100) if total_count > 0 else 0
            st.metric("System Health", f"{health_pct:.1f}%", delta=f"{enabled_count}/{total_count}")

        with col3:
            disabled_count = len(status.get("disabled_components", []))
            st.metric(
                "Disabled Components", disabled_count, delta="⚠️" if disabled_count > 0 else "✅"
            )

        with col4:
            recent_alerts = len(
                [
                    a
                    for a in status.get("recent_alerts", [])
                    if a.get("level") in ["high", "critical"]
                ]
            )
            st.metric("Critical Alerts", recent_alerts, delta="🚨" if recent_alerts > 0 else "✅")

        # Component status overview
        st.subheader("🔧 Component Status Matrix")

        performance_summary = status.get("performance_summary", {})

        if performance_summary:
            # Create status matrix
            components_data = []

            for comp_name, comp_data in performance_summary.items():
                status_icon = "🟢" if comp_data["enabled"] else "🔴"
                performance_score = comp_data.get("performance_score", 0) * 100
                failures = comp_data.get("consecutive_failures", 0)

                components_data.append(
                    {
                        "Component": comp_name.replace("_", " ").title(),
                        "Status": status_icon,
                        "Performance": f"{performance_score:.1f}%",
                        "Failures": failures,
                        "Last Disabled": comp_data.get("last_disabled", "Never")[:19]
                        if comp_data.get("last_disabled")
                        else "Never",
                        "Reason": comp_data.get("disable_reason", "").replace("_", " ").title()
                        if comp_data.get("disable_reason")
                        else "N/A",
                    }
                )

            components_df = pd.DataFrame(components_data)
            st.dataframe(components_df, use_container_width=True)

        # System health trend
        st.subheader("📈 System Health Trend")
        self._plot_system_health_trend()

        # Black swan indicators
        black_swan_count = status.get("black_swan_indicators", 0)
        if black_swan_count > 0:
            st.subheader("🌪️ Black Swan Risk Assessment")

            risk_level = (
                "HIGH" if black_swan_count > 3 else "MEDIUM" if black_swan_count > 1 else "LOW"
            )
            risk_color = (
                "red" if risk_level == "HIGH" else "orange" if risk_level == "MEDIUM" else "green"
            )

            st.markdown(
                f"**Current Risk Level:** :{risk_color}[{risk_level}] ({black_swan_count} indicators)"
            )

            if black_swan_count > 2:
                st.warning(
                    "⚠️ Multiple black swan indicators detected. System is in protective mode."
                )

    def _render_alerts_tab(self):
        """Render alerts and incidents tab"""
        st.header("🚨 Alerts & Incidents")

        # Get recent alerts
        status = self.healing_system.get_system_status()
        recent_alerts = status.get("recent_alerts", [])

        if not recent_alerts:
            st.success("✅ No recent alerts - system running smoothly!")
            return

        # Alert level filter
        col1, col2 = st.columns([3, 1])

        with col2:
            alert_level_filter = st.selectbox(
                "Filter by Level", options=["All", "Critical", "High", "Medium", "Low"], index=0
            )

        # Filter alerts
        if alert_level_filter != "All":
            filtered_alerts = [
                a for a in recent_alerts if a.get("level", "").lower() == alert_level_filter.lower()
            ]
        else:
            filtered_alerts = recent_alerts

        # Display alerts
        st.subheader(f"📋 Recent Alerts ({len(filtered_alerts)})")

        for alert in reversed(filtered_alerts[-20:]):  # Show latest 20
            level = alert.get("level", "low")
            level_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

            level_icon = level_icons.get(level, "🔵")
            component = alert.get("component", "unknown")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "")[:19]
            auto_disabled = alert.get("auto_disabled", False)

            with st.expander(f"{level_icon} {level.upper()} - {component} ({timestamp})"):
                st.write(f"**Message:** {message}")
                st.write(f"**Component:** {component}")
                st.write(f"**Level:** {level.upper()}")
                st.write(f"**Auto-disabled:** {'Yes' if auto_disabled else 'No'}")
                st.write(f"**Reason:** {alert.get('reason', 'unknown').replace('_', ' ').title()}")

                if alert.get("recovery_time"):
                    st.write(f"**Recovery Time:** {alert['recovery_time'][:19]}")

        # Alert statistics
        st.subheader("📊 Alert Statistics")

        if recent_alerts:
            # Alert level distribution
            level_counts = {}
            for alert in recent_alerts:
                level = alert.get("level", "unknown")
                level_counts[level] = level_counts.get(level, 0) + 1

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(level_counts.keys()),
                        values=list(level_counts.values()),
                        hole=0.3,
                    )
                ]
            )

            fig.update_layout(title="Alert Level Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _render_performance_tab(self):
        """Render performance monitoring tab"""
        st.header("📈 Performance Monitoring")

        # Component selection
        selected_component = st.selectbox(
            "Select Component to Monitor",
            options=[comp.value for comp in SystemComponent],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        component_enum = SystemComponent(selected_component)

        # Get component health
        health_data = self.healing_system.get_component_health(component_enum)

        if "error" in health_data:
            st.error(f"Error loading component data: {health_data['error']}")
            return

        # Component status
        col1, col2, col3, col4 = st.columns(4)

        component_status = health_data.get("status", {})

        with col1:
            enabled = component_status.get("is_enabled", False)
            status_text = "Enabled" if enabled else "Disabled"
            status_delta = "🟢" if enabled else "🔴"
            st.metric("Status", status_text, delta=status_delta)

        with col2:
            performance_score = component_status.get("performance_score", 0) * 100
            st.metric("Performance Score", f"{performance_score:.1f}%")

        with col3:
            failures = component_status.get("consecutive_failures", 0)
            st.metric("Consecutive Failures", failures)

        with col4:
            data_quality = health_data.get("data_quality_score", 1.0) * 100
            st.metric("Data Quality", f"{data_quality:.1f}%")

        # Performance metrics history
        st.subheader("📊 Performance Metrics History")

        recent_metrics = health_data.get("recent_metrics", [])

        if recent_metrics:
            # Convert to DataFrame
            metrics_df = pd.DataFrame(recent_metrics)

            if not metrics_df.empty:
                # Plot performance over time
                fig = go.Figure()

                # Group by metric name
                for metric_name in metrics_df["metric_name"].unique():
                    metric_data = metrics_df[metrics_df["metric_name"] == metric_name]

                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(metric_data["timestamp"]),
                            y=metric_data["value"],
                            mode="lines+markers",
                            name=metric_name.replace("_", " ").title(),
                            line=dict(width=2),
                        )
                    )

                    # Add threshold line
                    if len(metric_data) > 0:
                        threshold = metric_data["threshold"].iloc[0]
                        fig.add_hline(
                            y=threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"{metric_name} threshold",
                        )

                fig.update_layout(
                    title=f"Performance Metrics - {selected_component.replace('_', ' ').title()}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show recent metrics table
                st.subheader("📋 Recent Metrics")

                display_df = metrics_df.copy()
                display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                display_df["is_anomaly"] = display_df["is_anomaly"].map({True: "🔴", False: "🟢"})

                st.dataframe(
                    display_df[
                        ["timestamp", "metric_name", "value", "threshold", "is_anomaly"]
                    ].rename(
                        columns={
                            "timestamp": "Time",
                            "metric_name": "Metric",
                            "value": "Value",
                            "threshold": "Threshold",
                            "is_anomaly": "Status",
                        }
                    ),
                    use_container_width=True,
                )

        else:
            st.info("No performance metrics available for this component.")

        # Manual performance reporting
        st.subheader("📝 Manual Performance Report")

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_name = st.text_input("Metric Name", value="accuracy")

        with col2:
            metric_value = st.number_input(
                "Value", min_value=0.0, max_value=1.0, value=0.8, step=0.01
            )

        with col3:
            metric_threshold = st.number_input(
                "Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01
            )

        if st.button("📊 Report Metric", use_container_width=True):
            report_performance(component_enum, metric_name, metric_value, metric_threshold)
            st.success(f"Performance metric reported: {metric_name}={metric_value:.3f}")
            st.rerun()

    def _render_component_control_tab(self):
        """Render component control tab"""
        st.header("🔧 Component Control")

        # Get current status
        status = self.healing_system.get_system_status()
        performance_summary = status.get("performance_summary", {})

        # Component grid
        st.subheader("🎛️ Component Controls")

        # Create control grid
        cols = st.columns(3)

        for idx, (comp_name, comp_data) in enumerate(performance_summary.items()):
            with cols[idx % 3]:
                component_enum = SystemComponent(comp_name)

                enabled = comp_data["enabled"]
                performance_score = comp_data.get("performance_score", 0) * 100
                failures = comp_data.get("consecutive_failures", 0)

                # Component card
                with st.container():
                    st.markdown(f"**{comp_name.replace('_', ' ').title()}**")

                    status_color = "green" if enabled else "red"
                    status_text = "🟢 Enabled" if enabled else "🔴 Disabled"
                    st.markdown(f"Status: :{status_color}[{status_text}]")

                    st.metric(
                        "Performance", f"{performance_score:.1f}%", delta=f"{failures} failures"
                    )

                    # Control buttons
                    button_col1, button_col2 = st.columns(2)

                    with button_col1:
                        if st.button(
                            f"{'Disable' if enabled else 'Enable'}", key=f"toggle_{comp_name}"
                        ):
                            if enabled:
                                disable_component(component_enum, DisableReason.MANUAL_OVERRIDE)
                                st.success(f"Disabled {comp_name}")
                            else:
                                enable_component(component_enum)
                                st.success(f"Enabled {comp_name}")
                            st.rerun()

                    with button_col2:
                        if st.button("Details", key=f"details_{comp_name}"):
                            st.session_state[f"show_details_{comp_name}"] = True

                    # Show details if requested
                    if st.session_state.get(f"show_details_{comp_name}", False):
                        with st.expander(f"Details - {comp_name}", expanded=True):
                            health_data = self.healing_system.get_component_health(component_enum)
                            st.json(health_data)

                            if st.button("Close", key=f"close_{comp_name}"):
                                st.session_state[f"show_details_{comp_name}"] = False
                                st.rerun()

                    st.markdown("---")

        # Bulk operations
        st.subheader("🔄 Bulk Operations")

        bulk_col1, bulk_col2, bulk_col3 = st.columns(3)

        with bulk_col1:
            if st.button("🟢 Enable All", use_container_width=True):
                for component in SystemComponent:
                    enable_component(component)
                st.success("All components enabled!")
                st.rerun()

        with bulk_col2:
            if st.button("🔴 Disable All", use_container_width=True):
                for component in SystemComponent:
                    disable_component(component, DisableReason.MANUAL_OVERRIDE)
                st.warning("All components disabled!")
                st.rerun()

        with bulk_col3:
            if st.button("🔄 Reset All", use_container_width=True):
                # Reset all components (enable and clear failures)
                for component in SystemComponent:
                    enable_component(component)
                st.info("All components reset!")
                st.rerun()

        # Manual incident simulation
        st.subheader("🧪 Incident Simulation")

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            st.write("**Performance Issues**")

            sim_component = st.selectbox(
                "Component", options=[comp.value for comp in SystemComponent], key="sim_component"
            )

            if st.button("📉 Simulate Performance Drop", use_container_width=True):
                component_enum = SystemComponent(sim_component)
                report_performance(component_enum, "test_metric", 0.2, 0.6)
                st.warning(f"Performance drop simulated for {sim_component}")

            if st.button("📊 Simulate Data Gap", use_container_width=True):
                component_enum = SystemComponent(sim_component)
                report_data_gap(component_enum, 600, "Simulated 10-minute data gap")
                st.warning(f"Data gap simulated for {sim_component}")

        with sim_col2:
            st.write("**System-wide Issues**")

            if st.button("🌪️ Simulate Black Swan", use_container_width=True):
                report_black_swan("market_volatility", 0.95, "Simulated extreme market volatility")
                st.error("Black swan event simulated!")

            if st.button("🔒 Simulate Security Alert", use_container_width=True):
                report_security_alert(
                    "api_anomaly", AlertLevel.CRITICAL, "Simulated API security breach"
                )
                st.error("Security alert simulated!")

    def _render_configuration_tab(self):
        """Render configuration tab"""
        st.header("⚙️ Self-Healing Configuration")

        # Current configuration display
        config = self.healing_system.config

        st.subheader("📋 Current Configuration")

        # Performance thresholds
        with st.expander("🎯 Performance Thresholds", expanded=True):
            perf_thresholds = config.get("performance_thresholds", {})

            for component, thresholds in perf_thresholds.items():
                st.write(f"**{component.replace('_', ' ').title()}:**")

                threshold_cols = st.columns(len(thresholds))
                for idx, (metric, value) in enumerate(thresholds.items()):
                    with threshold_cols[idx % len(threshold_cols)]:
                        st.metric(metric.replace("_", " ").title(), f"{value}")

        # Auto-disable settings
        with st.expander("🔧 Auto-Disable Settings"):
            auto_settings = config.get("auto_disable_settings", {})

            settings_cols = st.columns(2)
            with settings_cols[0]:
                st.metric(
                    "Failure Threshold", auto_settings.get("consecutive_failure_threshold", 3)
                )
                st.metric(
                    "Performance Degradation",
                    f"{auto_settings.get('performance_degradation_threshold', 0.3):.1%}",
                )

            with settings_cols[1]:
                st.metric("Recovery Attempts", auto_settings.get("recovery_attempt_limit", 5))
                st.metric("Recovery Delay", f"{auto_settings.get('auto_recovery_delay', 300)}s")

        # Black swan thresholds
        with st.expander("🌪️ Black Swan Detection"):
            black_swan = config.get("black_swan_thresholds", {})

            bs_cols = st.columns(2)
            with bs_cols[0]:
                st.metric("Market Volatility", f"{black_swan.get('market_volatility', 0.05):.1%}")
                st.metric("Price Deviation", f"{black_swan.get('price_deviation', 0.15):.1%}")

            with bs_cols[1]:
                st.metric("Volume Spike", f"{black_swan.get('volume_spike', 10.0)}x")
                st.metric(
                    "Correlation Breakdown", f"{black_swan.get('correlation_breakdown', 0.3):.1f}"
                )

        # Configuration editing
        st.subheader("✏️ Edit Configuration")

        st.info(
            "Configuration editing will be available in the full system. Current values are displayed above for monitoring."
        )

        # System control
        st.subheader("🎛️ System Control")

        control_col1, control_col2 = st.columns(2)

        with control_col1:
            if st.button("🔄 Restart Monitoring", use_container_width=True):
                self.healing_system.stop_monitoring()
                self.healing_system.start_monitoring()
                st.success("Monitoring system restarted!")

        with control_col2:
            if st.button("💾 Save Configuration", use_container_width=True):
                st.info("Configuration saving will be implemented in full system")

    def _plot_system_health_trend(self):
        """Plot system health trend over time"""
        try:
            # Generate mock health trend data for demonstration
            dates = pd.date_range(
                start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq="H"
            )

            # Simulate health percentage (enabled components / total)
            np.random.seed(42)
            health_trend = 100 - np.random.exponential(
                2, len(dates)
            )  # Mostly high, occasional drops
            health_trend = np.clip(health_trend, 0, 100)

            # Add some realistic drops
            health_trend[10:15] = np.linspace(health_trend[9], 60, 5)  # Simulate incident
            health_trend[15:20] = np.linspace(60, health_trend[20], 5)  # Recovery

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=health_trend,
                    mode="lines",
                    name="System Health",
                    line=dict(color="blue", width=2),
                    fill="tonexty",
                    fillcolor="rgba(0,100,200,0.1)",
                )
            )

            # Add threshold lines
            fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Healthy")
            fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical")

            fig.update_layout(
                title="System Health Trend (24 Hours)",
                xaxis_title="Time",
                yaxis_title="Health Percentage (%)",
                height=400,
                yaxis=dict(range=[0, 105]),
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting health trend: {e}")


def main():
    """Main dashboard entry point"""
    try:
        dashboard = SelfHealingDashboard()
        dashboard.render()

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
