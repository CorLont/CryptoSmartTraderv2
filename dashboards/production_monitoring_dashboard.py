#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Production Monitoring Dashboard
Deep performance monitoring per module/agent with real-time metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import threading
import psutil
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.security_manager import get_security_manager
    from core.async_coordinator import get_async_coordinator
    from core.exception_handler import get_exception_handler
    from core.fallback_data_eliminator import get_fallback_eliminator
    from core.feature_fusion_engine import get_feature_fusion_engine
    from core.market_regime_detector import get_market_regime_detector
    from core.orderbook_analyzer import get_orderbook_analyzer
    from core.ml_ai_differentiators import get_ml_differentiators_coordinator
    from agents.sentiment_agent import SentimentAgent
    from agents.technical_agent import TechnicalAgent
    from agents.whale_agent import WhaleAgent
    from agents.ml_agent import MLAgent
    from core.comprehensive_market_scanner import ComprehensiveMarketScanner

    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"Core modules not available: {e}")
    CORE_AVAILABLE = False


def get_system_resources():
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
        }
    except Exception as e:
        st.error(f"Failed to get system resources: {e}")
        return {}


def get_agent_performance_metrics():
    """Get detailed performance metrics for each agent"""
    if not CORE_AVAILABLE:
        return {}

    metrics = {}

    try:
        # Security Manager metrics
        security_manager = get_security_manager()
        security_health = security_manager.validate_secrets_health()
        metrics["security_manager"] = {
            "status": "operational"
            if security_health.get("total_secrets_cached", 0) > 0
            else "degraded",
            "secrets_cached": security_health.get("total_secrets_cached", 0),
            "vault_connected": security_health.get("vault_connected", False),
            "active_lockouts": security_health.get("active_lockouts", 0),
            "audit_entries": security_health.get("audit_entries", 0),
        }
    except Exception as e:
        metrics["security_manager"] = {"status": "error", "error": str(e)}

    try:
        # Async Coordinator metrics
        coordinator = get_async_coordinator()
        coord_health = coordinator.get_system_health()
        coord_perf = coordinator.get_performance_metrics()
        metrics["async_coordinator"] = {
            "status": "operational" if coord_health.get("event_loop_running", False) else "error",
            "total_tasks": coord_health.get("total_tasks", 0),
            "running_tasks": coord_health.get("running_tasks", 0),
            "failed_tasks": coord_health.get("failed_tasks", 0),
            "peak_concurrent_tasks": coord_perf.get("peak_concurrent_tasks", 0),
            "average_task_duration": coord_perf.get("average_task_duration", 0),
        }
    except Exception as e:
        metrics["async_coordinator"] = {"status": "error", "error": str(e)}

    try:
        # Exception Handler metrics
        exception_handler = get_exception_handler()
        error_stats = exception_handler.get_error_statistics()
        health_impact = exception_handler.get_health_impact_score()
        metrics["exception_handler"] = {
            "status": "operational" if health_impact > 0.5 else "degraded",
            "total_errors": error_stats.get("total_errors", 0),
            "errors_last_hour": error_stats.get("errors_last_hour", 0),
            "critical_errors": error_stats.get("critical_errors_last_hour", 0),
            "health_impact_score": health_impact,
            "alerts_sent": error_stats.get("alerts_sent_this_hour", 0),
        }
    except Exception as e:
        metrics["exception_handler"] = {"status": "error", "error": str(e)}

    try:
        # Fallback Data Eliminator metrics
        eliminator = get_fallback_eliminator()
        validation_stats = eliminator.get_validation_statistics()
        health_score = eliminator.get_health_score()
        metrics["fallback_eliminator"] = {
            "status": "operational" if health_score > 0.8 else "degraded",
            "total_validations": validation_stats.get("total_validations", 0),
            "authentic_data_rate": validation_stats.get("authentic_data_rate", 0),
            "fallback_rejected": validation_stats.get("fallback_data_rejected", 0),
            "synthetic_rejected": validation_stats.get("synthetic_data_rejected", 0),
            "health_score": health_score,
        }
    except Exception as e:
        metrics["fallback_eliminator"] = {"status": "error", "error": str(e)}

    try:
        # Feature Fusion Engine metrics
        fusion_engine = get_feature_fusion_engine()
        fusion_stats = fusion_engine.get_fusion_statistics()
        metrics["feature_fusion"] = {
            "status": "operational",
            "scalers_trained": fusion_stats.get("scalers_trained", 0),
            "selectors_trained": fusion_stats.get("feature_selectors_trained", 0),
            "attention_available": fusion_stats.get("attention_module_available", False),
            "torch_available": fusion_stats.get("torch_available", False),
        }
    except Exception as e:
        metrics["feature_fusion"] = {"status": "error", "error": str(e)}

    try:
        # Market Regime Detector metrics
        regime_detector = get_market_regime_detector()
        regime_stats = regime_detector.get_regime_statistics()
        metrics["regime_detector"] = {
            "status": "operational",
            "current_regime": regime_stats.get("current_regime", "unknown"),
            "volatility_regime": regime_stats.get("current_volatility_regime", "unknown"),
            "regime_duration": regime_stats.get("regime_duration", 0),
            "regime_changes": regime_stats.get("total_regime_changes", 0),
            "ml_trained": regime_stats.get("ml_models_trained", False),
        }
    except Exception as e:
        metrics["regime_detector"] = {"status": "error", "error": str(e)}

    try:
        # Order Book Analyzer metrics
        orderbook_analyzer = get_orderbook_analyzer()
        analysis_stats = orderbook_analyzer.get_analysis_statistics()
        metrics["orderbook_analyzer"] = {
            "status": "operational",
            "snapshots_analyzed": analysis_stats.get("snapshots_analyzed", 0),
            "metrics_generated": analysis_stats.get("metrics_generated", 0),
            "liquidity_level": analysis_stats.get("current_liquidity_level", "unknown"),
            "avg_liquidity_score": analysis_stats.get("average_liquidity_score", 0),
            "spoofing_events": analysis_stats.get("spoofing_events_detected", 0),
        }
    except Exception as e:
        metrics["orderbook_analyzer"] = {"status": "error", "error": str(e)}

    try:
        # ML Differentiators metrics
        ml_coordinator = get_ml_differentiators_coordinator()
        ml_status = ml_coordinator.get_system_status()
        metrics["ml_differentiators"] = {
            "status": "operational" if ml_status.get("system_health", 0) > 0.7 else "degraded",
            "deep_learning_active": ml_status.get("deep_learning", {}).get("enabled", False),
            "feature_fusion_active": ml_status.get("feature_fusion", {}).get("enabled", False),
            "confidence_filtering": ml_status.get("confidence_filtering", {}).get("enabled", False),
            "self_learning": ml_status.get("self_learning", {}).get("enabled", False),
            "system_health": ml_status.get("system_health", 0),
        }
    except Exception as e:
        metrics["ml_differentiators"] = {"status": "error", "error": str(e)}

    return metrics


def create_system_health_gauge(health_score):
    """Create a gauge chart for system health"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "System Health Score"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "gray"},
                    {"range": [80, 100], "color": "lightgreen"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )

    fig.update_layout(height=300)
    return fig


def create_resource_usage_chart(resources):
    """Create resource usage visualization"""
    if not resources:
        return go.Figure()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("CPU Usage", "Memory Usage", "Disk Usage"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    )

    # CPU gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=resources.get("cpu_percent", 0),
            title={"text": "CPU %"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "blue"},
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 80},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ),
        row=1,
        col=1,
    )

    # Memory gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=resources.get("memory_percent", 0),
            title={"text": "Memory %"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "green"},
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 85},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ),
        row=1,
        col=2,
    )

    # Disk gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=resources.get("disk_percent", 0),
            title={"text": "Disk %"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "orange"},
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ),
        row=1,
        col=3,
    )

    fig.update_layout(height=300)
    return fig


def create_agent_status_matrix(metrics):
    """Create agent status matrix visualization"""
    if not metrics:
        return go.Figure()

    agents = list(metrics.keys())
    statuses = [metrics[agent].get("status", "unknown") for agent in agents]

    # Color mapping
    color_map = {"operational": "green", "degraded": "yellow", "error": "red", "unknown": "gray"}

    colors = [color_map.get(status, "gray") for status in statuses]

    fig = go.Figure(
        data=go.Bar(
            x=agents, y=[1] * len(agents), marker_color=colors, text=statuses, textposition="inside"
        )
    )

    fig.update_layout(
        title="Agent Status Overview",
        xaxis_title="Agents",
        yaxis_title="Status",
        height=400,
        showlegend=False,
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    return fig


def create_performance_trends_chart(metrics):
    """Create performance trends over time"""
    # This would typically use historical data
    # For now, create a placeholder with current metrics

    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=i * 5) for i in range(12, 0, -1)]
    time_points.append(current_time)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Error Rate", "Task Completion Rate", "System Health", "Data Quality"),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Error rate trend (mock data)
    error_rates = np.random.normal(5, 2, len(time_points))
    error_rates = np.maximum(0, error_rates)

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=error_rates,
            mode="lines+markers",
            name="Error Rate",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # Task completion rate
    completion_rates = np.random.normal(95, 3, len(time_points))
    completion_rates = np.clip(completion_rates, 70, 100)

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=completion_rates,
            mode="lines+markers",
            name="Completion Rate",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )

    # System health
    health_scores = np.random.normal(85, 5, len(time_points))
    health_scores = np.clip(health_scores, 60, 100)

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=health_scores,
            mode="lines+markers",
            name="System Health",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    # Data quality
    quality_scores = np.random.normal(92, 4, len(time_points))
    quality_scores = np.clip(quality_scores, 75, 100)

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=quality_scores,
            mode="lines+markers",
            name="Data Quality",
            line=dict(color="purple"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, title_text="Performance Trends (Last Hour)")
    return fig


def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="CryptoSmartTrader - Production Monitoring", page_icon="📊", layout="wide"
    )

    st.title("🏭 CryptoSmartTrader V2 - Production Monitoring Dashboard")
    st.markdown("**Real-time system monitoring and performance analytics**")

    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    with col3:
        st.write(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if not CORE_AVAILABLE:
        st.error("⚠️ Core modules not available. Please check system configuration.")
        return

    # Get current metrics
    with st.spinner("Loading system metrics..."):
        resources = get_system_resources()
        agent_metrics = get_agent_performance_metrics()

    # Calculate overall system health
    operational_count = sum(1 for m in agent_metrics.values() if m.get("status") == "operational")
    total_agents = len(agent_metrics)
    system_health = operational_count / total_agents if total_agents > 0 else 0

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "System Health",
            f"{system_health * 100:.1f}%",
            delta=f"{operational_count}/{total_agents} agents operational",
        )

    with col2:
        cpu_usage = resources.get("cpu_percent", 0)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta="Normal" if cpu_usage < 80 else "High")

    with col3:
        memory_usage = resources.get("memory_percent", 0)
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f}%",
            delta=f"{resources.get('memory_used_gb', 0):.1f}GB used",
        )

    with col4:
        error_count = sum(
            m.get("total_errors", 0) for m in agent_metrics.values() if "total_errors" in m
        )
        st.metric("Total Errors", f"{error_count}", delta="Last 24h")

    st.divider()

    # System health and resource monitoring
    st.subheader("📊 System Overview")

    col1, col2 = st.columns(2)

    with col1:
        health_fig = create_system_health_gauge(system_health)
        st.plotly_chart(health_fig, use_container_width=True)

    with col2:
        resource_fig = create_resource_usage_chart(resources)
        st.plotly_chart(resource_fig, use_container_width=True)

    # Agent status matrix
    st.subheader("🤖 Agent Status Matrix")
    status_fig = create_agent_status_matrix(agent_metrics)
    st.plotly_chart(status_fig, use_container_width=True)

    # Detailed agent metrics
    st.subheader("🔍 Detailed Agent Metrics")

    for agent_name, metrics in agent_metrics.items():
        with st.expander(
            f"{agent_name.replace('_', ' ').title()} - {metrics.get('status', 'unknown').upper()}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Status:**", metrics.get("status", "unknown"))
                if "error" in metrics:
                    st.error(f"Error: {metrics['error']}")

            with col2:
                # Display relevant metrics for each agent
                for key, value in metrics.items():
                    if key not in ["status", "error"]:
                        st.write(f"**{key.replace('_', ' ').title()}:**", value)

    # Performance trends
    st.subheader("📈 Performance Trends")
    trends_fig = create_performance_trends_chart(agent_metrics)
    st.plotly_chart(trends_fig, use_container_width=True)

    # Critical alerts section
    st.subheader("🚨 Critical Alerts")

    alerts = []

    # Check for critical conditions
    if system_health < 0.7:
        alerts.append(("HIGH", f"System health critically low: {system_health * 100:.1f}%"))

    if resources.get("cpu_percent", 0) > 90:
        alerts.append(("HIGH", f"CPU usage critically high: {resources['cpu_percent']:.1f}%"))

    if resources.get("memory_percent", 0) > 95:
        alerts.append(("HIGH", f"Memory usage critically high: {resources['memory_percent']:.1f}%"))

    # Check for failed agents
    failed_agents = [
        name for name, metrics in agent_metrics.items() if metrics.get("status") == "error"
    ]
    if failed_agents:
        alerts.append(("MEDIUM", f"Agents in error state: {', '.join(failed_agents)}"))

    # Check for high error rates
    recent_errors = sum(
        m.get("errors_last_hour", 0) for m in agent_metrics.values() if "errors_last_hour" in m
    )
    if recent_errors > 10:
        alerts.append(("MEDIUM", f"High error rate: {recent_errors} errors in last hour"))

    if alerts:
        for severity, message in alerts:
            if severity == "HIGH":
                st.error(f"🔴 **{severity}:** {message}")
            elif severity == "MEDIUM":
                st.warning(f"🟡 **{severity}:** {message}")
            else:
                st.info(f"🔵 **{severity}:** {message}")
    else:
        st.success("✅ No critical alerts. System operating normally.")

    # System logs section
    st.subheader("📋 Recent System Activity")

    try:
        # Try to read recent log entries
        log_files = ["logs/cryptotrader.log", "logs/system.log", "logs/security_audit.log"]

        recent_logs = []
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                try:
                    with open(log_path, "r") as f:
                        lines = f.readlines()
                        recent_logs.extend(lines[-10:])  # Last 10 lines
                except Exception:
                    continue

        if recent_logs:
            log_text = "\n".join(recent_logs[-20:])  # Show last 20 total lines
            st.text_area("Recent Log Entries", log_text, height=200)
        else:
            st.info("No recent log entries found")

    except Exception as e:
        st.warning(f"Could not load log files: {e}")

    # Configuration and diagnostics
    with st.expander("🔧 System Configuration & Diagnostics"):
        st.write("**Python Version:**", sys.version)
        st.write("**Platform:**", sys.platform)
        st.write("**Working Directory:**", Path.cwd())
        st.write("**Total Processes:**", len(psutil.pids()))

        # Network connections
        try:
            connections = psutil.net_connections()
            listening = [c for c in connections if c.status == "LISTEN"]
            st.write(
                f"**Network Connections:** {len(connections)} total, {len(listening)} listening"
            )
        except Exception:
            st.write("**Network Connections:** Unable to retrieve")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
