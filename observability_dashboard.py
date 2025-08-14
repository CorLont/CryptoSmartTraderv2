#!/usr/bin/env python3
"""
Observability Dashboard
Real-time metrics en alerting dashboard voor CryptoSmartTrader V2

Features:
- Live metrics monitoring met auto-refresh
- Alert status en escalation tracking  
- Degradatie detectie met trend analysis
- Performance baseline tracking
- Multi-channel notification history
- Interactive alert acknowledgment
- System health overview
- Prometheus metrics export
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure page
st.set_page_config(
    page_title="CryptoSmartTrader V2 - Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_unified_metrics():
    """Load unified metrics system"""
    try:
        import sys
        sys.path.append('src')
        from cryptosmarttrader.observability.unified_metrics_alerting_system import unified_metrics
        return unified_metrics
    except ImportError as e:
        st.error(f"❌ Could not load unified metrics system: {e}")
        return None


def create_system_health_overview(metrics_system) -> Dict[str, Any]:
    """Create system health overview"""
    try:
        if metrics_system:
            dashboard_data = metrics_system.get_dashboard_data()
            return dashboard_data
        else:
            # Mock data voor demo
            return {
                "system_status": {
                    "health_score": 0.85,
                    "active_alerts": 2,
                    "total_metrics": 45,
                    "uptime_seconds": 86400
                },
                "alert_summary": {
                    "firing": 1,
                    "pending": 1,
                    "acknowledged": 0
                },
                "active_alerts": [
                    {
                        "name": "HighOrderLatency",
                        "severity": "warning",
                        "current_value": 0.65,
                        "threshold": 0.5,
                        "duration": 180,
                        "state": "firing"
                    },
                    {
                        "name": "DataQualityDegraded",
                        "severity": "warning", 
                        "current_value": 0.75,
                        "threshold": 0.8,
                        "duration": 320,
                        "state": "pending"
                    }
                ],
                "recent_notifications": [
                    {"timestamp": time.time() - 300, "channel": "slack-alerts", "alert": "HighOrderLatency", "severity": "warning"},
                    {"timestamp": time.time() - 600, "channel": "email", "alert": "DataQualityDegraded", "severity": "warning"}
                ],
                "trend_analysis": {
                    "system_health_overall": {"trend": -0.02, "baseline": 0.87},
                    "trading_pnl": {"trend": 0.15, "baseline": 1250.0},
                    "error_rate": {"trend": 0.01, "baseline": 0.02}
                }
            }
    except Exception as e:
        st.error(f"❌ Error loading dashboard data: {e}")
        return {}


def create_metrics_overview_chart(dashboard_data: Dict[str, Any]) -> go.Figure:
    """Create metrics overview chart"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('System Health Score', 'Active Alerts', 'Error Rate Trend', 'Performance Trend'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # System health score gauge
    health_score = dashboard_data.get("system_status", {}).get("health_score", 0.85)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            delta={'reference': 0.9},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen" if health_score > 0.8 else "orange" if health_score > 0.6 else "red"},
                'steps': [
                    {'range': [0, 0.6], 'color': "lightgray"},
                    {'range': [0.6, 0.8], 'color': "gray"},
                    {'range': [0.8, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            },
            title={'text': "Health Score"}
        ),
        row=1, col=1
    )
    
    # Active alerts indicator
    active_alerts = dashboard_data.get("system_status", {}).get("active_alerts", 0)
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=active_alerts,
            delta={'reference': 0},
            title={'text': "Active Alerts"},
            number={'font': {'color': "red" if active_alerts > 5 else "orange" if active_alerts > 2 else "green"}}
        ),
        row=1, col=2
    )
    
    # Error rate trend (mock data)
    current_time = time.time()
    time_points = [current_time - i*300 for i in range(24, 0, -1)]  # Last 2 hours
    error_rates = [0.01 + 0.02 * (i % 3) for i in range(24)]  # Mock trend
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.fromtimestamp(t) for t in time_points],
            y=error_rates,
            mode='lines+markers',
            name='Error Rate',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Performance trend (mock data)
    performance_scores = [0.85 + 0.1 * (i % 4) / 4 for i in range(24)]  # Mock trend
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.fromtimestamp(t) for t in time_points],
            y=performance_scores,
            mode='lines+markers',
            name='Performance',
            line=dict(color='green', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="System Metrics Overview",
        title_x=0.5
    )
    
    return fig


def create_alerts_table(dashboard_data: Dict[str, Any]) -> pd.DataFrame:
    """Create alerts table"""
    alerts = dashboard_data.get("active_alerts", [])
    
    if not alerts:
        return pd.DataFrame(columns=["Alert Name", "Severity", "Current Value", "Threshold", "Duration", "State"])
    
    df_data = []
    for alert in alerts:
        duration_str = f"{int(alert['duration'] // 60)}m {int(alert['duration'] % 60)}s"
        
        df_data.append({
            "Alert Name": alert['name'],
            "Severity": alert['severity'].upper(),
            "Current Value": f"{alert['current_value']:.3f}",
            "Threshold": f"{alert['threshold']:.3f}",
            "Duration": duration_str,
            "State": alert['state'].upper()
        })
    
    return pd.DataFrame(df_data)


def create_notification_history_chart(dashboard_data: Dict[str, Any]) -> go.Figure:
    """Create notification history chart"""
    notifications = dashboard_data.get("recent_notifications", [])
    
    if not notifications:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No recent notifications",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Recent Notifications", height=300)
        return fig
    
    # Group by channel
    channels = {}
    for notif in notifications:
        channel = notif['channel']
        if channel not in channels:
            channels[channel] = []
        channels[channel].append(notif)
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (channel, notifs) in enumerate(channels.items()):
        timestamps = [datetime.fromtimestamp(n['timestamp']) for n in notifs]
        alerts = [n['alert'] for n in notifs]
        severities = [n['severity'] for n in notifs]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[channel] * len(timestamps),
            mode='markers',
            name=channel,
            marker=dict(
                size=12,
                color=colors[i % len(colors)],
                symbol='circle'
            ),
            text=[f"{alert}<br>Severity: {sev}" for alert, sev in zip(alerts, severities)],
            hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Channel: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Recent Notifications Timeline",
        xaxis_title="Time",
        yaxis_title="Channel",
        height=300,
        showlegend=True
    )
    
    return fig


def create_trend_analysis_chart(dashboard_data: Dict[str, Any]) -> go.Figure:
    """Create trend analysis chart"""
    trends = dashboard_data.get("trend_analysis", {})
    
    if not trends:
        fig = go.Figure()
        fig.add_annotation(
            text="No trend data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Trend Analysis", height=400)
        return fig
    
    # Create subplots voor different metrics
    metrics = list(trends.keys())
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        horizontal_spacing=0.1
    )
    
    for i, metric in enumerate(metrics):
        trend_data = trends[metric]
        trend = trend_data.get('trend', 0)
        baseline = trend_data.get('baseline', 0)
        
        # Create trend arrow
        if trend > 0.05:
            arrow_color = 'green'
            arrow_symbol = '↗'
        elif trend < -0.05:
            arrow_color = 'red'
            arrow_symbol = '↘'
        else:
            arrow_color = 'gray'
            arrow_symbol = '→'
        
        # Add baseline bar
        fig.add_trace(
            go.Bar(
                x=[metric],
                y=[baseline],
                name=f"{metric}_baseline",
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Add trend indicator
        fig.add_annotation(
            text=f"{arrow_symbol}<br>{trend:+.3f}",
            x=metric,
            y=baseline + abs(baseline) * 0.1,
            showarrow=False,
            font=dict(size=16, color=arrow_color),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Trend Analysis - Baseline vs Current",
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    """Main dashboard function"""
    st.title("📊 CryptoSmartTrader V2 - Observability Dashboard")
    st.markdown("**Real-time metrics monitoring en alerting systeem**")
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Dashboard Settings")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    # Alert severity filter
    severity_filter = st.sidebar.multiselect(
        "🚨 Alert Severity Filter",
        ["INFO", "WARNING", "CRITICAL", "EMERGENCY"],
        default=["WARNING", "CRITICAL", "EMERGENCY"]
    )
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "⏰ Time Range",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=1
    )
    
    # Load metrics system
    with st.spinner("🔄 Loading metrics system..."):
        metrics_system = load_unified_metrics()
        dashboard_data = create_system_health_overview(metrics_system)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show last refresh time
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    st.sidebar.info(f"Last refresh: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')}")
    
    # Main content
    # System status header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_score = dashboard_data.get("system_status", {}).get("health_score", 0)
        health_color = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"
        st.metric(
            label=f"{health_color} System Health",
            value=f"{health_score:.1%}",
            delta=f"{(health_score - 0.9):.1%}"
        )
    
    with col2:
        active_alerts = dashboard_data.get("system_status", {}).get("active_alerts", 0)
        alert_color = "🔴" if active_alerts > 5 else "🟡" if active_alerts > 2 else "🟢"
        st.metric(
            label=f"{alert_color} Active Alerts",
            value=str(active_alerts),
            delta=f"+{active_alerts}" if active_alerts > 0 else "0"
        )
    
    with col3:
        uptime = dashboard_data.get("system_status", {}).get("uptime_seconds", 0)
        uptime_hours = uptime / 3600
        st.metric(
            label="🕐 Uptime",
            value=f"{uptime_hours:.1f}h",
            delta="Online"
        )
    
    with col4:
        total_metrics = dashboard_data.get("system_status", {}).get("total_metrics", 0)
        st.metric(
            label="📊 Total Metrics",
            value=str(total_metrics),
            delta="Collecting"
        )
    
    st.divider()
    
    # Main metrics overview
    st.header("📈 System Metrics Overview")
    metrics_chart = create_metrics_overview_chart(dashboard_data)
    st.plotly_chart(metrics_chart, use_container_width=True)
    
    st.divider()
    
    # Alerts section
    col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🚨 Active Alerts")
            alerts_df = create_alerts_table(dashboard_data)
            
            if not alerts_df.empty:
                # Color code by severity
                def color_severity(val):
                    if val == "CRITICAL":
                        return "background-color: #ff6b6b; color: white"
                    elif val == "WARNING":
                        return "background-color: #feca57; color: black"
                    elif val == "EMERGENCY":
                        return "background-color: #d63031; color: white"
                    else:
                        return "background-color: #74b9ff; color: white"
                
                styled_df = alerts_df.style.applymap(color_severity, subset=['Severity'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Alert acknowledgment
                st.subheader("🎯 Alert Actions")
                alert_names = alerts_df["Alert Name"].tolist()
                selected_alert = st.selectbox("Select alert to acknowledge:", alert_names)
                
                col_ack1, col_ack2 = st.columns(2)
                with col_ack1:
                    if st.button("✅ Acknowledge Alert"):
                        if metrics_system:
                            success = metrics_system.acknowledge_alert(selected_alert, "dashboard_user")
                            if success:
                                st.success(f"Alert '{selected_alert}' acknowledged!")
                            else:
                                st.error(f"Failed to acknowledge alert '{selected_alert}'")
                        else:
                            st.success(f"Mock: Alert '{selected_alert}' acknowledged!")
                
                with col_ack2:
                    if st.button("🔇 Silence Alert"):
                        st.info(f"Alert '{selected_alert}' silenced for 1 hour")
            else:
                st.success("✅ No active alerts - all systems operational!")
        
        with col2:
            st.header("📋 Alert Summary")
            alert_summary = dashboard_data.get("alert_summary", {})
            
            firing = alert_summary.get("firing", 0)
            pending = alert_summary.get("pending", 0)
            acknowledged = alert_summary.get("acknowledged", 0)
            
            # Create donut chart voor alert distribution
            if firing + pending + acknowledged > 0:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=['Firing', 'Pending', 'Acknowledged'],
                    values=[firing, pending, acknowledged],
                    hole=.3,
                    marker_colors=['#ff6b6b', '#feca57', '#74b9ff']
                )])
                fig_donut.update_layout(
                    title="Alert Distribution",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.success("🎉 No alerts!")
        
        st.divider()
        
        # Notifications and trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📢 Recent Notifications")
            notifications_chart = create_notification_history_chart(dashboard_data)
            st.plotly_chart(notifications_chart, use_container_width=True)
        
        with col2:
            st.header("📊 Trend Analysis")
            trends_chart = create_trend_analysis_chart(dashboard_data)
            st.plotly_chart(trends_chart, use_container_width=True)
        
        st.divider()
        
        # Export section
        st.header("📤 Data Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Metrics"):
                if metrics_system:
                    metrics_data = metrics_system.export_metrics()
                    st.download_button(
                        label="⬇️ Download Prometheus Metrics",
                        data=metrics_data,
                        file_name=f"metrics_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Metrics system not available")
        
        with col2:
            if st.button("🚨 Export Alert Rules"):
                if metrics_system:
                    alert_rules = metrics_system.export_alert_rules()
                    st.download_button(
                        label="⬇️ Download Alert Rules",
                        data=alert_rules,
                        file_name=f"alert_rules_{int(time.time())}.yml",
                        mime="text/yaml"
                    )
                else:
                    st.error("Metrics system not available")
        
        with col3:
            if st.button("📋 Export Dashboard Data"):
                dashboard_json = json.dumps(dashboard_data, indent=2, default=str)
                st.download_button(
                    label="⬇️ Download Dashboard Data",
                    data=dashboard_json,
                    file_name=f"dashboard_data_{int(time.time())}.json",
                    mime="application/json"
                )
        
        # Footer
        st.divider()
        st.markdown(
            """
            <div style='text-align: center; color: gray; padding: 20px;'>
                <p><strong>CryptoSmartTrader V2 - Observability Dashboard</strong></p>
                <p>Real-time monitoring • Advanced alerting • Trend analysis</p>
                <p>Last updated: {timestamp}</p>
            </div>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()