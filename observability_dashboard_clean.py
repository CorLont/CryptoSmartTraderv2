#!/usr/bin/env python3
"""
Observability Dashboard - Clean Version
Real-time metrics en alerting dashboard voor CryptoSmartTrader V2
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

st.set_page_config(
    page_title="CryptoSmartTrader V2 - Observability",
    page_icon="📊",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mock_data() -> Dict[str, Any]:
    """Load mock dashboard data"""
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
        ]
    }


def create_metrics_chart(data: Dict[str, Any]) -> go.Figure:
    """Create metrics overview chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('System Health', 'Active Alerts', 'Error Rate', 'Performance'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Health score gauge
    health_score = data.get("system_status", {}).get("health_score", 0.85)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "green" if health_score > 0.8 else "orange"},
                   'steps': [{'range': [0, 0.6], 'color': "lightgray"},
                            {'range': [0.6, 1], 'color': "lightgreen"}]}
        ),
        row=1, col=1
    )
    
    # Active alerts
    active_alerts = data.get("system_status", {}).get("active_alerts", 0)
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=active_alerts,
            title={'text': "Active Alerts"}
        ),
        row=1, col=2
    )
    
    # Mock time series data
    current_time = time.time()
    time_points = [current_time - i*300 for i in range(24, 0, -1)]
    error_rates = [0.01 + 0.02 * (i % 3) for i in range(24)]
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.fromtimestamp(t) for t in time_points],
            y=error_rates,
            mode='lines',
            name='Error Rate'
        ),
        row=2, col=1
    )
    
    performance_scores = [0.85 + 0.1 * (i % 4) / 4 for i in range(24)]
    fig.add_trace(
        go.Scatter(
            x=[datetime.fromtimestamp(t) for t in time_points],
            y=performance_scores,
            mode='lines',
            name='Performance'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig


def create_alerts_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Create alerts table"""
    alerts = data.get("active_alerts", [])
    
    if not alerts:
        return pd.DataFrame()
    
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


def main():
    """Main dashboard function"""
    st.title("📊 CryptoSmartTrader V2 - Observability Dashboard")
    st.markdown("**Real-time metrics monitoring en alerting systeem**")
    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Settings")
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    time_range = st.sidebar.selectbox(
        "⏰ Time Range",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours"],
        index=1
    )
    
    # Load data
    dashboard_data = load_mock_data()
    
    # System status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_score = dashboard_data["system_status"]["health_score"]
        health_color = "🟢" if health_score > 0.8 else "🟡"
        st.metric(
            label=f"{health_color} System Health",
            value=f"{health_score:.1%}",
            delta="Normal"
        )
    
    with col2:
        active_alerts = dashboard_data["system_status"]["active_alerts"]
        alert_color = "🔴" if active_alerts > 5 else "🟡" if active_alerts > 2 else "🟢"
        st.metric(
            label=f"{alert_color} Active Alerts",
            value=str(active_alerts),
            delta=f"+{active_alerts}" if active_alerts > 0 else "0"
        )
    
    with col3:
        uptime = dashboard_data["system_status"]["uptime_seconds"]
        uptime_hours = uptime / 3600
        st.metric(
            label="🕐 Uptime",
            value=f"{uptime_hours:.1f}h",
            delta="Online"
        )
    
    with col4:
        total_metrics = dashboard_data["system_status"]["total_metrics"]
        st.metric(
            label="📊 Total Metrics",
            value=str(total_metrics),
            delta="Collecting"
        )
    
    st.divider()
    
    # Main metrics chart
    st.header("📈 System Metrics Overview")
    metrics_chart = create_metrics_chart(dashboard_data)
    st.plotly_chart(metrics_chart, use_container_width=True)
    
    st.divider()
    
    # Alerts section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚨 Active Alerts")
        alerts_df = create_alerts_table(dashboard_data)
        
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True)
            
            # Alert actions
            st.subheader("🎯 Alert Actions")
            alert_names = alerts_df["Alert Name"].tolist()
            selected_alert = st.selectbox("Select alert:", alert_names)
            
            col_ack1, col_ack2 = st.columns(2)
            with col_ack1:
                if st.button("✅ Acknowledge"):
                    st.success(f"Alert '{selected_alert}' acknowledged!")
            
            with col_ack2:
                if st.button("🔇 Silence"):
                    st.info(f"Alert '{selected_alert}' silenced!")
        else:
            st.success("✅ No active alerts - all systems operational!")
    
    with col2:
        st.header("📋 Alert Summary")
        alert_summary = dashboard_data["alert_summary"]
        
        firing = alert_summary["firing"]
        pending = alert_summary["pending"]
        acknowledged = alert_summary["acknowledged"]
        
        if firing + pending + acknowledged > 0:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Firing', 'Pending', 'Acknowledged'],
                values=[firing, pending, acknowledged],
                hole=.3
            )])
            fig_donut.update_layout(height=300)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.success("🎉 No alerts!")
    
    st.divider()
    
    # Export section
    st.header("📤 Data Export")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Metrics"):
            st.info("Metrics exported!")
    
    with col2:
        if st.button("🚨 Export Alerts"):
            st.info("Alert rules exported!")
    
    with col3:
        if st.button("📋 Export Data"):
            dashboard_json = json.dumps(dashboard_data, indent=2)
            st.download_button(
                label="⬇️ Download",
                data=dashboard_json,
                file_name="dashboard_data.json",
                mime="application/json"
            )
    
    # Footer
    st.divider()
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            <p><strong>CryptoSmartTrader V2 - Observability Dashboard</strong></p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()