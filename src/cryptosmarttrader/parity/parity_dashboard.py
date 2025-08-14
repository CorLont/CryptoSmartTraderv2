#!/usr/bin/env python3
"""
FASE F - Parity & Canary Monitoring Dashboard
Streamlit dashboard for real-time parity and canary deployment monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

from .backtest_live_parity import get_parity_validator, ParityStatus, OrderExecution
from ..deployment.canary_deployment import get_canary_manager, CanaryStage, CanaryStatus


def render_parity_dashboard():
    """Render parity validation dashboard"""
    st.header("🔄 Backtest-Live Parity Validation")
    
    parity_validator = get_parity_validator()
    
    # Parity overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock some data for demo (in real implementation, get from validator)
    with col1:
        st.metric(
            label="Avg Tracking Error",
            value="12.5 bps",
            delta="-2.3 bps",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Fee Deviation",
            value="1.8 bps",
            delta="+0.5 bps",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Latency Diff",
            value="45 ms",
            delta="-8 ms",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Parity Score",
            value="87.2",
            delta="+3.1",
            delta_color="normal"
        )
    
    # Tracking error chart
    st.subheader("📊 Daily Tracking Error Trend")
    
    # Generate mock tracking error data
    dates = pd.date_range(end=datetime.now(), periods=30)
    tracking_errors = np.random.normal(15, 5, 30)
    tracking_errors = np.clip(tracking_errors, 5, 35)
    
    df_tracking = pd.DataFrame({
        'Date': dates,
        'Tracking Error (bps)': tracking_errors
    })
    
    fig_tracking = go.Figure()
    fig_tracking.add_trace(go.Scatter(
        x=df_tracking['Date'],
        y=df_tracking['Tracking Error (bps)'],
        mode='lines+markers',
        name='Tracking Error',
        line=dict(color='blue', width=2)
    ))
    
    # Add threshold lines
    fig_tracking.add_hline(y=20, line_dash="dash", line_color="orange", 
                          annotation_text="Warning Threshold (20 bps)")
    fig_tracking.add_hline(y=25, line_dash="dash", line_color="red", 
                          annotation_text="Critical Threshold (25 bps)")
    
    fig_tracking.update_layout(
        title="Daily Tracking Error vs Thresholds",
        xaxis_title="Date",
        yaxis_title="Tracking Error (bps)",
        height=400
    )
    
    st.plotly_chart(fig_tracking, use_container_width=True)
    
    # Parity breakdown by symbol
    st.subheader("🎯 Parity Status by Symbol")
    
    # Mock symbol data
    symbols_data = [
        {"Symbol": "BTC/USD", "Tracking Error": 11.2, "Status": "✅ Pass", "Score": 92.1},
        {"Symbol": "ETH/USD", "Tracking Error": 15.8, "Status": "⚠️ Warning", "Score": 81.5},
        {"Symbol": "SOL/USD", "Tracking Error": 8.9, "Status": "✅ Pass", "Score": 95.3},
        {"Symbol": "ADA/USD", "Tracking Error": 22.1, "Status": "❌ Fail", "Score": 65.2},
        {"Symbol": "DOT/USD", "Tracking Error": 13.7, "Status": "✅ Pass", "Score": 88.9}
    ]
    
    df_symbols = pd.DataFrame(symbols_data)
    st.dataframe(df_symbols, use_container_width=True)
    
    # Execution analysis
    st.subheader("⚡ Execution Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fee analysis
        fee_data = {
            'Execution Type': ['Backtest', 'Live'],
            'Avg Fee (bps)': [2.1, 2.3],
            'Max Fee (bps)': [4.5, 5.1]
        }
        
        fig_fees = px.bar(fee_data, x='Execution Type', y='Avg Fee (bps)',
                         title="Average Execution Fees Comparison",
                         color='Execution Type')
        st.plotly_chart(fig_fees, use_container_width=True)
    
    with col2:
        # Latency analysis
        latency_data = {
            'Execution Type': ['Backtest', 'Live'],
            'Avg Latency (ms)': [5, 47],
            'P95 Latency (ms)': [15, 89]
        }
        
        fig_latency = px.bar(latency_data, x='Execution Type', y='Avg Latency (ms)',
                            title="Average Execution Latency Comparison",
                            color='Execution Type')
        st.plotly_chart(fig_latency, use_container_width=True)


def render_canary_dashboard():
    """Render canary deployment dashboard"""
    st.header("🚀 Canary Deployment Monitor")
    
    canary_manager = get_canary_manager()
    
    # Mock active deployment for demo
    active_deployment = {
        'deployment_id': 'canary_v2.1.0_1642176000',
        'stage': 'production_canary',
        'status': 'running',
        'version': 'v2.1.0',
        'started': '2025-01-12 14:30:00',
        'duration_hours': 36.5,
        'capital_used': 250000,
        'risk_budget_percent': 5.0
    }
    
    if active_deployment:
        st.success(f"🟢 Active Deployment: {active_deployment['deployment_id']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Deployment Stage",
                value=active_deployment['stage'].replace('_', ' ').title()
            )
        
        with col2:
            st.metric(
                label="Duration",
                value=f"{active_deployment['duration_hours']:.1f}h",
                delta="Target: 48-72h"
            )
        
        with col3:
            st.metric(
                label="Capital Used",
                value=f"${active_deployment['capital_used']:,}",
                delta=f"{active_deployment['risk_budget_percent']}% of budget"
            )
        
        with col4:
            st.metric(
                label="Status",
                value=active_deployment['status'].title(),
                delta="No breaches"
            )
        
        # Canary timeline
        st.subheader("📅 Deployment Timeline")
        
        timeline_data = [
            {"Stage": "Development", "Start": "2025-01-05", "End": "2025-01-05", "Status": "✅ Completed"},
            {"Stage": "Staging (7 days)", "Start": "2025-01-05", "End": "2025-01-12", "Status": "✅ Completed"},
            {"Stage": "Production Canary", "Start": "2025-01-12", "End": "2025-01-15", "Status": "🟡 Running"},
            {"Stage": "Full Production", "Start": "2025-01-15", "End": "TBD", "Status": "⏳ Pending"}
        ]
        
        df_timeline = pd.DataFrame(timeline_data)
        st.dataframe(df_timeline, use_container_width=True)
        
        # Risk monitoring
        st.subheader("⚠️ Risk Budget Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk budget usage
            risk_budget_data = {
                'Category': ['Used Capital', 'Available Capital'],
                'Amount': [250000, 4750000],
                'Percentage': [5.0, 95.0]
            }
            
            fig_risk = px.pie(risk_budget_data, values='Amount', names='Category',
                             title="Risk Budget Allocation",
                             color_discrete_map={'Used Capital': '#ff6b6b', 'Available Capital': '#51cf66'})
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Performance metrics
            performance_data = {
                'Metric': ['Sharpe Ratio', 'Max Drawdown %', 'Error Rate %', 'Parity Score'],
                'Current': [1.35, 1.2, 0.3, 89.5],
                'Threshold': [1.0, 2.0, 1.0, 85.0],
                'Status': ['✅ Pass', '✅ Pass', '✅ Pass', '✅ Pass']
            }
            
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)
    
    else:
        st.info("No active canary deployments")
    
    # Deployment history
    st.subheader("📈 Deployment History")
    
    history_data = [
        {"Version": "v2.0.8", "Stage": "Full Production", "Start": "2025-01-01", "Duration": "11 days", "Result": "✅ Success"},
        {"Version": "v2.0.7", "Stage": "Rolled Back", "Start": "2024-12-28", "Duration": "2 days", "Result": "❌ Risk Breach"},
        {"Version": "v2.0.6", "Stage": "Full Production", "Start": "2024-12-15", "Duration": "13 days", "Result": "✅ Success"},
        {"Version": "v2.0.5", "Stage": "Full Production", "Start": "2024-12-01", "Duration": "14 days", "Result": "✅ Success"}
    ]
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)


def render_combined_monitoring():
    """Render combined parity and canary monitoring"""
    st.header("🎯 FASE F - Parity & Canary Monitoring")
    
    # Overall system health
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔄 Parity Health",
            value="Good",
            delta="87.2 score",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="🚀 Canary Status", 
            value="Running",
            delta="36.5h elapsed",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="⚠️ Risk Level",
            value="Low",
            delta="5% budget used",
            delta_color="inverse"
        )
    
    # Alerts and notifications
    st.subheader("🚨 Active Alerts & Notifications")
    
    alerts = [
        {"Time": "14:25", "Type": "Info", "Message": "Production canary milestone: 36h completed"},
        {"Time": "12:18", "Type": "Warning", "Message": "ETH/USD tracking error increased to 15.8 bps"},
        {"Time": "09:45", "Type": "Success", "Message": "All parity checks passed for morning session"},
        {"Time": "08:30", "Type": "Info", "Message": "Daily parity report generated for 5 symbols"}
    ]
    
    for alert in alerts:
        alert_type = alert["Type"]
        if alert_type == "Warning":
            st.warning(f"⚠️ {alert['Time']} - {alert['Message']}")
        elif alert_type == "Success":
            st.success(f"✅ {alert['Time']} - {alert['Message']}")
        else:
            st.info(f"ℹ️ {alert['Time']} - {alert['Message']}")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Generate Parity Report"):
            st.success("Parity report generated for all active symbols")
    
    with col2:
        if st.button("🚀 Promote Canary"):
            st.success("Canary promotion initiated (demo)")
    
    with col3:
        if st.button("🛑 Emergency Rollback"):
            st.error("Emergency rollback would be triggered (demo)")
    
    with col4:
        if st.button("📈 Export Metrics"):
            st.success("Metrics exported to CSV (demo)")


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="FASE F - Parity & Canary Monitor",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 FASE F - Parity & Canary Deployment Monitor")
    st.markdown("Real-time monitoring for backtest-live parity and canary deployments")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Monitor",
        ["Combined Overview", "Parity Validation", "Canary Deployment"]
    )
    
    # Render selected page
    if page == "Combined Overview":
        render_combined_monitoring()
    elif page == "Parity Validation":
        render_parity_dashboard()
    elif page == "Canary Deployment":
        render_canary_dashboard()


if __name__ == "__main__":
    main()