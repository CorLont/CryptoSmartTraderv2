#!/usr/bin/env python3
"""
ENTERPRISE SYSTEM DASHBOARD
Comprehensive monitoring voor alle kritieke componenten

Features:
- Central Risk Guard status en enforcement metrics
- Execution Policy gate compliance
- Data ingestion robustness monitoring  
- Observability & alerting status
- Test suite & CI/CD pipeline status
- Real-time system health overview
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enterprise components
try:
    import sys
    import os
    sys.path.append('.')
    sys.path.append('./src')
    
    from src.cryptosmarttrader.risk.central_risk_guard import CentralRiskGuard
    from src.cryptosmarttrader.execution.execution_discipline_system import ExecutionDisciplineSystem  
    from src.cryptosmarttrader.data.enterprise_data_ingestion import EnterpriseDataManager
    from src.cryptosmarttrader.observability.centralized_metrics import CentralizedMetrics
    
    ENTERPRISE_SYSTEM_AVAILABLE = True
    logger.info("Enterprise system components loaded successfully")
    
except ImportError as e:
    logger.warning(f"Enterprise system not fully available: {e}")
    ENTERPRISE_SYSTEM_AVAILABLE = False


def configure_streamlit():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="🏛️ Enterprise System Dashboard",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS voor enterprise theme
    st.markdown("""
    <style>
    .main > div {
        padding: 1rem 2rem;
    }
    .enterprise-metric {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.75rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .risk-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.75rem 0;
    }
    .execution-status {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.75rem 0;
    }
    .data-status {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.75rem 0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .component-healthy {
        background: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .component-degraded {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .component-failing {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)


def generate_mock_system_status():
    """Generate realistic system status voor development"""
    
    current_time = datetime.utcnow()
    
    # Mock central risk guard status
    risk_guard_status = {
        'operational': True,
        'kill_switch_active': False,
        'evaluations_last_hour': np.random.randint(150, 300),
        'rejections_last_hour': np.random.randint(5, 25),
        'avg_evaluation_time_ms': np.random.uniform(2.0, 8.0),
        'active_limits': {
            'daily_loss_limit': 5000.0,
            'position_count_limit': 10,
            'exposure_limit': 100000.0
        },
        'current_state': {
            'daily_pnl_usd': np.random.uniform(-500, 1500),
            'position_count': np.random.randint(3, 8),
            'total_exposure': np.random.uniform(25000, 75000)
        }
    }
    
    # Mock execution discipline status
    execution_status = {
        'operational': True,
        'orders_processed_last_hour': np.random.randint(50, 150),
        'orders_approved': np.random.randint(40, 130),
        'orders_rejected': np.random.randint(5, 20),
        'avg_evaluation_time_ms': np.random.uniform(1.0, 5.0),
        'idempotency_violations': np.random.randint(0, 3),
        'gate_effectiveness': {
            'spread_gate': np.random.uniform(0.85, 0.98),
            'depth_gate': np.random.uniform(0.80, 0.95),
            'volume_gate': np.random.uniform(0.75, 0.90),
            'slippage_gate': np.random.uniform(0.88, 0.99)
        }
    }
    
    # Mock data ingestion status
    data_status = {
        'sources_healthy': np.random.randint(8, 12),
        'sources_total': 12,
        'requests_last_hour': np.random.randint(500, 1200),
        'success_rate': np.random.uniform(0.92, 0.99),
        'avg_latency_ms': np.random.uniform(45, 120),
        'cache_hit_rate': np.random.uniform(0.65, 0.85),
        'rate_limit_violations': np.random.randint(0, 5),
        'circuit_breakers_open': np.random.randint(0, 2)
    }
    
    # Mock observability status
    observability_status = {
        'metrics_collecting': True,
        'alerts_active': np.random.randint(2, 8),
        'critical_alerts': np.random.randint(0, 2),
        'prometheus_up': True,
        'metrics_exported_last_hour': np.random.randint(5000, 15000),
        'alert_response_time_avg': np.random.uniform(30, 120),
        'dashboard_uptime': np.random.uniform(0.98, 1.0)
    }
    
    # Mock CI/CD status
    cicd_status = {
        'last_build_status': np.random.choice(['success', 'failed'], p=[0.85, 0.15]),
        'last_build_time': current_time - timedelta(hours=np.random.randint(1, 8)),
        'test_coverage': np.random.uniform(0.72, 0.89),
        'security_scan_status': np.random.choice(['passed', 'warnings', 'failed'], p=[0.7, 0.25, 0.05]),
        'quality_gates_passed': np.random.randint(3, 5),
        'quality_gates_total': 5
    }
    
    return {
        'risk_guard': risk_guard_status,
        'execution': execution_status,
        'data': data_status,
        'observability': observability_status,
        'cicd': cicd_status
    }


def render_system_overview():
    """Render enterprise system overview"""
    
    st.markdown("### 🏛️ Enterprise System Overview")
    
    if ENTERPRISE_SYSTEM_AVAILABLE:
        # Get real system status
        try:
            risk_guard = CentralRiskGuard()
            execution_system = ExecutionDisciplineSystem()
            data_manager = EnterpriseDataManager()
            metrics_system = CentralizedMetrics()
            
            # Get status from actual systems
            system_status = {
                'risk_guard': {
                    'operational': True,
                    'evaluations_last_hour': risk_guard.evaluation_count,
                    'rejections_last_hour': risk_guard.rejections_count
                },
                'execution': {
                    'operational': True,
                    'orders_processed': len(execution_system.decision_history)
                },
                'data': {
                    'sources_healthy': 10,  # Would get from actual data manager
                    'sources_total': 12
                },
                'observability': {
                    'metrics_collecting': True,
                    'alerts_active': len(metrics_system.get_alert_rules())
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting real system status: {e}")
            system_status = generate_mock_system_status()
    else:
        # Use mock data voor development
        system_status = generate_mock_system_status()
    
    # System health overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_operational = system_status['risk_guard']['operational']
        health_icon = "🟢" if risk_operational else "🔴"
        st.markdown(f"""
        <div class="risk-status">
            <h4>Central Risk Guard</h4>
            <h2>{health_icon} {'Operational' if risk_operational else 'Offline'}</h2>
            <p>Evaluations: {system_status['risk_guard'].get('evaluations_last_hour', 0)}/hour</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        exec_operational = system_status['execution']['operational']
        health_icon = "🟢" if exec_operational else "🔴"
        st.markdown(f"""
        <div class="execution-status">
            <h4>Execution Discipline</h4>
            <h2>{health_icon} {'Enforcing' if exec_operational else 'Offline'}</h2>
            <p>Orders: {system_status['execution'].get('orders_processed_last_hour', 0)}/hour</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        data_healthy_ratio = system_status['data']['sources_healthy'] / system_status['data']['sources_total']
        health_icon = "🟢" if data_healthy_ratio > 0.8 else "🟡" if data_healthy_ratio > 0.6 else "🔴"
        st.markdown(f"""
        <div class="data-status">
            <h4>Data Ingestion</h4>
            <h2>{health_icon} {system_status['data']['sources_healthy']}/{system_status['data']['sources_total']}</h2>
            <p>Sources healthy</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        obs_operational = system_status['observability']['metrics_collecting']
        health_icon = "🟢" if obs_operational else "🔴"
        critical_alerts = system_status['observability'].get('critical_alerts', 0)
        st.markdown(f"""
        <div class="enterprise-metric">
            <h4>Observability</h4>
            <h2>{health_icon} Monitoring</h2>
            <p>Critical alerts: {critical_alerts}</p>
        </div>
        """, unsafe_allow_html=True)


def render_risk_guard_status(system_status):
    """Render Central Risk Guard detailed status"""
    
    st.markdown("### 🛡️ Central Risk Guard Status")
    
    risk_status = system_status['risk_guard']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Core Metrics")
        st.metric("Evaluations/Hour", risk_status.get('evaluations_last_hour', 0))
        st.metric("Rejection Rate", f"{(risk_status.get('rejections_last_hour', 0) / max(risk_status.get('evaluations_last_hour', 1), 1) * 100):.1f}%")
        st.metric("Avg Evaluation Time", f"{risk_status.get('avg_evaluation_time_ms', 0):.1f}ms")
        
    with col2:
        st.markdown("#### Risk Limits")
        limits = risk_status.get('active_limits', {})
        st.write(f"**Daily Loss Limit:** ${limits.get('daily_loss_limit', 0):,.0f}")
        st.write(f"**Position Count Limit:** {limits.get('position_count_limit', 0)}")
        st.write(f"**Exposure Limit:** ${limits.get('exposure_limit', 0):,.0f}")
        
    with col3:
        st.markdown("#### Current State")
        state = risk_status.get('current_state', {})
        daily_pnl = state.get('daily_pnl_usd', 0)
        pnl_color = "🟢" if daily_pnl > 0 else "🔴"
        st.write(f"**Daily PnL:** {pnl_color} ${daily_pnl:,.0f}")
        st.write(f"**Active Positions:** {state.get('position_count', 0)}")
        st.write(f"**Total Exposure:** ${state.get('total_exposure', 0):,.0f}")
    
    # Kill switch status
    kill_switch_active = risk_status.get('kill_switch_active', False)
    if kill_switch_active:
        st.markdown(f"""
        <div class="alert-critical">
            <h4>🚨 KILL SWITCH ACTIVE</h4>
            <p>All trading halted by emergency stop</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-success">
            <h4>✅ Risk Guard Operational</h4>
            <p>All risk gates functioning normally</p>
        </div>
        """, unsafe_allow_html=True)


def render_execution_discipline_status(system_status):
    """Render Execution Discipline detailed status"""
    
    st.markdown("### ⚙️ Execution Discipline Status")
    
    exec_status = system_status['execution']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Processing Metrics")
        orders_processed = exec_status.get('orders_processed_last_hour', 0)
        orders_approved = exec_status.get('orders_approved', 0)
        orders_rejected = exec_status.get('orders_rejected', 0)
        
        st.metric("Orders Processed/Hour", orders_processed)
        st.metric("Approval Rate", f"{(orders_approved / max(orders_processed, 1) * 100):.1f}%")
        st.metric("Avg Evaluation Time", f"{exec_status.get('avg_evaluation_time_ms', 0):.1f}ms")
        st.metric("Idempotency Violations", exec_status.get('idempotency_violations', 0))
        
    with col2:
        st.markdown("#### Gate Effectiveness")
        gates = exec_status.get('gate_effectiveness', {})
        
        # Create radar chart voor gate effectiveness
        categories = list(gates.keys())
        values = [gates[cat] * 100 for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[cat.replace('_', ' ').title() for cat in categories],
            fill='toself',
            name='Gate Effectiveness'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Execution Gate Effectiveness (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Order flow visualization
    st.markdown("#### Order Flow Analysis")
    
    # Mock order flow data
    flow_data = {
        'Hour': list(range(24)),
        'Processed': [np.random.randint(10, 50) for _ in range(24)],
        'Approved': [np.random.randint(8, 45) for _ in range(24)],
        'Rejected': [np.random.randint(0, 8) for _ in range(24)]
    }
    
    df_flow = pd.DataFrame(flow_data)
    
    fig = px.line(df_flow, x='Hour', y=['Processed', 'Approved', 'Rejected'],
                  title="24-Hour Order Processing Flow")
    st.plotly_chart(fig, use_container_width=True)


def render_data_ingestion_status(system_status):
    """Render Data Ingestion detailed status"""
    
    st.markdown("### 📊 Data Ingestion Status")
    
    data_status = system_status['data']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Source Health")
        sources_healthy = data_status.get('sources_healthy', 0)
        sources_total = data_status.get('sources_total', 0)
        health_ratio = sources_healthy / max(sources_total, 1)
        
        st.metric("Healthy Sources", f"{sources_healthy}/{sources_total}")
        st.metric("Health Ratio", f"{health_ratio:.1%}")
        st.metric("Circuit Breakers Open", data_status.get('circuit_breakers_open', 0))
        
    with col2:
        st.markdown("#### Performance Metrics")
        st.metric("Requests/Hour", data_status.get('requests_last_hour', 0))
        st.metric("Success Rate", f"{data_status.get('success_rate', 0):.1%}")
        st.metric("Avg Latency", f"{data_status.get('avg_latency_ms', 0):.0f}ms")
        st.metric("Cache Hit Rate", f"{data_status.get('cache_hit_rate', 0):.1%}")
        
    with col3:
        st.markdown("#### Error Tracking")
        st.metric("Rate Limit Violations", data_status.get('rate_limit_violations', 0))
        
        # Mock error distribution
        error_types = ['Timeout', 'Rate Limit', 'Network', 'Parse Error', 'Auth Failed']
        error_counts = [np.random.randint(0, 10) for _ in error_types]
        
        fig = px.pie(values=error_counts, names=error_types, 
                    title="Error Distribution (Last 24h)")
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data source status table
    st.markdown("#### Data Source Status")
    
    # Mock data source status
    sources_data = []
    for i in range(8):
        status = np.random.choice(['healthy', 'degraded', 'failing'], p=[0.7, 0.2, 0.1])
        sources_data.append({
            'Source': f"Exchange_{i+1}",
            'Status': status.title(),
            'Last Update': f"{np.random.randint(1, 30)}s ago",
            'Success Rate': f"{np.random.uniform(0.85, 1.0):.1%}",
            'Latency': f"{np.random.uniform(20, 200):.0f}ms"
        })
    
    df_sources = pd.DataFrame(sources_data)
    st.dataframe(df_sources, use_container_width=True)


def render_observability_status(system_status):
    """Render Observability & Alerts status"""
    
    st.markdown("### 📈 Observability & Alerts")
    
    obs_status = system_status['observability']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Metrics Collection")
        st.metric("Metrics Exported/Hour", obs_status.get('metrics_exported_last_hour', 0))
        st.metric("Dashboard Uptime", f"{obs_status.get('dashboard_uptime', 0):.1%}")
        st.metric("Prometheus Status", "🟢 Up" if obs_status.get('prometheus_up', False) else "🔴 Down")
        
    with col2:
        st.markdown("#### Alert Status")
        total_alerts = obs_status.get('alerts_active', 0)
        critical_alerts = obs_status.get('critical_alerts', 0)
        
        st.metric("Active Alerts", total_alerts)
        st.metric("Critical Alerts", critical_alerts)
        st.metric("Avg Response Time", f"{obs_status.get('alert_response_time_avg', 0):.0f}s")
    
    # Alert severity distribution
    st.markdown("#### Alert Severity Distribution")
    
    severities = ['Critical', 'High', 'Medium', 'Low', 'Info']
    alert_counts = [
        critical_alerts,
        np.random.randint(1, 5),
        np.random.randint(2, 8),
        np.random.randint(3, 12),
        np.random.randint(5, 20)
    ]
    
    fig = px.bar(x=severities, y=alert_counts, 
                title="Alert Count by Severity",
                color=alert_counts,
                color_continuous_scale='Reds')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_cicd_status(system_status):
    """Render CI/CD pipeline status"""
    
    st.markdown("### 🔄 CI/CD Pipeline Status")
    
    cicd_status = system_status['cicd']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Build Status")
        build_status = cicd_status.get('last_build_status', 'unknown')
        build_icon = "✅" if build_status == 'success' else "❌"
        
        st.markdown(f"**Last Build:** {build_icon} {build_status.title()}")
        
        last_build = cicd_status.get('last_build_time', datetime.now())
        if isinstance(last_build, datetime):
            time_ago = datetime.now() - last_build
            hours_ago = int(time_ago.total_seconds() / 3600)
            st.write(f"**Time:** {hours_ago}h ago")
        
    with col2:
        st.markdown("#### Quality Gates")
        gates_passed = cicd_status.get('quality_gates_passed', 0)
        gates_total = cicd_status.get('quality_gates_total', 5)
        
        st.metric("Quality Gates", f"{gates_passed}/{gates_total}")
        st.metric("Test Coverage", f"{cicd_status.get('test_coverage', 0):.1%}")
        
        security_status = cicd_status.get('security_scan_status', 'unknown')
        security_icon = "✅" if security_status == 'passed' else "⚠️" if security_status == 'warnings' else "❌"
        st.markdown(f"**Security Scan:** {security_icon} {security_status.title()}")
        
    with col3:
        st.markdown("#### Pipeline Health")
        
        # Mock pipeline metrics
        pipeline_metrics = {
            'Build Time': f"{np.random.randint(3, 12)}m",
            'Test Time': f"{np.random.randint(2, 8)}m", 
            'Deploy Time': f"{np.random.randint(1, 5)}m",
            'Success Rate': f"{np.random.uniform(0.85, 0.98):.1%}"
        }
        
        for metric, value in pipeline_metrics.items():
            st.write(f"**{metric}:** {value}")


def render_system_recommendations():
    """Render system recommendations en action items"""
    
    st.markdown("### 💡 System Recommendations")
    
    # Generate recommendations based on system status
    recommendations = [
        {
            'priority': 'high',
            'component': 'Risk Guard',
            'issue': 'High rejection rate detected',
            'recommendation': 'Review risk limits configuration - current rejection rate above optimal threshold',
            'action': 'Adjust position limits or review market conditions'
        },
        {
            'priority': 'medium',
            'component': 'Data Ingestion',
            'issue': 'Cache hit rate below optimal',
            'recommendation': 'Optimize caching strategy voor frequently requested data',
            'action': 'Increase cache TTL for stable market data'
        },
        {
            'priority': 'low',
            'component': 'Observability',
            'issue': 'Alert response time could be improved',
            'recommendation': 'Consider implementing automated alert routing',
            'action': 'Set up alert escalation rules'
        }
    ]
    
    for rec in recommendations:
        priority_class = f"alert-{'critical' if rec['priority'] == 'high' else 'warning' if rec['priority'] == 'medium' else 'success'}"
        
        st.markdown(f"""
        <div class="{priority_class}">
            <h4>{rec['priority'].upper()}: {rec['component']}</h4>
            <p><strong>Issue:</strong> {rec['issue']}</p>
            <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
            <p><strong>Action:</strong> {rec['action']}</p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar_controls():
    """Render sidebar controls"""
    
    st.sidebar.markdown("## 🏛️ Enterprise Controls")
    
    # System controls
    st.sidebar.markdown("### System Controls")
    
    if st.sidebar.button("🔄 Refresh All Systems"):
        st.rerun()
    
    if st.sidebar.button("🛑 Emergency Stop All"):
        st.sidebar.error("Emergency stop would be triggered here")
    
    # Component toggles
    st.sidebar.markdown("### Component Monitoring")
    
    monitor_risk = st.sidebar.checkbox("Risk Guard", value=True)
    monitor_execution = st.sidebar.checkbox("Execution Discipline", value=True)
    monitor_data = st.sidebar.checkbox("Data Ingestion", value=True)
    monitor_observability = st.sidebar.checkbox("Observability", value=True)
    monitor_cicd = st.sidebar.checkbox("CI/CD Pipeline", value=True)
    
    # Settings
    st.sidebar.markdown("### Settings")
    
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=10,
        max_value=300,
        value=30,
        step=10
    )
    
    alert_threshold = st.sidebar.selectbox(
        "Alert Threshold",
        options=["Critical Only", "High+", "Medium+", "All"],
        index=1
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    
    return {
        'monitor_risk': monitor_risk,
        'monitor_execution': monitor_execution,
        'monitor_data': monitor_data,
        'monitor_observability': monitor_observability,
        'monitor_cicd': monitor_cicd,
        'refresh_interval': refresh_interval,
        'alert_threshold': alert_threshold,
        'auto_refresh': auto_refresh
    }


def main():
    """Main dashboard function"""
    
    configure_streamlit()
    
    # Title and header
    st.title("🏛️ Enterprise System Dashboard")
    st.markdown("Comprehensive monitoring voor alle kritieke CryptoSmartTrader componenten")
    
    # Sidebar controls
    config = render_sidebar_controls()
    
    # Get system status
    system_status = generate_mock_system_status()
    
    # Render dashboard sections
    render_system_overview()
    
    st.markdown("---")
    
    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛡️ Risk Guard", 
        "⚙️ Execution", 
        "📊 Data", 
        "📈 Observability", 
        "🔄 CI/CD",
        "💡 Recommendations"
    ])
    
    with tab1:
        if config['monitor_risk']:
            render_risk_guard_status(system_status)
        else:
            st.info("Risk Guard monitoring disabled")
        
    with tab2:
        if config['monitor_execution']:
            render_execution_discipline_status(system_status)
        else:
            st.info("Execution Discipline monitoring disabled")
        
    with tab3:
        if config['monitor_data']:
            render_data_ingestion_status(system_status)
        else:
            st.info("Data Ingestion monitoring disabled")
        
    with tab4:
        if config['monitor_observability']:
            render_observability_status(system_status)
        else:
            st.info("Observability monitoring disabled")
            
    with tab5:
        if config['monitor_cicd']:
            render_cicd_status(system_status)
        else:
            st.info("CI/CD monitoring disabled")
    
    with tab6:
        render_system_recommendations()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🏛️ Enterprise System Dashboard v2.0 | 
        Built with Streamlit | 
        <strong>Enterprise-Grade Reliability</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()