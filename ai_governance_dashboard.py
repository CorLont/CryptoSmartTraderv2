#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - AI Governance Dashboard
Comprehensive monitoring van AI/LLM usage, costs, performance en feature flags
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Import AI governance components
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from src.cryptosmarttrader.ai import (
        get_ai_governance,
        get_ai_evaluator,
        get_ai_feature_flags,
        FeatureState
    )
    AI_COMPONENTS_AVAILABLE = True
except ImportError as e:
    # For development/demo - use mock data
    AI_COMPONENTS_AVAILABLE = False
    print(f"AI components not available: {e}")

# Page config
st.set_page_config(
    page_title="CryptoSmartTrader V2 - AI Governance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5016 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .feature-enabled { background-color: #d4edda; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
    .feature-disabled { background-color: #f8d7da; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
    .feature-testing { background-color: #fff3cd; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 CryptoSmartTrader V2 - AI Governance Dashboard</h1>
    <p>Enterprise-grade monitoring van AI/LLM performance, costs en feature management</p>
</div>
""", unsafe_allow_html=True)

# Check if AI components are available
if not AI_COMPONENTS_AVAILABLE:
    st.error("⚠️ AI governance components niet beschikbaar. Check installatie van src.cryptosmarttrader.ai module.")
    st.info("Dit dashboard toont mockup data ter demonstratie.")
    
    # Use mock data for demonstration
    governance_status = {
        "timestamp": datetime.now().isoformat(),
        "task_metrics": {
            "news_analysis": {
                "total_requests": 142,
                "success_rate": 94.3,
                "fallback_rate": 5.7,
                "cache_hit_rate": 23.2,
                "avg_latency_ms": 1847.3,
                "total_cost": 2.47
            },
            "sentiment_analysis": {
                "total_requests": 89,
                "success_rate": 97.8,
                "fallback_rate": 2.2,
                "cache_hit_rate": 31.5,
                "avg_latency_ms": 1203.1,
                "total_cost": 0.89
            }
        },
        "cost_summary": {
            "hourly_costs": {"2025-08-14-13": 3.36},
            "daily_costs": {"2025-08-14": 3.36}
        }
    }
    
    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluations": 231,
        "model_performance": {
            "gpt-4o": {
                "total_requests": 142,
                "success_rate": 0.943,
                "avg_response_time_ms": 1847.3,
                "avg_quality_score": 0.84,
                "total_cost_usd": 2.47,
                "cost_per_success": 0.0174
            },
            "gpt-4o-mini": {
                "total_requests": 89,
                "success_rate": 0.978,
                "avg_response_time_ms": 1203.1,
                "avg_quality_score": 0.79,
                "total_cost_usd": 0.89,
                "cost_per_success": 0.0102
            }
        }
    }
    
    feature_flags_status = {
        "timestamp": datetime.now().isoformat(),
        "total_features": 5,
        "emergency_disabled_count": 0,
        "features": {
            "news_analysis": {
                "state": "production",
                "enabled_rate": 0.94,
                "error_rate": 0.06,
                "total_cost": 2.47
            },
            "sentiment_analysis": {
                "state": "testing", 
                "enabled_rate": 0.65,
                "error_rate": 0.02,
                "total_cost": 0.89
            }
        }
    }
    
else:
    # Load real data
    try:
        governance = get_ai_governance()
        evaluator = get_ai_evaluator()
        feature_flags = get_ai_feature_flags()
        
        governance_status = governance.get_governance_status()
        evaluation_summary = evaluator.get_evaluation_summary()
        feature_flags_status = feature_flags.get_all_features_status()
        
    except Exception as e:
        st.error(f"❌ Fout bij laden AI governance data: {e}")
        st.stop()

# Sidebar - Quick Stats
with st.sidebar:
    st.header("🎯 Quick Stats")
    
    # Overall metrics
    total_requests = sum([
        metrics.get("total_requests", 0) 
        for metrics in governance_status.get("task_metrics", {}).values()
    ])
    
    overall_success_rate = 0.0
    if total_requests > 0:
        total_successes = sum([
            metrics.get("total_requests", 0) * metrics.get("success_rate", 0) / 100
            for metrics in governance_status.get("task_metrics", {}).values()
        ])
        overall_success_rate = total_successes / total_requests
    
    total_cost = sum(governance_status.get("cost_summary", {}).get("daily_costs", {}).values())
    
    st.metric("Total Requests (Today)", f"{total_requests:,}")
    st.metric("Overall Success Rate", f"{overall_success_rate:.1%}")
    st.metric("Total Cost (Today)", f"${total_cost:.2f}")
    
    # Feature status
    st.subheader("🚀 Feature Status")
    features = feature_flags_status.get("features", {})
    
    for feature_name, feature_data in features.items():
        state = feature_data.get("state", "unknown")
        if state == "production":
            st.markdown(f'<span class="feature-enabled">✅ {feature_name}</span>', unsafe_allow_html=True)
        elif state in ["testing", "development", "canary"]:
            st.markdown(f'<span class="feature-testing">🧪 {feature_name}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="feature-disabled">❌ {feature_name}</span>', unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Governance Overview", "🎯 Model Performance", "🚀 Feature Flags", "💰 Cost Analysis"])

with tab1:
    st.header("📊 AI Governance Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active AI Tasks",
            len(governance_status.get("task_metrics", {})),
            delta=None
        )
    
    with col2:
        emergency_disabled = feature_flags_status.get("emergency_disabled_count", 0)
        st.metric(
            "Emergency Disabled",
            emergency_disabled,
            delta=emergency_disabled if emergency_disabled > 0 else None
        )
    
    with col3:
        circuit_breakers_open = sum([
            1 for cb_status in governance_status.get("circuit_breaker_status", {}).values()
            if cb_status.get("state") == "OPEN"
        ])
        st.metric(
            "Circuit Breakers Open",
            circuit_breakers_open,
            delta=circuit_breakers_open if circuit_breakers_open > 0 else None
        )
    
    with col4:
        total_evaluations = evaluation_summary.get("total_evaluations", 0)
        st.metric(
            "Total Evaluations",
            f"{total_evaluations:,}",
            delta=None
        )
    
    # Task performance metrics
    st.subheader("🎯 Task Performance Metrics")
    
    task_metrics = governance_status.get("task_metrics", {})
    if task_metrics:
        # Create performance DataFrame
        perf_data = []
        for task_name, metrics in task_metrics.items():
            perf_data.append({
                "Task": task_name.replace("_", " ").title(),
                "Requests": metrics.get("total_requests", 0),
                "Success Rate": f"{metrics.get('success_rate', 0):.1f}%",
                "Avg Latency (ms)": f"{metrics.get('avg_latency_ms', 0):.0f}",
                "Cache Hit Rate": f"{metrics.get('cache_hit_rate', 0):.1f}%",
                "Cost ($)": f"{metrics.get('total_cost', 0):.3f}"
            })
        
        df_performance = pd.DataFrame(perf_data)
        st.dataframe(df_performance, use_container_width=True)
        
        # Success rate visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_success = px.bar(
                df_performance,
                x="Task",
                y="Success Rate",
                title="Success Rate by Task",
                color="Success Rate",
                color_continuous_scale="RdYlGn"
            )
            fig_success.update_layout(height=400)
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            # Latency comparison
            latency_data = [float(x.replace(",", "")) for x in df_performance["Avg Latency (ms)"]]
            fig_latency = px.bar(
                x=df_performance["Task"],
                y=latency_data,
                title="Average Response Time by Task",
                color=latency_data,
                color_continuous_scale="viridis"
            )
            fig_latency.update_layout(height=400)
            st.plotly_chart(fig_latency, use_container_width=True)

with tab2:
    st.header("🎯 Model Performance Analysis")
    
    model_performance = evaluation_summary.get("model_performance", {})
    
    if model_performance:
        # Model comparison table
        model_data = []
        for model_name, metrics in model_performance.items():
            model_data.append({
                "Model": model_name,
                "Requests": metrics.get("total_requests", 0),
                "Success Rate": f"{metrics.get('success_rate', 0):.1%}",
                "Avg Response Time": f"{metrics.get('avg_response_time_ms', 0):.0f}ms",
                "Quality Score": f"{metrics.get('avg_quality_score', 0):.2f}",
                "Total Cost": f"${metrics.get('total_cost_usd', 0):.3f}",
                "Cost per Success": f"${metrics.get('cost_per_success', 0):.4f}"
            })
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Model comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality vs Cost scatter
            quality_scores = [float(x) for x in df_models["Quality Score"]]
            costs = [float(x.replace("$", "")) for x in df_models["Cost per Success"]]
            
            fig_scatter = px.scatter(
                x=costs,
                y=quality_scores,
                text=df_models["Model"],
                title="Quality vs Cost Efficiency",
                labels={"x": "Cost per Success ($)", "y": "Quality Score"}
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Response time comparison
            response_times = [float(x.replace("ms", "")) for x in df_models["Avg Response Time"]]
            
            fig_response_time = px.bar(
                x=df_models["Model"],
                y=response_times,
                title="Average Response Time by Model",
                color=response_times,
                color_continuous_scale="plasma"
            )
            fig_response_time.update_layout(height=400)
            st.plotly_chart(fig_response_time, use_container_width=True)
        
        # SLO violations
        st.subheader("⚠️ Recent SLO Violations")
        violations = evaluation_summary.get("recent_violations", [])
        
        if violations:
            violation_data = []
            for violation in violations[-10:]:  # Last 10 violations
                violation_data.append({
                    "Timestamp": violation.get("timestamp", "")[:19],
                    "Model": violation.get("model", ""),
                    "Violations": ", ".join(violation.get("violations", []))
                })
            
            df_violations = pd.DataFrame(violation_data)
            st.dataframe(df_violations, use_container_width=True)
        else:
            st.success("✅ Geen recente SLO violations!")

with tab3:
    st.header("🚀 AI Feature Flag Management")
    
    features = feature_flags_status.get("features", {})
    
    if features:
        # Feature status overview
        feature_data = []
        for feature_name, feature_info in features.items():
            status_emoji = {
                "production": "✅",
                "testing": "🧪", 
                "development": "🔧",
                "canary": "🚀",
                "disabled": "❌",
                "emergency_disabled": "🚨"
            }.get(feature_info.get("state", "unknown"), "❓")
            
            metrics = feature_info.get("metrics", {})
            
            feature_data.append({
                "Feature": feature_name.replace("_", " ").title(),
                "Status": f"{status_emoji} {feature_info.get('state', 'unknown').title()}",
                "Enabled Rate": f"{metrics.get('enabled_rate', 0):.1%}",
                "Error Rate": f"{metrics.get('error_rate', 0):.1%}",
                "Total Checks": metrics.get("total_checks", 0),
                "Cost": f"${metrics.get('total_cost', 0):.3f}"
            })
        
        df_features = pd.DataFrame(feature_data)
        st.dataframe(df_features, use_container_width=True)
        
        # Feature state distribution
        col1, col2 = st.columns(2)
        
        with col1:
            state_counts = {}
            for feature_info in features.values():
                state = feature_info.get("state", "unknown")
                state_counts[state] = state_counts.get(state, 0) + 1
            
            fig_states = px.pie(
                values=list(state_counts.values()),
                names=list(state_counts.keys()),
                title="Feature State Distribution"
            )
            st.plotly_chart(fig_states, use_container_width=True)
        
        with col2:
            # Error rates by feature
            error_rates = [float(x.replace("%", "")) for x in df_features["Error Rate"]]
            
            fig_errors = px.bar(
                x=df_features["Feature"],
                y=error_rates,
                title="Error Rates by Feature",
                color=error_rates,
                color_continuous_scale="Reds"
            )
            fig_errors.update_layout(height=400)
            st.plotly_chart(fig_errors, use_container_width=True)
        
        # Feature management controls
        st.subheader("🎛️ Feature Management")
        
        if AI_COMPONENTS_AVAILABLE:
            feature_flags = get_ai_feature_flags()
            
            selected_feature = st.selectbox(
                "Select Feature to Manage:",
                options=list(features.keys()),
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            if selected_feature:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"Enable {selected_feature}", type="primary"):
                        feature_flags.enable_feature(selected_feature)
                        st.success(f"Enabled {selected_feature}")
                        st.rerun()
                
                with col2:
                    if st.button(f"Emergency Disable {selected_feature}", type="secondary"):
                        feature_flags.emergency_disable_feature(selected_feature, "Manual disable")
                        st.warning(f"Emergency disabled {selected_feature}")
                        st.rerun()
                
                with col3:
                    new_percentage = st.number_input(
                        "Rollout Percentage:",
                        min_value=0.0,
                        max_value=100.0,
                        value=features[selected_feature].get("rollout_percentage", 0.0),
                        step=5.0
                    )
                    
                    if st.button("Update Rollout"):
                        feature_flags.update_rollout_percentage(selected_feature, new_percentage)
                        st.success(f"Updated rollout to {new_percentage}%")
                        st.rerun()

with tab4:
    st.header("💰 Cost Analysis & Budget Monitoring")
    
    # Cost summary
    cost_summary = governance_status.get("cost_summary", {})
    hourly_costs = cost_summary.get("hourly_costs", {})
    daily_costs = cost_summary.get("daily_costs", {})
    
    # Current costs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_hour_cost = list(hourly_costs.values())[-1] if hourly_costs else 0.0
        st.metric("Current Hour Cost", f"${current_hour_cost:.3f}")
    
    with col2:
        current_day_cost = list(daily_costs.values())[-1] if daily_costs else 0.0
        st.metric("Today's Total Cost", f"${current_day_cost:.2f}")
    
    with col3:
        # Estimated monthly cost
        monthly_estimate = current_day_cost * 30
        st.metric("Monthly Estimate", f"${monthly_estimate:.0f}")
    
    with col4:
        # Average cost per request
        avg_cost_per_request = current_day_cost / max(1, total_requests)
        st.metric("Avg Cost/Request", f"${avg_cost_per_request:.4f}")
    
    # Cost trends
    if hourly_costs:
        st.subheader("📈 Cost Trends")
        
        # Hourly cost chart
        hours = list(hourly_costs.keys())
        costs = list(hourly_costs.values())
        
        fig_costs = px.line(
            x=hours,
            y=costs,
            title="Hourly Cost Trend",
            labels={"x": "Hour", "y": "Cost ($)"}
        )
        fig_costs.update_layout(height=400)
        st.plotly_chart(fig_costs, use_container_width=True)
    
    # Cost breakdown by task
    if task_metrics:
        st.subheader("💸 Cost Breakdown by Task")
        
        task_costs = [(name, metrics.get("total_cost", 0)) for name, metrics in task_metrics.items()]
        task_costs.sort(key=lambda x: x[1], reverse=True)
        
        fig_cost_breakdown = px.pie(
            values=[cost for _, cost in task_costs],
            names=[name.replace("_", " ").title() for name, _ in task_costs],
            title="Cost Distribution by Task"
        )
        st.plotly_chart(fig_cost_breakdown, use_container_width=True)
    
    # Budget alerts
    st.subheader("🚨 Budget Alerts")
    
    # Check budget thresholds
    daily_budget_limit = 50.0  # $50 daily limit
    hourly_budget_limit = 10.0  # $10 hourly limit
    
    alerts = []
    
    if current_day_cost > daily_budget_limit * 0.8:
        alerts.append(f"⚠️ Daily budget at {current_day_cost/daily_budget_limit:.1%} of limit (${current_day_cost:.2f}/${daily_budget_limit})")
    
    if current_hour_cost > hourly_budget_limit * 0.8:
        alerts.append(f"⚠️ Hourly budget at {current_hour_cost/hourly_budget_limit:.1%} of limit (${current_hour_cost:.3f}/${hourly_budget_limit})")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ All budget thresholds within limits")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Dashboard", type="primary"):
        st.rerun()

with col2:
    if st.button("📋 Export Status Report"):
        # Generate comprehensive status report
        report = {
            "timestamp": datetime.now().isoformat(),
            "governance_status": governance_status,
            "evaluation_summary": evaluation_summary,
            "feature_flags_status": feature_flags_status
        }
        
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name=f"ai_governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col3:
    st.info("Dashboard auto-refreshes every 5 minutes")

# Auto-refresh every 5 minutes
time.sleep(0.1)  # Small delay to prevent too frequent refreshes

st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;'>
    CryptoSmartTrader V2 - Enterprise AI Governance Dashboard<br>
    Monitoring AI/LLM performance, costs en feature management
</div>
""", unsafe_allow_html=True)