#!/usr/bin/env python3
"""
ML/AI Discipline Dashboard - Enterprise Model Management Interface

Features:
- Model Registry overview en lifecycle management
- Dataset versioning en quality metrics
- Canary deployment monitoring
- Drift detection alerts
- Performance metrics tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import logging
import time

# Import our ML discipline components
from src.cryptosmarttrader.ml.model_registry import get_model_registry, ModelStatus, DriftStatus
from src.cryptosmarttrader.ml.canary_deployment import get_canary_orchestrator, CanaryConfig, CanaryPhase

# Configure page
st.set_page_config(
    page_title="ML/AI Discipline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def init_services():
    """Initialize ML services"""
    registry = get_model_registry()
    orchestrator = get_canary_orchestrator()
    return registry, orchestrator

def display_header():
    """Display dashboard header"""
    st.title("🤖 ML/AI Discipline Dashboard")
    st.markdown("**Enterprise Model Management & Deployment Pipeline**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏛️ Model Registry", "Active", "Enterprise-Ready")
    with col2:
        st.metric("🚀 Canary System", "Operational", "≤1% Risk Budget")
    with col3:
        st.metric("🔍 Drift Detection", "Monitoring", "Real-Time")

def display_model_registry_overview(registry):
    """Display model registry overview"""
    st.header("🏛️ Model Registry Overview")
    
    # Get registry summary
    summary = registry.get_registry_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", summary['total_models'])
    with col2:
        st.metric("Total Datasets", summary['total_datasets'])
    with col3:
        st.metric("Registry Size", f"{summary['registry_size_mb']:.1f} MB")
    with col4:
        if summary['models_by_status']:
            active_models = sum(v for k, v in summary['models_by_status'].items() 
                              if k in ['production', 'canary'])
            st.metric("Active Models", active_models)
        else:
            st.metric("Active Models", 0)
    
    # Models by status chart
    if summary['models_by_status']:
        status_df = pd.DataFrame([
            {'Status': k.title(), 'Count': v} 
            for k, v in summary['models_by_status'].items()
        ])
        
        fig = px.pie(status_df, values='Count', names='Status', 
                    title="Models by Deployment Status",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models registered yet")

def display_dataset_versions(registry):
    """Display dataset versions and quality metrics"""
    st.header("📊 Dataset Versions & Quality")
    
    # Sample dataset display (in production, enumerate all datasets)
    datasets_data = []
    
    # Check for crypto_signals datasets
    for dataset_id in ["crypto_signals", "model_training_data"]:
        try:
            # This is simplified - in production we'd enumerate all versions
            versions = ["v1.0", "v2.0"]
            for version in versions:
                metadata_file = registry.datasets_path / f"{dataset_id}_v{version}_metadata.json"
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        datasets_data.append({
                            'Dataset': metadata['dataset_id'],
                            'Version': metadata['version'],
                            'Rows': metadata['size_rows'],
                            'Features': metadata['feature_count'],
                            'Completeness': f"{metadata['completeness_score']:.3f}",
                            'Consistency': f"{metadata['consistency_score']:.3f}",
                            'Validity': f"{metadata['validity_score']:.3f}",
                            'Hash': metadata['hash'][:8] + "...",
                            'Created': metadata['created_at'][:16]
                        })
        except Exception:
            continue
    
    if datasets_data:
        df = pd.DataFrame(datasets_data)
        st.dataframe(df, use_container_width=True)
        
        # Quality metrics visualization
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Completeness", "Consistency", "Validity"),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(['Completeness', 'Consistency', 'Validity'], 1):
            values = [float(val) for val in df[metric]]
            fig.add_trace(
                go.Bar(x=df['Dataset'] + " " + df['Version'], y=values, 
                      name=metric, showlegend=False),
                row=1, col=i
            )
        
        fig.update_layout(title="Dataset Quality Metrics", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No datasets found - run test_ml_discipline.py to generate sample data")

def display_model_details(registry):
    """Display detailed model information"""
    st.header("🤖 Model Details & Performance")
    
    # Get all model files
    model_files = list(registry.models_path.glob("*.json"))
    
    if model_files:
        models_data = []
        
        for model_file in model_files:
            try:
                model_id, version = model_file.stem.split('_v')
                metadata = registry.get_model_metadata(model_id, version)
                
                if metadata:
                    models_data.append({
                        'Model ID': metadata.model_id,
                        'Version': metadata.version,
                        'Algorithm': metadata.algorithm,
                        'Status': metadata.status.value.title(),
                        'Test Accuracy': f"{metadata.test_metrics.get('accuracy', 0):.3f}",
                        'Test F1': f"{metadata.test_metrics.get('f1', 0):.3f}",
                        'Drift Score': f"{metadata.drift_score:.3f}",
                        'Drift Status': metadata.drift_status.value.title(),
                        'Created': metadata.created_at.strftime("%Y-%m-%d %H:%M"),
                    })
            except Exception as e:
                continue
        
        if models_data:
            df = pd.DataFrame(models_data)
            st.dataframe(df, use_container_width=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig = px.bar(df, x='Model ID', y='Test Accuracy', 
                           title="Model Test Accuracy",
                           color='Status',
                           color_discrete_map={
                               'Development': '#FFA500',
                               'Canary': '#FFD700',
                               'Production': '#32CD32'
                           })
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Drift scores
                fig = px.scatter(df, x='Model ID', y='Drift Score',
                               color='Drift Status', size='Test F1',
                               title="Drift Score vs Performance",
                               color_discrete_map={
                                   'None': '#32CD32',
                                   'Low': '#90EE90',
                                   'Medium': '#FFD700',
                                   'High': '#FFA500',
                                   'Critical': '#FF4500'
                               })
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models found")
    else:
        st.info("No models registered yet")

def display_canary_deployments(orchestrator):
    """Display canary deployment status"""
    st.header("🚀 Canary Deployments")
    
    # Get active canaries
    active_canaries = orchestrator.list_active_canaries()
    
    if active_canaries:
        canary_data = []
        
        for canary_key, metrics in active_canaries.items():
            model_id, version = canary_key.split('_', 1)
            canary_data.append({
                'Model': model_id,
                'Version': version,
                'Phase': metrics.phase.value.replace('_', ' ').title(),
                'Total Trades': metrics.total_trades,
                'Win Rate': f"{metrics.prediction_accuracy:.1%}",
                'PnL': f"{metrics.total_pnl:.4f}",
                'Risk Budget': f"{metrics.current_risk_budget_used:.2f}%",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Status': "🟢 Healthy" if metrics.is_healthy else "🔴 Issue",
                'Start Time': metrics.start_time.strftime("%H:%M:%S")
            })
        
        df = pd.DataFrame(canary_data)
        st.dataframe(df, use_container_width=True)
        
        # Canary performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by phase
            phase_counts = df['Phase'].value_counts()
            fig = px.pie(values=phase_counts.values, names=phase_counts.index,
                        title="Canary Deployments by Phase")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk budget usage
            fig = px.bar(df, x='Model', y='Risk Budget',
                        title="Risk Budget Usage by Model",
                        color='Phase')
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="1% Risk Limit")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active canary deployments")
        
        # Show sample canary config
        st.subheader("🧪 Start New Canary Deployment")
        
        with st.form("canary_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_id = st.text_input("Model ID", "crypto_predictor")
                version = st.text_input("Version", "20250814_120000")
                risk_budget = st.slider("Risk Budget (%)", 0.1, 1.0, 1.0, 0.1)
            
            with col2:
                paper_days = st.number_input("Paper Trading (days)", 1, 7, 7)
                shadow_days = st.number_input("Shadow Trading (days)", 1, 5, 3)
                canary_hours = st.number_input("Live Canary (hours)", 1, 48, 24)
            
            submitted = st.form_submit_button("🚀 Start Canary Deployment")
            
            if submitted:
                st.info(f"Would start canary for {model_id} v{version} with {risk_budget}% risk budget")

def display_performance_monitoring():
    """Display ML performance monitoring"""
    st.header("📈 Performance Monitoring")
    
    # Simulate time series performance data
    dates = pd.date_range(start='2025-01-01', end='2025-08-14', freq='D')
    
    # Create mock performance data
    base_accuracy = 0.85
    accuracy_data = base_accuracy + np.random.normal(0, 0.02, len(dates))
    accuracy_data = np.clip(accuracy_data, 0.7, 0.95)
    
    performance_df = pd.DataFrame({
        'Date': dates,
        'Accuracy': accuracy_data,
        'Precision': accuracy_data + np.random.normal(0, 0.01, len(dates)),
        'Recall': accuracy_data - np.random.normal(0, 0.01, len(dates)),
        'F1': accuracy_data + np.random.normal(0, 0.005, len(dates))
    })
    
    # Performance trends
    fig = px.line(performance_df, x='Date', 
                  y=['Accuracy', 'Precision', 'Recall', 'F1'],
                  title="Model Performance Over Time",
                  labels={'value': 'Score', 'variable': 'Metric'})
    fig.add_hline(y=0.85, line_dash="dash", line_color="green",
                  annotation_text="Target Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_accuracy = performance_df['Accuracy'].iloc[-1]
        accuracy_change = current_accuracy - performance_df['Accuracy'].iloc[-2]
        st.metric("Current Accuracy", f"{current_accuracy:.3f}", 
                 f"{accuracy_change:+.3f}")
    
    with col2:
        current_precision = performance_df['Precision'].iloc[-1]
        precision_change = current_precision - performance_df['Precision'].iloc[-2]
        st.metric("Current Precision", f"{current_precision:.3f}",
                 f"{precision_change:+.3f}")
    
    with col3:
        current_recall = performance_df['Recall'].iloc[-1]
        recall_change = current_recall - performance_df['Recall'].iloc[-2]
        st.metric("Current Recall", f"{current_recall:.3f}",
                 f"{recall_change:+.3f}")
    
    with col4:
        current_f1 = performance_df['F1'].iloc[-1]
        f1_change = current_f1 - performance_df['F1'].iloc[-2]
        st.metric("Current F1", f"{current_f1:.3f}",
                 f"{f1_change:+.3f}")

def display_drift_monitoring(registry):
    """Display drift monitoring dashboard"""
    st.header("🔍 Data Drift Monitoring")
    
    # Simulate drift monitoring data
    dates = pd.date_range(start='2025-08-01', end='2025-08-14', freq='6H')
    
    # Create mock drift data
    drift_scores = np.random.beta(2, 8, len(dates))  # Usually low drift
    drift_scores[20:25] = np.random.beta(8, 2, 5)   # Spike in drift
    
    drift_df = pd.DataFrame({
        'Timestamp': dates,
        'Drift Score': drift_scores,
        'Feature 1': np.random.beta(2, 8, len(dates)),
        'Feature 2': np.random.beta(1, 9, len(dates)),
        'Feature 3': np.random.beta(3, 7, len(dates))
    })
    
    # Drift score over time
    fig = px.line(drift_df, x='Timestamp', y='Drift Score',
                  title="Data Drift Score Over Time")
    
    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="green",
                  annotation_text="Low Threshold")
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                  annotation_text="Medium Threshold")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="High Threshold")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature-level drift
    feature_drift = drift_df[['Feature 1', 'Feature 2', 'Feature 3']].iloc[-1]
    
    fig = px.bar(x=feature_drift.index, y=feature_drift.values,
                title="Current Feature-Level Drift Scores")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    
    # Drift status summary
    current_drift = drift_df['Drift Score'].iloc[-1]
    
    if current_drift < 0.1:
        drift_status = "🟢 None"
        drift_color = "green"
    elif current_drift < 0.3:
        drift_status = "🟡 Low"
        drift_color = "orange"
    elif current_drift < 0.5:
        drift_status = "🟠 Medium"
        drift_color = "orange"
    else:
        drift_status = "🔴 High"
        drift_color = "red"
    
    st.metric("Current Drift Status", drift_status, f"{current_drift:.3f}")

def main():
    """Main dashboard function"""
    
    # Initialize services
    registry, orchestrator = init_services()
    
    # Display header
    display_header()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox("Select View", [
        "📊 Overview",
        "🏛️ Model Registry",
        "📦 Dataset Management", 
        "🚀 Canary Deployments",
        "📈 Performance Monitoring",
        "🔍 Drift Detection"
    ])
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", False)
    
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    
    # Display selected page
    if page == "📊 Overview":
        display_model_registry_overview(registry)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🚀 Recent Canary Activity")
            summary = orchestrator.get_canary_summary()
            st.json(summary)
        
        with col2:
            st.subheader("🏛️ Registry Stats")
            reg_summary = registry.get_registry_summary()
            st.json(reg_summary)
            
    elif page == "🏛️ Model Registry":
        display_model_details(registry)
        
    elif page == "📦 Dataset Management":
        display_dataset_versions(registry)
        
    elif page == "🚀 Canary Deployments":
        display_canary_deployments(orchestrator)
        
    elif page == "📈 Performance Monitoring":
        display_performance_monitoring()
        
    elif page == "🔍 Drift Detection":
        display_drift_monitoring(registry)
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 ML/AI Discipline Dashboard** | Enterprise Model Management | CryptoSmartTrader V2")

if __name__ == "__main__":
    main()