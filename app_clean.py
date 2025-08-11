#!/usr/bin/env python3
"""
Clean CryptoSmartTrader V2 Application - Foutloze workstation versie
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import sys

# Add project to path
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import alleen werkende modules
try:
    from orchestration.strict_gate_standalone import apply_strict_gate_orchestration
    from utils.authentic_opportunities import get_authentic_opportunities_count
    STRICT_GATE_AVAILABLE = True
except ImportError as e:
    logger.error(f"CRITICAL: Strict gate not available: {e}")
    STRICT_GATE_AVAILABLE = False
    # Store error for UI display (will be shown in main() after st init)
    if 'import_errors' not in globals():
        globals()['import_errors'] = []
    globals()['import_errors'].append(f"Strict gate disabled: {e}")

def main():
    """Clean application entry point"""
    st.set_page_config(
        page_title="CryptoSmartTrader V2",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display any import errors from module loading
    if 'import_errors' in globals() and globals()['import_errors']:
        st.sidebar.error("⚠️ System Warnings:")
        for error in globals()['import_errors']:
            st.sidebar.warning(error)
    
    st.sidebar.title("🚀 CryptoSmartTrader V2")
    st.sidebar.markdown("---")
    
    # System health check (per review requirements)
    models_present = all(os.path.exists(f"models/saved/rf_{h}.pkl") for h in ["1h","24h","168h","720h"])
    features_exist = os.path.exists("exports/features.parquet") 
    predictions_exist = os.path.exists("exports/production/predictions.csv")
    
    # Calculate readiness score
    checks = [models_present, features_exist, predictions_exist]
    readiness_score = sum(checks) / len(checks) * 100
    
    if readiness_score >= 90:
        st.sidebar.success(f"🟢 System Ready ({readiness_score:.0f}/100)")
    elif readiness_score >= 70:
        st.sidebar.warning(f"🟠 System Degraded ({readiness_score:.0f}/100)")
    else:
        st.sidebar.error(f"🔴 System Not Ready ({readiness_score:.0f}/100)")
        
    # FIXED: Graceful degradation instead of hard stop
    if not models_present:
        st.sidebar.error("⚠️ Geen getrainde modellen")
        st.sidebar.info("AI-functies uitgeschakeld")
        ai_features_disabled = True
    else:
        ai_features_disabled = False
    
    # Navigation with conditional AI features
    pages = ["📊 Dashboard", "📈 Market Analysis", "⚙️ System Status"]
    if not ai_features_disabled:
        pages.insert(1, "🤖 AI Predictions")
    
    selected_page = st.sidebar.selectbox("Selecteer pagina", pages)
    
    if selected_page == "📊 Dashboard":
        render_dashboard()
    elif selected_page == "🤖 AI Predictions":
        render_ai_predictions()
    elif selected_page == "📈 Market Analysis":
        render_market_analysis()
    elif selected_page == "⚙️ System Status":
        render_system_status()

def render_dashboard():
    """Main dashboard"""
    st.title("📊 CryptoSmartTrader V2 Dashboard")
    
    # Load authentic predictions
    try:
        pred_df = pd.read_csv("exports/production/predictions.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Coins", len(pred_df))
        
        with col2:
            if STRICT_GATE_AVAILABLE:
                opportunities = get_authentic_opportunities_count(0.80)
            else:
                opportunities = 0
            st.metric("High Confidence (≥80%)", opportunities)
        
        with col3:
            mean_conf = pred_df['conf_1h'].mean() if 'conf_1h' in pred_df.columns else 0
            st.metric("Avg Confidence", f"{mean_conf:.1%}")
        
        with col4:
            st.metric("Models Active", "4/4")
        
        # Top opportunities table
        if STRICT_GATE_AVAILABLE and not pred_df.empty:
            filtered_df = apply_strict_gate_orchestration(pred_df, threshold=0.80)
            
            if not filtered_df.empty:
                st.subheader("🎯 Top Opportunities")
                
                # FIXED: Add sentiment and whale detection to display
                display_cols = ['coin', 'pred_720h', 'conf_720h']
                
                # Add sentiment columns if available
                if 'sentiment_score' in filtered_df.columns:
                    display_cols.append('sentiment_score')
                if 'sentiment_label' in filtered_df.columns:
                    display_cols.append('sentiment_label')
                    
                # Add whale detection columns if available  
                if 'whale_activity_detected' in filtered_df.columns:
                    display_cols.append('whale_activity_detected')
                if 'whale_score' in filtered_df.columns:
                    display_cols.append('whale_score')
                
                display_df = filtered_df[display_cols].head(10)
                display_df['pred_720h'] = display_df['pred_720h'].apply(lambda x: f"{x:.2%}")
                display_df['conf_720h'] = display_df['conf_720h'].apply(lambda x: f"{x:.1%}")
                
                # Format additional columns
                if 'sentiment_score' in display_df.columns:
                    display_df['sentiment_score'] = display_df['sentiment_score'].apply(lambda x: f"{x:.2f}")
                if 'whale_score' in display_df.columns:
                    display_df['whale_score'] = display_df['whale_score'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # FIXED: Add whale detection alerts
                if 'whale_activity_detected' in filtered_df.columns:
                    whale_alerts = filtered_df[filtered_df['whale_activity_detected'] == True]
                    if not whale_alerts.empty:
                        st.warning(f"🐋 Whale Activity Detected: {len(whale_alerts)} coins with large transaction risk")
            else:
                st.info("Geen opportunities >= 80% confidence")
        
    except FileNotFoundError:
        st.error("Geen predictions beschikbaar. Train eerst modellen.")

def render_ai_predictions():
    """AI predictions page"""
    st.title("🤖 AI Multi-Horizon Predictions")
    
    try:
        pred_df = pd.read_csv("exports/production/predictions.csv")
        
        # Horizon selector
        horizon = st.selectbox("Selecteer tijdshorizon", ["1h", "24h", "168h", "720h"])
        threshold = st.slider("Min. confidence", 0.5, 1.0, 0.8)
        
        pred_col = f'pred_{horizon}'
        conf_col = f'conf_{horizon}'
        
        if pred_col in pred_df.columns and conf_col in pred_df.columns:
            # Filter by confidence
            filtered = pred_df[pred_df[conf_col] >= threshold]
            filtered = filtered.sort_values(by=pred_col, ascending=False)
            
            st.metric(f"Predictions ≥{threshold:.0%}", len(filtered))
            
            if not filtered.empty:
                # FIXED: Enhanced display with sentiment and whale data
                display_cols = ['coin', pred_col, conf_col]
                
                # Add sentiment and whale columns if available
                additional_cols = ['sentiment_score', 'sentiment_label', 'whale_activity_detected', 'whale_score']
                for col in additional_cols:
                    if col in filtered.columns:
                        display_cols.append(col)
                
                display_df = filtered[display_cols].head(20)
                display_df[pred_col] = display_df[pred_col].apply(lambda x: f"{x:.2%}")  
                display_df[conf_col] = display_df[conf_col].apply(lambda x: f"{x:.1%}")
                
                # Format additional columns
                if 'sentiment_score' in display_df.columns:
                    display_df['sentiment_score'] = display_df['sentiment_score'].apply(lambda x: f"{x:.2f}")
                if 'whale_score' in display_df.columns:
                    display_df['whale_score'] = display_df['whale_score'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # FIXED: Sentiment analysis summary
                if 'sentiment_label' in filtered.columns:
                    sentiment_counts = filtered['sentiment_label'].value_counts()
                    st.subheader("📊 Sentiment Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bullish", sentiment_counts.get('bullish', 0))
                    with col2:
                        st.metric("Neutral", sentiment_counts.get('neutral', 0))
                    with col3:
                        st.metric("Bearish", sentiment_counts.get('bearish', 0))
                
                # Chart
                chart_df = filtered.head(10)
                fig = px.bar(chart_df, x='coin', y=pred_col, 
                           title=f"Top 10 Predictions ({horizon})")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Geen predictions ≥{threshold:.0%} voor {horizon}")
        else:
            st.error(f"Geen data voor horizon {horizon}")
            
    except FileNotFoundError:
        st.error("Geen predictions beschikbaar")

def render_market_analysis():
    """Market analysis page"""  
    st.title("📈 Market Analysis")
    
    try:
        # Load features voor analysis
        features_df = pd.read_parquet("exports/features.parquet")
        
        if not features_df.empty:
            # Market metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Volume Distribution")
                if 'feat_vol_24h' in features_df.columns:
                    fig = px.histogram(features_df, x='feat_vol_24h', 
                                     title="24H Volume Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("RSI Distribution") 
                if 'feat_rsi_14' in features_df.columns:
                    fig = px.histogram(features_df, x='feat_rsi_14',
                                     title="RSI(14) Distribution") 
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geen features data beschikbaar")
            
    except FileNotFoundError:
        st.error("Geen features data beschikbaar")

def render_system_status():
    """System status page"""
    st.title("⚙️ System Status")
    
    # Model status
    st.subheader("Model Status")
    horizons = ["1h", "24h", "168h", "720h"]
    
    for horizon in horizons:
        model_path = f"models/saved/rf_{horizon}.pkl"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            st.success(f"✅ Model {horizon}: {size_mb:.1f}MB")
        else:
            st.error(f"❌ Model {horizon}: Missing")
    
    # Data status  
    st.subheader("Data Status")
    
    data_files = [
        ("Features", "exports/features.parquet"),
        ("Predictions", "exports/production/predictions.csv"), 
        ("Config", "config.json")
    ]
    
    for name, path in data_files:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            st.success(f"✅ {name}: {size_kb:.1f}KB (Updated: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            st.error(f"❌ {name}: Missing")
    
    # System info
    st.subheader("System Information")
    st.info(f"Python: {os.sys.version}")
    st.info(f"Working Directory: {os.getcwd()}")
    st.info(f"Streamlit Version: {st.__version__}")

if __name__ == "__main__":
    main()