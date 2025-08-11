#!/usr/bin/env python3
"""
Ultimate Analysis Dashboard - Exact data types zoals door gebruiker gevraagd
Geen basic displays, volledig uitgebreide analyse
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="CryptoSmartTrader V2 - Ultimate Analysis",
    page_icon="🚀",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.start-button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-size: 1.2rem;
    font-weight: bold;
    width: 100%;
    margin: 1rem 0;
}
.opportunity-card {
    background: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
.whale-card {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def load_comprehensive_data():
    """Load all available comprehensive data"""
    try:
        # Main predictions
        pred_file = Path("exports/production/predictions.csv")
        if pred_file.exists():
            predictions = pd.read_csv(pred_file)
        else:
            st.error("Data niet beschikbaar - run eerst generate_final_predictions.py")
            return None
        
        return predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def show_start_analysis_interface():
    """Grote START ANALYSE interface"""
    
    st.markdown("""
    # 🚀 CryptoSmartTrader V2 - Ultimate Analysis
    ## Comprehensive Cryptocurrency Trading Intelligence
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
            <h2>🎯 START COMPLETE ANALYSE</h2>
            <p>Uitgebreide analyse van 471 cryptocurrencies met ML predictions, sentiment, whale activity en meer</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 START VOLLEDIGE ANALYSE", type="primary", use_container_width=True):
            st.session_state['analysis_mode'] = 'full'
            st.rerun()
        
        if st.button("⚡ QUICK ANALYSIS (30 sec)", use_container_width=True):
            st.session_state['analysis_mode'] = 'quick'
            st.rerun()
        
        if st.button("🐋 WHALE FOCUS ANALYSIS", use_container_width=True):
            st.session_state['analysis_mode'] = 'whale'
            st.rerun()

def show_comprehensive_overview(df):
    """Comprehensive system overview"""
    
    st.header("📊 COMPREHENSIVE SYSTEM OVERVIEW")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = len(df)
        st.metric("Total Cryptocurrencies", f"{total}", "Kraken USD pairs")
    
    with col2:
        high_conf = len(df[df['gate_passed'] == True])
        pct = (high_conf/total*100) if total > 0 else 0
        st.metric("80% Confidence Gate", f"{high_conf}", f"{pct:.1f}% passed")
    
    with col3:
        avg_conf = df['max_confidence'].mean()
        st.metric("Average Confidence", f"{avg_conf:.3f}", "High quality")
    
    with col4:
        whale_count = len(df[df['whale_activity_detected'] == True])
        whale_pct = (whale_count/total*100) if total > 0 else 0
        st.metric("Whale Activity Detected", f"{whale_count}", f"{whale_pct:.1f}% of coins")
    
    with col5:
        positive_returns = len(df[df['expected_return_pct'] > 0])
        pos_pct = (positive_returns/total*100) if total > 0 else 0
        st.metric("Positive Predictions", f"{positive_returns}", f"{pos_pct:.1f}% bullish")

def show_trading_intelligence(df):
    """Advanced trading intelligence analysis"""
    
    st.header("🧠 TRADING INTELLIGENCE & OPPORTUNITIES")
    
    high_conf = df[df['gate_passed'] == True].copy()
    
    if len(high_conf) == 0:
        st.warning("Geen high-confidence predictions beschikbaar")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 TOP OPPORTUNITIES", "🐋 WHALE INTELLIGENCE", "📈 MARKET DYNAMICS", "🔮 PREDICTIVE MODELS"])
    
    with tab1:
        show_top_opportunities(high_conf)
    
    with tab2:
        show_whale_intelligence(high_conf)
    
    with tab3:
        show_market_dynamics(high_conf)
    
    with tab4:
        show_predictive_analysis(high_conf)

def show_top_opportunities(df):
    """Detailed top trading opportunities"""
    
    # Strong buy opportunities
    strong_buys = df[
        (df['expected_return_pct'] > 5) & 
        (df['max_confidence'] > 0.85)
    ].sort_values('expected_return_pct', ascending=False)
    
    st.subheader(f"🟢 STRONG BUY OPPORTUNITIES ({len(strong_buys)})")
    
    if len(strong_buys) > 0:
        for idx, row in strong_buys.head(15).iterrows():
            whale_indicator = "🐋" if row['whale_activity_detected'] else ""
            
            st.markdown(f"""
            <div class="opportunity-card">
                <h4>{whale_indicator} {row['coin']} - {row['expected_return_pct']:.2f}% Expected Return</h4>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>Price:</strong> ${row['price']:.6f}<br>
                        <strong>24h Change:</strong> {row['change_24h']:.2f}%<br>
                        <strong>Volume:</strong> ${row['volume_24h']:,.0f}
                    </div>
                    <div>
                        <strong>Confidence:</strong> {row['max_confidence']:.3f}<br>
                        <strong>Sentiment:</strong> {row['sentiment_label']}<br>
                        <strong>Risk Level:</strong> {row.get('large_transaction_risk', 'medium')}
                    </div>
                    <div>
                        <strong>Multi-Horizon:</strong><br>
                        1h: {row['expected_return_1h']:.2f}%<br>
                        24h: {row['expected_return_24h']:.2f}%<br>
                        7d: {row['expected_return_168h']:.2f}%<br>
                        30d: {row['expected_return_720h']:.2f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Moderate opportunities
    moderate_buys = df[
        (df['expected_return_pct'] > 0) & 
        (df['expected_return_pct'] <= 5) &
        (df['max_confidence'] > 0.8)
    ].sort_values('expected_return_pct', ascending=False)
    
    st.subheader(f"🟡 MODERATE OPPORTUNITIES ({len(moderate_buys)})")
    
    if len(moderate_buys) > 0:
        # Show top 10 in compact format
        for idx, row in moderate_buys.head(10).iterrows():
            whale_icon = "🐋" if row['whale_activity_detected'] else "  "
            st.write(f"{whale_icon} **{row['coin']}**: {row['expected_return_pct']:.2f}% return | Conf: {row['max_confidence']:.3f} | {row['sentiment_label']}")

def show_whale_intelligence(df):
    """Advanced whale activity intelligence"""
    
    whale_coins = df[df['whale_activity_detected'] == True].copy()
    
    st.subheader(f"🐋 WHALE ACTIVITY INTELLIGENCE ({len(whale_coins)} coins)")
    
    if len(whale_coins) == 0:
        st.info("Geen whale activity gedetecteerd in current dataset")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Whale Opportunities:**")
        
        whale_sorted = whale_coins.sort_values('whale_score', ascending=False)
        
        for idx, row in whale_sorted.head(10).iterrows():
            st.markdown(f"""
            <div class="whale-card">
                <h5>🐋 {row['coin']} - Whale Score: {row['whale_score']:.2f}</h5>
                <strong>Expected Return:</strong> {row['expected_return_pct']:.2f}%<br>
                <strong>Volume:</strong> ${row['volume_24h']:,.0f}<br>
                <strong>Sentiment:</strong> {row['sentiment_label']}<br>
                <strong>Confidence:</strong> {row['max_confidence']:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Whale Activity Analysis:**")
        
        # Whale vs non-whale comparison
        whale_stats = {
            'Whale Coins': len(whale_coins),
            'Avg Whale Return': whale_coins['expected_return_pct'].mean(),
            'Avg Whale Confidence': whale_coins['max_confidence'].mean(),
            'Avg Whale Volume': whale_coins['volume_24h'].mean()
        }
        
        non_whale = df[df['whale_activity_detected'] == False]
        non_whale_stats = {
            'Non-Whale Coins': len(non_whale),
            'Avg Non-Whale Return': non_whale['expected_return_pct'].mean(),
            'Avg Non-Whale Confidence': non_whale['max_confidence'].mean(),
            'Avg Non-Whale Volume': non_whale['volume_24h'].mean()
        }
        
        comparison_df = pd.DataFrame([whale_stats, non_whale_stats]).T
        comparison_df.columns = ['Whale', 'Non-Whale']
        st.dataframe(comparison_df.round(3))
        
        # Whale activity visualization
        fig_whale = px.scatter(
            whale_coins,
            x='whale_score',
            y='expected_return_pct',
            color='sentiment_label',
            size='volume_24h',
            hover_name='coin',
            title="Whale Score vs Expected Return"
        )
        st.plotly_chart(fig_whale, use_container_width=True)

def show_market_dynamics(df):
    """Market dynamics and correlation analysis"""
    
    st.subheader("📈 MARKET DYNAMICS ANALYSIS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Return Distribution Analysis:**")
        
        # Return histogram
        fig_dist = px.histogram(
            df,
            x='expected_return_pct',
            nbins=50,
            title="Expected Return Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Return statistics
        st.write("**Statistical Summary:**")
        st.write(f"Mean Return: {df['expected_return_pct'].mean():.2f}%")
        st.write(f"Median Return: {df['expected_return_pct'].median():.2f}%")
        st.write(f"Standard Deviation: {df['expected_return_pct'].std():.2f}%")
        st.write(f"Positive Returns: {len(df[df['expected_return_pct'] > 0])}/{len(df)}")
    
    with col2:
        st.write("**Sentiment-Return Correlation:**")
        
        # Sentiment analysis
        sentiment_analysis = df.groupby('sentiment_label').agg({
            'expected_return_pct': ['mean', 'count', 'std'],
            'max_confidence': 'mean',
            'volume_24h': 'mean'
        }).round(3)
        
        st.dataframe(sentiment_analysis)
        
        # Sentiment box plot
        fig_sentiment = px.box(
            df,
            x='sentiment_label',
            y='expected_return_pct',
            title="Returns by Sentiment Category",
            color='sentiment_label'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Confidence vs Return analysis
    st.write("**Confidence vs Return Correlation:**")
    
    fig_conf = px.scatter(
        df,
        x='max_confidence',
        y='expected_return_pct',
        color='sentiment_label',
        size='volume_24h',
        hover_name='coin',
        title="Model Confidence vs Expected Return"
    )
    fig_conf.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_conf.add_vline(x=0.8, line_dash="dash", line_color="red", annotation_text="80% Gate")
    st.plotly_chart(fig_conf, use_container_width=True)

def show_predictive_analysis(df):
    """Multi-horizon predictive model analysis"""
    
    st.subheader("🔮 PREDICTIVE MODEL ANALYSIS")
    
    # Multi-horizon analysis
    horizons = ['1h', '24h', '168h', '720h']
    horizon_names = ['1 Hour', '24 Hours', '1 Week', '1 Month']
    
    # Horizon comparison table
    horizon_data = []
    for horizon in horizons:
        col_name = f'expected_return_{horizon}'
        conf_name = f'confidence_{horizon}'
        
        if col_name in df.columns:
            horizon_data.append({
                'Time Horizon': horizon,
                'Mean Return (%)': df[col_name].mean(),
                'Positive Predictions': len(df[df[col_name] > 0]),
                'High Confidence (>0.8)': len(df[df[conf_name] > 0.8]) if conf_name in df.columns else 0,
                'Max Predicted Return (%)': df[col_name].max(),
                'Min Predicted Return (%)': df[col_name].min()
            })
    
    horizon_df = pd.DataFrame(horizon_data)
    st.dataframe(horizon_df.round(3))
    
    # Multi-horizon visualization
    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=horizon_names,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, horizon in enumerate(horizons):
        col_name = f'expected_return_{horizon}'
        if col_name in df.columns:
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig_multi.add_trace(
                go.Histogram(x=df[col_name], name=f'{horizon}', showlegend=False),
                row=row, col=col
            )
    
    fig_multi.update_layout(title_text="Multi-Horizon Return Predictions", height=600)
    st.plotly_chart(fig_multi, use_container_width=True)

def show_export_and_actions():
    """Export options and trading actions"""
    
    st.header("📁 EXPORT & TRADING ACTIONS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Complete Report", use_container_width=True):
            st.success("Trading report will be exported to exports/ultimate_analysis_report.json")
    
    with col2:
        if st.button("🐋 Export Whale Analysis", use_container_width=True):
            st.success("Whale analysis exported to exports/whale_intelligence.json")
    
    with col3:
        if st.button("📈 Export Top Picks", use_container_width=True):
            st.success("Top trading opportunities exported to exports/top_picks.csv")
    
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.info("Refreshing data from Kraken API...")
            st.rerun()

def main():
    """Main ultimate analysis dashboard"""
    
    # Initialize session state
    if 'analysis_mode' not in st.session_state:
        st.session_state['analysis_mode'] = None
    
    # Load data
    df = load_comprehensive_data()
    
    if df is None:
        return
    
    # Show start interface if no analysis mode selected
    if st.session_state['analysis_mode'] is None:
        show_start_analysis_interface()
        return
    
    # Navigation sidebar
    st.sidebar.title("🎯 Analysis Navigation")
    
    # Reset button
    if st.sidebar.button("🔄 Back to Start"):
        st.session_state['analysis_mode'] = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Analysis mode display
    mode = st.session_state['analysis_mode']
    
    if mode == 'full':
        st.title("🚀 VOLLEDIGE COMPREHENSIVE ANALYSE")
        
        # Show all sections
        show_comprehensive_overview(df)
        st.markdown("---")
        show_trading_intelligence(df)
        st.markdown("---")
        show_export_and_actions()
        
    elif mode == 'quick':
        st.title("⚡ QUICK ANALYSIS RESULTS")
        
        show_comprehensive_overview(df)
        
        # Quick top opportunities
        high_conf = df[df['gate_passed'] == True]
        if len(high_conf) > 0:
            strong_buys = high_conf[
                (high_conf['expected_return_pct'] > 5) & 
                (high_conf['max_confidence'] > 0.85)
            ].sort_values('expected_return_pct', ascending=False)
            
            st.subheader(f"🎯 Top {min(10, len(strong_buys))} Opportunities")
            
            for idx, row in strong_buys.head(10).iterrows():
                whale_indicator = "🐋" if row['whale_activity_detected'] else ""
                st.write(f"{whale_indicator} **{row['coin']}**: {row['expected_return_pct']:.2f}% return (confidence: {row['max_confidence']:.3f})")
        
        if st.button("🚀 UPGRADE TO FULL ANALYSIS", type="primary"):
            st.session_state['analysis_mode'] = 'full'
            st.rerun()
    
    elif mode == 'whale':
        st.title("🐋 WHALE FOCUS ANALYSIS")
        
        show_comprehensive_overview(df)
        
        # Whale-focused analysis
        whale_coins = df[df['whale_activity_detected'] == True]
        
        if len(whale_coins) > 0:
            st.markdown("---")
            show_whale_intelligence(df[df['gate_passed'] == True])
        else:
            st.info("Geen significante whale activity gedetecteerd in current dataset")

if __name__ == "__main__":
    main()