#!/usr/bin/env python3
"""
Trading Recommendations Ledger Dashboard

Real-time monitoring van alle trading aanbevelingen met:
- Live recommendation feed
- Performance analytics 
- Signal attribution analysis
- Training data export
- Historical performance trends
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Voeg src toe aan Python path
sys.path.append('src')

from cryptosmarttrader.trading.recommendation_ledger import (
    RecommendationLedger, get_recommendation_ledger, TradingSide, ExitReason
)

# Streamlit configuratie
st.set_page_config(
    page_title="Trading Recommendations Ledger",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_ledger_data():
    """Load recommendation ledger data"""
    try:
        ledger = get_recommendation_ledger()
        
        # Probeer verschillende ledger files
        ledger_files = [
            "data/recommendations_ledger.json",
            "test_data/simple_test_recommendations.json",
            "test_data/test_recommendations.json"
        ]
        
        for file_path in ledger_files:
            if Path(file_path).exists():
                ledger = RecommendationLedger(file_path)
                break
        
        return ledger
    except Exception as e:
        st.error(f"Error loading ledger: {e}")
        return None

def display_recommendation_metrics(ledger):
    """Display key recommendation metrics"""
    
    analytics = ledger.get_performance_analytics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Recommendations",
            analytics.get('total_recommendations', 0),
            help="Total aantal gegenereerde aanbevelingen"
        )
    
    with col2:
        st.metric(
            "Active Positions",
            analytics.get('active_recommendations', 0),
            help="Momenteel actieve aanbevelingen"
        )
    
    with col3:
        hit_rate = analytics.get('hit_rate', 0)
        st.metric(
            "Hit Rate",
            f"{hit_rate:.1%}" if hit_rate else "N/A",
            help="Percentage winstgevende trades"
        )
    
    with col4:
        avg_pnl = analytics.get('avg_pnl_bps', 0)
        st.metric(
            "Avg PnL",
            f"{avg_pnl:.0f} bps" if avg_pnl else "N/A",
            delta=f"{avg_pnl:.0f}" if avg_pnl else None,
            help="Gemiddelde PnL per trade in basis points"
        )

def display_active_recommendations(ledger):
    """Display current active recommendations"""
    
    st.subheader("🔄 Active Recommendations")
    
    active_recs = ledger.get_active_recommendations()
    
    if not active_recs:
        st.info("Geen actieve aanbevelingen momenteel")
        return
    
    # Convert to DataFrame voor display
    active_data = []
    for rec in active_recs:
        active_data.append({
            "Symbol": rec.symbol,
            "Side": rec.side.value,
            "Score": f"{rec.signal_scores.combined_score:.3f}",
            "Confidence": f"{rec.signal_scores.confidence:.2f}",
            "Expected Return": f"{rec.expected_return_bps} bps",
            "Risk Budget": f"{rec.risk_budget_bps} bps",
            "Regime": rec.market_regime,
            "Time": rec.ts_signal.strftime("%H:%M:%S"),
            "ID": rec.recommendation_id[:8] + "..."
        })
    
    df = pd.DataFrame(active_data)
    st.dataframe(df, use_container_width=True)

def display_performance_charts(ledger):
    """Display performance visualization charts"""
    
    st.subheader("📈 Performance Analysis")
    
    # Get historical data
    history_df = ledger.get_recommendation_history(days_back=30)
    
    if history_df.empty:
        st.info("Geen historische data beschikbaar voor charts")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PnL Distribution
        completed_df = history_df[history_df['ts_exit'].notna()]
        
        if not completed_df.empty:
            fig = px.histogram(
                completed_df, 
                x='realized_pnl_bps',
                title="PnL Distribution (Completed Trades)",
                labels={'realized_pnl_bps': 'PnL (basis points)', 'count': 'Number of Trades'},
                nbins=20
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geen voltooide trades voor PnL distributie")
    
    with col2:
        # Signal Performance Attribution
        analytics = ledger.get_performance_analytics()
        signal_perf = analytics.get('signal_performance', {})
        
        if signal_perf and any(signal_perf.values()):
            signals = list(signal_perf.keys())
            values = [signal_perf.get(s, 0) for s in signals]
            
            fig = go.Figure(data=[
                go.Bar(x=signals, y=values, name="Signal Performance")
            ])
            fig.update_layout(
                title="Signal Attribution Analysis",
                xaxis_title="Signal Type",
                yaxis_title="Performance Correlation",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geen signal attribution data beschikbaar")

def display_recent_trades(ledger):
    """Display recent completed trades"""
    
    st.subheader("🎯 Recent Completed Trades")
    
    history_df = ledger.get_recommendation_history(days_back=7)
    completed_df = history_df[history_df['ts_exit'].notna()]
    
    if completed_df.empty:
        st.info("Geen recent voltooide trades")
        return
    
    # Sort by exit time
    completed_df['ts_exit'] = pd.to_datetime(completed_df['ts_exit'])
    completed_df = completed_df.sort_values('ts_exit', ascending=False)
    
    # Display recent trades
    recent_data = []
    for _, row in completed_df.head(10).iterrows():
        pnl_bps = row.get('realized_pnl_bps', 0)
        color = "🟢" if pnl_bps > 0 else "🔴"
        
        recent_data.append({
            "": color,
            "Symbol": row['symbol'],
            "Side": row['side'],
            "PnL (bps)": f"{pnl_bps:.0f}",
            "Holding (hrs)": f"{row.get('holding_period_hours', 0):.1f}",
            "Exit Reason": row.get('reason_exit', 'N/A'),
            "Exit Time": row['ts_exit'].strftime("%m/%d %H:%M"),
            "ID": str(row['recommendation_id'])[:8] + "..."
        })
    
    df = pd.DataFrame(recent_data)
    st.dataframe(df, use_container_width=True)

def display_signal_breakdown(ledger):
    """Display detailed signal score breakdown"""
    
    st.subheader("🔍 Signal Score Analysis")
    
    active_recs = ledger.get_active_recommendations()
    
    if not active_recs:
        st.info("Geen actieve aanbevelingen voor signal analyse")
        return
    
    # Create signal breakdown data
    signal_data = []
    for rec in active_recs:
        scores = rec.signal_scores
        signal_data.append({
            "Symbol": rec.symbol,
            "Momentum": scores.momentum_score,
            "Mean Revert": scores.mean_revert_score,
            "Funding": scores.funding_score,
            "Sentiment": scores.sentiment_score,
            "Whale": scores.whale_score,
            "Technical": scores.technical_score,
            "Combined": scores.combined_score
        })
    
    df = pd.DataFrame(signal_data)
    
    # Heatmap van signal scores
    fig = px.imshow(
        df.set_index('Symbol').T,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Signal Score Heatmap per Symbol"
    )
    fig.update_layout(
        xaxis_title="Symbols",
        yaxis_title="Signal Types"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_training_data_export(ledger):
    """Display training data export functionality"""
    
    st.subheader("🎓 ML Training Data Export")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Exporteer data voor machine learning model training:")
        
        # Parameters voor label generatie
        lookforward_hours = st.slider("Lookforward Hours", 1, 168, 24)
        return_threshold_bps = st.slider("Return Threshold (bps)", 10, 200, 50)
    
    with col2:
        if st.button("Generate Training Labels", type="primary"):
            try:
                labels_df = ledger.generate_training_labels(
                    lookforward_hours=lookforward_hours,
                    return_threshold_bps=return_threshold_bps
                )
                
                if not labels_df.empty:
                    st.success(f"Generated {len(labels_df)} training samples")
                    
                    # Download button
                    csv = labels_df.to_csv(index=False)
                    st.download_button(
                        label="Download Training Data CSV",
                        data=csv,
                        file_name=f"trading_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Preview
                    st.write("Preview:")
                    st.dataframe(labels_df.head(), use_container_width=True)
                else:
                    st.warning("Geen training data beschikbaar")
                    
            except Exception as e:
                st.error(f"Error generating training data: {e}")

def main():
    """Main dashboard application"""
    
    st.title("📋 Trading Recommendations Ledger")
    st.markdown("Real-time monitoring van alle trading aanbevelingen en performance analytics")
    
    # Load ledger data
    ledger = load_ledger_data()
    
    if not ledger:
        st.error("Kan recommendation ledger niet laden")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (10s)", value=False)
    if auto_refresh:
        st.rerun()
    
    refresh_button = st.sidebar.button("🔄 Manual Refresh")
    if refresh_button:
        st.rerun()
    
    days_filter = st.sidebar.slider("Historical Days", 1, 30, 7)
    
    # Main dashboard content
    display_recommendation_metrics(ledger)
    
    st.markdown("---")
    
    # Tabs voor verschillende views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Active", "📈 Performance", "🎯 Recent Trades", 
        "🔍 Signals", "🎓 ML Export"
    ])
    
    with tab1:
        display_active_recommendations(ledger)
    
    with tab2:
        display_performance_charts(ledger)
    
    with tab3:
        display_recent_trades(ledger)
    
    with tab4:
        display_signal_breakdown(ledger)
    
    with tab5:
        display_training_data_export(ledger)
    
    # Footer info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ledger.ledger_path.exists():
            file_size = ledger.ledger_path.stat().st_size
            st.caption(f"📁 Ledger file: {file_size} bytes")
    
    with col2:
        analytics = ledger.get_performance_analytics()
        total_recs = analytics.get('total_recommendations', 0)
        st.caption(f"📊 Total records: {total_recs}")
    
    with col3:
        st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()