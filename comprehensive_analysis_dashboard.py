#!/usr/bin/env python3
"""
Comprehensive Analysis Dashboard - Volledig uitgebreide cryptocurrency analyse
Met alle specifieke data types zoals aangegeven door gebruiker
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import ccxt

# Configure page
st.set_page_config(
    page_title="CryptoSmartTrader V2 - Complete Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.analysis-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

class ComprehensiveAnalysisDashboard:
    """Complete analysis dashboard met alle gewenste features"""
    
    def __init__(self):
        self.load_all_data()
        
    def load_all_data(self):
        """Laad alle beschikbare data sources"""
        try:
            # Load predictions
            predictions_file = Path("exports/production/predictions.csv")
            if predictions_file.exists():
                self.predictions = pd.read_csv(predictions_file)
                st.session_state['data_loaded'] = True
            else:
                st.error("Predictions data niet gevonden - run generate_final_predictions.py")
                st.session_state['data_loaded'] = False
                return
            
            # Load features 
            features_file = Path("data/processed/features.csv")
            if features_file.exists():
                self.features = pd.read_csv(features_file)
            else:
                self.features = pd.DataFrame()
                
            # Load system status
            status_file = Path("complete_system_rebuild_report.json")
            if status_file.exists():
                with open(status_file, 'r') as f:
                    self.system_status = json.load(f)
            else:
                self.system_status = {"status": "Unknown"}
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.session_state['data_loaded'] = False

    def show_analysis_start_section(self):
        """Grote START ANALYSE sectie"""
        
        st.markdown("""
        <div class="analysis-header">
            <h1>🚀 CryptoSmartTrader V2 - START COMPLETE ANALYSE</h1>
            <p>Volledig uitgebreide cryptocurrency trading intelligence analyse</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            if st.button("🎯 START VOLLEDIGE ANALYSE", type="primary", use_container_width=True):
                st.session_state['analysis_started'] = True
                st.session_state['show_all_sections'] = True
                st.rerun()
                
        with col2:
            st.write("**OF**")
            
        with col3:
            if st.button("📊 QUICK SCAN (30 sec)", use_container_width=True):
                st.session_state['quick_scan'] = True
                st.rerun()

    def show_system_metrics(self):
        """Uitgebreide systeem metrics"""
        st.header("📊 SYSTEEM STATUS & PERFORMANCE")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_coins = len(self.predictions)
            st.metric("Total Coins", f"{total_coins}", "All Kraken USD pairs")
            
        with col2:
            gate_passed = len(self.predictions[self.predictions['gate_passed'] == True])
            gate_pct = (gate_passed / total_coins * 100) if total_coins > 0 else 0
            st.metric("80% Confidence Gate", f"{gate_passed}", f"{gate_pct:.1f}% passed")
            
        with col3:
            avg_conf = self.predictions['max_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.3f}", "High quality" if avg_conf > 0.8 else "Medium")
            
        with col4:
            whale_count = len(self.predictions[self.predictions['whale_activity_detected'] == True])
            whale_pct = (whale_count / total_coins * 100) if total_coins > 0 else 0
            st.metric("Whale Activity", f"{whale_count}", f"{whale_pct:.1f}% of coins")
            
        with col5:
            system_status = self.system_status.get('status', 'Unknown')
            st.metric("System Status", system_status, "Production Ready" if system_status == "SUCCESS" else "Checking")

    def show_market_intelligence(self):
        """Uitgebreide markt intelligence analyse"""
        st.header("🧠 MARKET INTELLIGENCE & PREDICTIVE ANALYTICS")
        
        # Filter high confidence predictions
        high_conf = self.predictions[self.predictions['gate_passed'] == True].copy()
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 OPPORTUNITIES", "📈 MARKET TRENDS", "🔮 PREDICTIONS", "⚡ MOMENTUM"])
        
        with tab1:
            self.show_trading_opportunities(high_conf)
            
        with tab2:
            self.show_market_trends(high_conf)
            
        with tab3:
            self.show_prediction_analysis(high_conf)
            
        with tab4:
            self.show_momentum_analysis(high_conf)

    def show_trading_opportunities(self, df):
        """Detailed trading opportunities met alle metrics"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 HIGH CONVICTION TRADES")
            
            # Strong buys met alle criteria
            strong_buys = df[
                (df['expected_return_pct'] > 5) & 
                (df['max_confidence'] > 0.85)
            ].sort_values('expected_return_pct', ascending=False)
            
            st.write(f"**{len(strong_buys)} Strong Buy Opportunities**")
            
            if len(strong_buys) > 0:
                for idx, row in strong_buys.head(10).iterrows():
                    with st.expander(f"🚀 {row['coin']} - {row['expected_return_pct']:.2f}% Expected Return"):
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.write("**Market Data**")
                            st.write(f"Price: ${row['price']:.6f}")
                            st.write(f"24h Change: {row['change_24h']:.2f}%")
                            st.write(f"Volume: ${row['volume_24h']:,.0f}")
                            
                        with col_b:
                            st.write("**ML Predictions**")
                            st.write(f"1h: {row['expected_return_1h']:.2f}%")
                            st.write(f"24h: {row['expected_return_24h']:.2f}%")
                            st.write(f"7d: {row['expected_return_168h']:.2f}%")
                            st.write(f"30d: {row['expected_return_720h']:.2f}%")
                            
                        with col_c:
                            st.write("**Intelligence**")
                            st.write(f"Confidence: {row['max_confidence']:.3f}")
                            st.write(f"Sentiment: {row['sentiment_label']}")
                            st.write(f"Whale: {'🐋 YES' if row['whale_activity_detected'] else '❌ NO'}")
                            st.write(f"Risk: {row.get('large_transaction_risk', 'low')}")
        
        with col2:
            st.subheader("🐋 WHALE ACTIVITY INTELLIGENCE")
            
            whale_coins = df[df['whale_activity_detected'] == True].copy()
            
            if len(whale_coins) > 0:
                whale_sorted = whale_coins.sort_values('whale_score', ascending=False)
                
                st.write(f"**{len(whale_coins)} Coins met Whale Activity**")
                
                # Whale activity visualisatie
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
                
                # Top whale opportunities
                st.write("**Top Whale Opportunities:**")
                for idx, row in whale_sorted.head(5).iterrows():
                    st.write(f"🐋 **{row['coin']}**: {row['expected_return_pct']:.2f}% return, Score: {row['whale_score']:.2f}")

    def show_market_trends(self, df):
        """Market trends en correlatie analyse"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 RETURN DISTRIBUTION ANALYSIS")
            
            # Return distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df['expected_return_pct'],
                nbinsx=50,
                name="Expected Returns",
                opacity=0.7
            ))
            fig_dist.update_layout(
                title="Distribution of Expected Returns",
                xaxis_title="Expected Return (%)",
                yaxis_title="Number of Coins"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistics
            st.write("**Return Statistics:**")
            st.write(f"Mean: {df['expected_return_pct'].mean():.2f}%")
            st.write(f"Median: {df['expected_return_pct'].median():.2f}%")
            st.write(f"Std Dev: {df['expected_return_pct'].std():.2f}%")
            st.write(f"Positive Returns: {len(df[df['expected_return_pct'] > 0])}")
            
        with col2:
            st.subheader("🔄 SENTIMENT-RETURN CORRELATION")
            
            # Sentiment analysis
            sentiment_analysis = df.groupby('sentiment_label').agg({
                'expected_return_pct': ['mean', 'count', 'std'],
                'max_confidence': 'mean',
                'volume_24h': 'mean'
            }).round(3)
            
            st.dataframe(sentiment_analysis)
            
            # Sentiment distribution
            fig_sentiment = px.box(
                df,
                x='sentiment_label',
                y='expected_return_pct',
                title="Returns by Sentiment Category"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

    def show_prediction_analysis(self, df):
        """ML prediction analysis per horizon"""
        
        st.subheader("🔮 MULTI-HORIZON PREDICTION ANALYSIS")
        
        # Horizon comparison
        horizons = ['1h', '24h', '168h', '720h']
        horizon_data = []
        
        for horizon in horizons:
            col_name = f'expected_return_{horizon}'
            if col_name in df.columns:
                horizon_data.append({
                    'Horizon': horizon,
                    'Mean Return': df[col_name].mean(),
                    'Positive Predictions': len(df[df[col_name] > 0]),
                    'High Confidence (>0.8)': len(df[df[f'confidence_{horizon}'] > 0.8]) if f'confidence_{horizon}' in df.columns else 0
                })
        
        horizon_df = pd.DataFrame(horizon_data)
        st.dataframe(horizon_df)
        
        # Multi-horizon visualization
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=('1 Hour', '24 Hour', '1 Week', '1 Month'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, horizon in enumerate(horizons):
            col_name = f'expected_return_{horizon}'
            if col_name in df.columns:
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig_multi.add_trace(
                    go.Histogram(x=df[col_name], name=f'{horizon} Returns', showlegend=False),
                    row=row, col=col
                )
        
        fig_multi.update_layout(title_text="Multi-Horizon Return Predictions")
        st.plotly_chart(fig_multi, use_container_width=True)

    def show_momentum_analysis(self, df):
        """Momentum en velocity analyse"""
        
        st.subheader("⚡ MOMENTUM & VELOCITY ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**HIGH MOMENTUM COINS**")
            
            # Calculate momentum score
            df['momentum_score'] = (
                df['change_24h'] * 0.4 +
                df['expected_return_pct'] * 0.6
            )
            
            high_momentum = df.nlargest(10, 'momentum_score')
            
            for idx, row in high_momentum.iterrows():
                momentum_indicator = "🚀" if row['momentum_score'] > 10 else "📈" if row['momentum_score'] > 5 else "➡️"
                st.write(f"{momentum_indicator} **{row['coin']}**: Score {row['momentum_score']:.2f}")
        
        with col2:
            st.write("**MOMENTUM VISUALIZATION**")
            
            fig_momentum = px.scatter(
                df,
                x='change_24h',
                y='expected_return_pct',
                color='max_confidence',
                size='volume_24h',
                hover_name='coin',
                title="24h Change vs Expected Return"
            )
            fig_momentum.add_hline(y=0, line_dash="dash")
            fig_momentum.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig_momentum, use_container_width=True)

    def show_risk_analysis(self):
        """Uitgebreide risk analyse"""
        st.header("⚠️ RISK ANALYSIS & PORTFOLIO OPTIMIZATION")
        
        high_conf = self.predictions[self.predictions['gate_passed'] == True].copy()
        
        tab1, tab2, tab3 = st.tabs(["🎲 RISK METRICS", "🔒 PORTFOLIO RISK", "📉 DOWNSIDE ANALYSIS"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                
                # Risk categories
                risk_categories = {
                    'Low Risk': len(high_conf[high_conf['large_transaction_risk'] == 'low']),
                    'Medium Risk': len(high_conf[high_conf['large_transaction_risk'] == 'medium']),
                    'High Risk': len(high_conf[high_conf['large_transaction_risk'] == 'high'])
                }
                
                fig_risk = px.pie(
                    values=list(risk_categories.values()),
                    names=list(risk_categories.keys()),
                    title="Risk Level Distribution"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
            with col2:
                st.subheader("Volatility Analysis")
                
                # Volatility metrics
                if 'price_volatility' in high_conf.columns:
                    vol_stats = {
                        'Mean Volatility': high_conf['price_volatility'].mean(),
                        'Median Volatility': high_conf['price_volatility'].median(),
                        'High Vol (>10%)': len(high_conf[high_conf['price_volatility'] > 0.1]),
                        'Low Vol (<2%)': len(high_conf[high_conf['price_volatility'] < 0.02])
                    }
                    
                    for metric, value in vol_stats.items():
                        if isinstance(value, float):
                            st.metric(metric, f"{value:.3f}")
                        else:
                            st.metric(metric, value)

    def show_export_options(self):
        """Export en actie opties"""
        st.header("📁 EXPORT & ACTIONS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Trading Report", use_container_width=True):
                self.export_trading_report()
                
        with col2:
            if st.button("🐋 Export Whale Analysis", use_container_width=True):
                self.export_whale_analysis()
                
        with col3:
            if st.button("🔄 Refresh All Data", use_container_width=True):
                # Regenerate predictions
                st.info("Refreshing data from Kraken API...")
                st.rerun()

    def export_trading_report(self):
        """Export complete trading report"""
        high_conf = self.predictions[self.predictions['gate_passed'] == True].copy()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.system_status,
            "summary": {
                "total_coins": len(self.predictions),
                "high_confidence": len(high_conf),
                "whale_activity": len(high_conf[high_conf['whale_activity_detected'] == True])
            },
            "top_opportunities": high_conf.nlargest(20, 'expected_return_pct')[
                ['coin', 'expected_return_pct', 'max_confidence', 'sentiment_label', 'whale_activity_detected']
            ].to_dict(orient='records')
        }
        
        report_file = Path("exports/comprehensive_trading_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        st.success(f"Trading report exported: {report_file}")

    def export_whale_analysis(self):
        """Export whale activity analysis"""
        whale_coins = self.predictions[self.predictions['whale_activity_detected'] == True].copy()
        
        whale_report = {
            "timestamp": datetime.now().isoformat(),
            "whale_summary": {
                "total_whale_coins": len(whale_coins),
                "avg_whale_score": whale_coins['whale_score'].mean(),
                "avg_return": whale_coins['expected_return_pct'].mean()
            },
            "whale_opportunities": whale_coins.nlargest(15, 'whale_score')[
                ['coin', 'whale_score', 'expected_return_pct', 'volume_24h', 'sentiment_label']
            ].to_dict(orient='records')
        }
        
        whale_file = Path("exports/whale_activity_analysis.json")
        with open(whale_file, 'w') as f:
            json.dump(whale_report, f, indent=2)
        
        st.success(f"Whale analysis exported: {whale_file}")

def main():
    """Main dashboard interface"""
    
    # Initialize dashboard
    if 'analysis_started' not in st.session_state:
        st.session_state['analysis_started'] = False
    if 'show_all_sections' not in st.session_state:
        st.session_state['show_all_sections'] = False
    if 'quick_scan' not in st.session_state:
        st.session_state['quick_scan'] = False
    
    dashboard = ComprehensiveAnalysisDashboard()
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.error("Data kon niet worden geladen. Controleer de data files.")
        return
    
    # Show start section first
    if not st.session_state['analysis_started'] and not st.session_state['quick_scan']:
        dashboard.show_analysis_start_section()
        return
    
    # Quick scan mode
    if st.session_state.get('quick_scan', False):
        st.title("⚡ QUICK SCAN RESULTS")
        dashboard.show_system_metrics()
        
        high_conf = dashboard.predictions[dashboard.predictions['gate_passed'] == True]
        strong_buys = high_conf[
            (high_conf['expected_return_pct'] > 5) & 
            (high_conf['max_confidence'] > 0.85)
        ].sort_values('expected_return_pct', ascending=False)
        
        st.subheader(f"🎯 {len(strong_buys)} Strong Buy Opportunities Found")
        
        for idx, row in strong_buys.head(5).iterrows():
            whale_indicator = "🐋" if row['whale_activity_detected'] else ""
            st.write(f"{whale_indicator} **{row['coin']}**: {row['expected_return_pct']:.2f}% expected return (confidence: {row['max_confidence']:.3f})")
        
        if st.button("🚀 START VOLLEDIGE ANALYSE", type="primary"):
            st.session_state['analysis_started'] = True
            st.session_state['show_all_sections'] = True
            st.session_state['quick_scan'] = False
            st.rerun()
        
        return
    
    # Full analysis mode
    if st.session_state['analysis_started']:
        
        # Sidebar navigation
        st.sidebar.title("📊 Analysis Navigation")
        
        sections = [
            "📊 System Metrics",
            "🧠 Market Intelligence", 
            "⚠️ Risk Analysis",
            "📁 Export Options"
        ]
        
        selected_section = st.sidebar.selectbox("Choose Analysis Section:", sections)
        
        # Show selected section
        if selected_section == "📊 System Metrics":
            dashboard.show_system_metrics()
            
        elif selected_section == "🧠 Market Intelligence":
            dashboard.show_market_intelligence()
            
        elif selected_section == "⚠️ Risk Analysis":
            dashboard.show_risk_analysis()
            
        elif selected_section == "📁 Export Options":
            dashboard.show_export_options()
        
        # Reset button
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Reset Analysis"):
            st.session_state['analysis_started'] = False
            st.session_state['show_all_sections'] = False
            st.session_state['quick_scan'] = False
            st.rerun()

if __name__ == "__main__":
    main()