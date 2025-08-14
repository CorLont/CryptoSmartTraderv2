#!/usr/bin/env python3
"""
Enhanced Alpha Motor Dashboard - Real-time coin selection en portfolio monitoring

Features:
- Live alpha generation cycle met 4 signal buckets
- Real-time portfolio construction met Kelly sizing
- Signal attribution analysis per bucket  
- Risk-adjusted ranking en execution quality scoring
- Universe filtering met liquiditeit gates
- Performance monitoring met backtest simulation
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
import json

# Import Alpha Motor components
from src.cryptosmarttrader.alpha.coin_picker_alpha_motor import (
    get_alpha_motor, CoinCandidate, SignalBucket
)
from src.cryptosmarttrader.alpha.enhanced_signal_generators import (
    get_enhanced_signal_generator, generate_sample_market_data,
    TechnicalIndicators, FundingData, SentimentData
)

# Configure page
st.set_page_config(
    page_title="Enhanced Alpha Motor Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_alpha_components():
    """Initialize alpha motor en signal generator"""
    alpha_motor = get_alpha_motor()
    signal_generator = get_enhanced_signal_generator()
    return alpha_motor, signal_generator

def generate_crypto_universe(num_coins=25):
    """Generate realistic crypto universe voor alpha testing"""
    
    # Top crypto symbols
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
        'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'SHIB/USDT',
        'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT',
        'FTM/USDT', 'ALGO/USDT', 'ICP/USDT', 'VET/USDT', 'ETC/USDT',
        'HBAR/USDT', 'FIL/USDT', 'TRON/USDT', 'XLM/USDT', 'MANA/USDT',
        'SAND/USDT', 'CRO/USDT', 'NEAR/USDT', 'APE/USDT', 'LRC/USDT'
    ]
    
    # Generate market data
    market_data = generate_sample_market_data(symbols[:num_coins])
    
    # Convert to Alpha Motor format
    coins = []
    for symbol, data in market_data.items():
        # Calculate realistic market metrics
        base_volume = data['volume_24h_usd']
        
        coin = {
            'symbol': symbol,
            'market_cap_usd': base_volume * np.random.uniform(50, 500),  # Volume multiple
            'volume_24h_usd': base_volume,
            'spread_bps': min(50, max(5, np.random.exponential(15))),    # Realistic spreads
            'depth_1pct_usd': base_volume * np.random.uniform(0.02, 0.08), # 2-8% of volume
            
            # Technical indicators
            'rsi_14': data['indicators'].rsi_14,
            'price_change_24h_pct': np.random.normal(0, 0.05),  # ±5% daily moves
            'volume_7d_avg': base_volume * np.random.uniform(0.7, 1.3),
            
            # Funding data
            'funding_rate_8h_pct': data['funding'].funding_rate_8h * 100,
            'oi_change_24h_pct': data['funding'].oi_change_24h_pct,
            
            # Sentiment data
            'social_mentions_24h': (data['sentiment'].reddit_mentions_24h + 
                                  data['sentiment'].twitter_mentions_24h),
            'sentiment_score': (data['sentiment'].reddit_sentiment + 
                              data['sentiment'].twitter_sentiment) / 2,
        }
        coins.append(coin)
    
    return {'coins': coins}

async def run_alpha_generation():
    """Run complete alpha generation cycle"""
    
    alpha_motor, signal_generator = init_alpha_components()
    
    # Generate market universe
    market_data = generate_crypto_universe(25)
    
    # Run alpha cycle
    positions = await alpha_motor.run_alpha_cycle(market_data)
    
    # Get performance attribution
    attribution = alpha_motor.get_performance_attribution(positions)
    
    return positions, attribution, market_data

def display_header():
    """Display dashboard header"""
    st.title("🎯 Enhanced Alpha Motor Dashboard")
    st.markdown("**Real-time Coin Selection & Portfolio Construction**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Universe Filter", "Active", "Top 25 by Volume")
    with col2:
        st.metric("📊 Signal Buckets", "4", "Multi-Factor")
    with col3:
        st.metric("⚖️ Kelly Sizing", "Enabled", "Vol-Targeting")
    with col4:
        st.metric("🛡️ Risk Controls", "Active", "Execution Gates")

def display_alpha_cycle_results(positions, attribution):
    """Display alpha generation results"""
    
    if not positions:
        st.warning("🚫 No positions generated - check market conditions and risk filters")
        return
    
    st.header("🎯 Alpha Generation Results")
    
    # Portfolio summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_weight = sum(p.final_weight for p in positions)
    avg_alpha = sum(p.total_score for p in positions) / len(positions)
    avg_execution_quality = sum(p.execution_quality for p in positions) / len(positions)
    max_single_weight = max(p.final_weight for p in positions)
    
    with col1:
        st.metric("📈 Total Positions", len(positions))
    with col2:
        st.metric("💰 Portfolio Weight", f"{total_weight:.1%}")
    with col3:
        st.metric("🎯 Average Alpha", f"{avg_alpha:.3f}")
    with col4:
        st.metric("⚡ Execution Quality", f"{avg_execution_quality:.3f}")
    
    # Top positions table
    st.subheader("🏆 Selected Positions")
    
    positions_data = []
    for i, pos in enumerate(positions):
        positions_data.append({
            'Rank': i + 1,
            'Symbol': pos.symbol,
            'Weight': f"{pos.final_weight:.1%}",
            'Alpha Score': f"{pos.total_score:.3f}",
            'Risk-Adjusted': f"{pos.risk_adjusted_score:.3f}",
            'Momentum': f"{pos.momentum_score:.3f}",
            'Mean-Revert': f"{pos.mean_revert_score:.3f}",
            'Funding': f"{pos.funding_score:.3f}",
            'Sentiment': f"{pos.sentiment_score:.3f}",
            'Exec Quality': f"{pos.execution_quality:.3f}",
            'Cluster': pos.correlation_cluster
        })
    
    df = pd.DataFrame(positions_data)
    st.dataframe(df, use_container_width=True)

def display_signal_attribution(attribution, positions):
    """Display signal bucket attribution analysis"""
    
    st.header("📊 Signal Attribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal contribution pie chart
        contrib_data = {
            'Momentum': attribution.get('momentum_contribution', 0),
            'Mean-Revert': attribution.get('mean_revert_contribution', 0),
            'Funding': attribution.get('funding_contribution', 0),
            'Sentiment': attribution.get('sentiment_contribution', 0)
        }
        
        fig = px.pie(
            values=list(contrib_data.values()),
            names=list(contrib_data.keys()),
            title="Portfolio Signal Attribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal strength by position
        if positions:
            signals_df = pd.DataFrame([
                {
                    'Symbol': pos.symbol,
                    'Weight': pos.final_weight,
                    'Momentum': pos.momentum_score,
                    'Mean-Revert': pos.mean_revert_score,
                    'Funding': pos.funding_score,
                    'Sentiment': pos.sentiment_score
                }
                for pos in positions[:10]  # Top 10
            ])
            
            fig = px.bar(
                signals_df, 
                x='Symbol', 
                y=['Momentum', 'Mean-Revert', 'Funding', 'Sentiment'],
                title="Signal Scores by Position",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(barmode='group')
            st.plotly_chart(fig, use_container_width=True)

def display_portfolio_construction(positions):
    """Display portfolio construction analysis"""
    
    if not positions:
        return
        
    st.header("🏗️ Portfolio Construction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Position weights visualization
        weights_data = [(pos.symbol, pos.final_weight) for pos in positions]
        weights_df = pd.DataFrame(weights_data, columns=['Symbol', 'Weight'])
        
        fig = px.bar(
            weights_df, 
            x='Symbol', 
            y='Weight',
            title="Portfolio Weights",
            color='Weight',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics
        st.subheader("⚖️ Risk Metrics")
        
        # Cluster diversification
        clusters = {}
        for pos in positions:
            cluster = pos.correlation_cluster
            clusters[cluster] = clusters.get(cluster, 0) + pos.final_weight
        
        cluster_df = pd.DataFrame([
            {'Cluster': f'Cluster {k}', 'Weight': v}
            for k, v in clusters.items()
        ])
        
        fig = px.pie(
            cluster_df,
            values='Weight',
            names='Cluster',
            title="Cluster Diversification"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk summary
        max_cluster = max(clusters.values()) if clusters else 0
        num_clusters = len(clusters)
        
        st.metric("Max Cluster Weight", f"{max_cluster:.1%}")
        st.metric("Number of Clusters", num_clusters)
        st.metric("Concentration Risk", "Low" if max_cluster < 0.4 else "High")

def display_universe_analysis(market_data):
    """Display universe filtering analysis"""
    
    st.header("🌍 Universe Analysis")
    
    coins = market_data['coins']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume distribution
        volumes = [coin['volume_24h_usd'] for coin in coins]
        
        fig = px.histogram(
            x=volumes,
            nbins=20,
            title="Volume Distribution (24h USD)",
            labels={'x': 'Volume ($)', 'y': 'Count'}
        )
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Spread vs Depth analysis
        spreads = [coin['spread_bps'] for coin in coins]
        depths = [coin['depth_1pct_usd'] for coin in coins]
        symbols = [coin['symbol'] for coin in coins]
        
        fig = px.scatter(
            x=spreads,
            y=depths,
            hover_name=symbols,
            title="Liquidity Profile: Spread vs Depth",
            labels={'x': 'Spread (bps)', 'y': 'Depth at 1% ($)'},
            size=[coin['volume_24h_usd'] for coin in coins],
            size_max=20
        )
        st.plotly_chart(fig, use_container_width=True)

def display_performance_simulation():
    """Display simulated performance analysis"""
    
    st.header("📈 Performance Simulation")
    
    # Generate mock performance data
    dates = pd.date_range(start='2025-01-01', end='2025-08-14', freq='D')
    
    # Simulate alpha motor performance
    np.random.seed(42)  # Reproducible results
    daily_returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily alpha, 2% vol
    cumulative_returns = (1 + pd.Series(daily_returns)).cumprod()
    
    # Benchmark comparison
    benchmark_returns = np.random.normal(0.0005, 0.025, len(dates))  # Market returns
    benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()
    
    perf_df = pd.DataFrame({
        'Date': dates,
        'Alpha Motor': cumulative_returns,
        'Benchmark': benchmark_cumulative,
        'Alpha': cumulative_returns / benchmark_cumulative
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cumulative performance
        fig = px.line(
            perf_df,
            x='Date',
            y=['Alpha Motor', 'Benchmark'],
            title="Cumulative Performance Comparison",
            labels={'value': 'Cumulative Return', 'variable': 'Strategy'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rolling alpha
        fig = px.line(
            perf_df,
            x='Date',
            y='Alpha',
            title="Rolling Alpha vs Benchmark",
            labels={'Alpha': 'Relative Performance'}
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    volatility = pd.Series(daily_returns).std() * np.sqrt(252) * 100
    sharpe = (total_return / 365 * 252) / (volatility) if volatility > 0 else 0
    max_dd = (1 - cumulative_returns / cumulative_returns.cummax()).max() * 100
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Annualized Vol", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_dd:.1f}%")

def main():
    """Main dashboard function"""
    
    # Display header
    display_header()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Alpha Motor Controls")
    
    # Run controls
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", False)
    run_alpha = st.sidebar.button("🚀 Generate Alpha Positions", type="primary")
    
    # Configuration
    st.sidebar.subheader("⚙️ Configuration")
    universe_size = st.sidebar.slider("Universe Size", 10, 50, 25)
    max_positions = st.sidebar.slider("Max Positions", 5, 20, 15)
    
    # Run alpha generation
    if run_alpha or auto_refresh:
        with st.spinner("🔄 Running alpha generation cycle..."):
            try:
                positions, attribution, market_data = asyncio.run(run_alpha_generation())
                
                # Store results in session state
                st.session_state['positions'] = positions
                st.session_state['attribution'] = attribution  
                st.session_state['market_data'] = market_data
                st.session_state['last_update'] = datetime.now()
                
                st.success(f"✅ Alpha cycle completed - {len(positions)} positions generated")
                
            except Exception as e:
                st.error(f"❌ Alpha generation failed: {e}")
                return
    
    # Display results if available
    if 'positions' in st.session_state:
        positions = st.session_state['positions']
        attribution = st.session_state['attribution']
        market_data = st.session_state['market_data']
        last_update = st.session_state['last_update']
        
        st.info(f"📊 Last update: {last_update.strftime('%H:%M:%S')}")
        
        # Main dashboard sections
        display_alpha_cycle_results(positions, attribution)
        display_signal_attribution(attribution, positions)
        display_portfolio_construction(positions)
        display_universe_analysis(market_data)
        display_performance_simulation()
        
    else:
        st.info("👆 Click 'Generate Alpha Positions' to start alpha motor analysis")
        
        # Show sample universe analysis
        sample_data = generate_crypto_universe(15)
        display_universe_analysis(sample_data)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**🎯 Enhanced Alpha Motor Dashboard** | Multi-Factor Coin Selection | CryptoSmartTrader V2")

if __name__ == "__main__":
    main()