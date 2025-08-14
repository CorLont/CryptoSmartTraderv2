#!/usr/bin/env python3
"""
Alpha Motor Dashboard - Live coin selection en portfolio construction

Demonstreert de coin-picking alpha motor met:
- Universe filtering results
- Signal scores per bucket  
- Portfolio construction
- Risk-adjusted rankings
- Performance attribution
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime

# Import alpha motor components
from src.cryptosmarttrader.alpha.coin_picker_alpha_motor import get_alpha_motor
from src.cryptosmarttrader.alpha.market_data_simulator import MarketDataSimulator

# Page configuration
st.set_page_config(
    page_title="Alpha Motor - Coin Selection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .signal-score {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .portfolio-weight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem;
        border-radius: 0.3rem;
        color: white;
        text-align: center;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute
def generate_market_data():
    """Generate fresh market data"""
    simulator = MarketDataSimulator()
    return simulator.generate_market_snapshot()

@st.cache_data(ttl=60)
def run_alpha_cycle_cached(market_data_json):
    """Run alpha cycle with caching"""
    market_data = json.loads(market_data_json)
    alpha_motor = get_alpha_motor()
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(alpha_motor.run_alpha_cycle(market_data))
        attribution = alpha_motor.get_performance_attribution(result)
        return result, attribution
    finally:
        loop.close()

def main():
    st.title("🎯 Alpha Motor - Coin Selection & Portfolio Construction")
    st.markdown("**Real-time coin picking met multi-factor signals en risk management**")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Generate New Market Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
            
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)
        if auto_refresh:
            st.rerun()
            
        st.markdown("---")
        st.header("📊 Configuration")
        
        universe_size = st.slider("Universe Size", 20, 100, 50)
        max_positions = st.slider("Max Positions", 5, 20, 15)
        vol_target = st.slider("Vol Target (%)", 10, 30, 20)
        
        st.markdown("---")
        st.header("🎯 Signal Weights")
        momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.30, 0.05)
        mean_revert_weight = st.slider("Mean Revert", 0.0, 1.0, 0.25, 0.05)
        funding_weight = st.slider("Funding/Basis", 0.0, 1.0, 0.25, 0.05)
        sentiment_weight = st.slider("Sentiment", 0.0, 1.0, 0.20, 0.05)

    # Generate market data
    with st.spinner("🔄 Generating market data..."):
        market_data = generate_market_data()

    # Run alpha motor
    with st.spinner("🧠 Running alpha analysis..."):
        market_data_json = json.dumps(market_data)
        positions, attribution = run_alpha_cycle_cached(market_data_json)

    # Display results
    if not positions:
        st.error("❌ No positions generated - check market conditions or risk settings")
        return

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Universe Filtered</h3>
            <div class="signal-score">{len(market_data['coins'])}</div>
            <p>coins analyzed</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Positions Selected</h3>
            <div class="signal-score">{len(positions)}</div>
            <p>final positions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        total_score = sum(p.total_score for p in positions) / len(positions)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Alpha Score</h3>
            <div class="signal-score">{total_score:.3f}</div>
            <p>portfolio average</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        total_weight = sum(p.final_weight for p in positions)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Allocation</h3>
            <div class="signal-score">{total_weight:.1%}</div>
            <p>portfolio weight</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Portfolio", "📊 Signal Analysis", "🔍 Universe", "⚖️ Attribution", "📈 Execution"
    ])

    with tab1:
        st.subheader("🎯 Final Portfolio Positions")
        
        # Portfolio table
        portfolio_data = []
        for pos in positions:
            portfolio_data.append({
                'Symbol': pos.symbol,
                'Weight': f"{pos.final_weight:.2%}",
                'Total Score': f"{pos.total_score:.3f}",
                'Risk Adj Score': f"{pos.risk_adjusted_score:.3f}",
                'Kelly Weight': f"{pos.kelly_weight:.2%}",
                'Market Cap': f"${pos.market_cap_usd:,.0f}",
                'Volume 24h': f"${pos.volume_24h_usd:,.0f}",
                'Spread (bps)': f"{pos.spread_bps:.1f}",
                'Cluster': pos.correlation_cluster
            })
            
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)

        # Portfolio allocation pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=[p.final_weight for p in positions],
                names=[p.symbol for p in positions],
                title="Portfolio Allocation"
            )
            fig_pie.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # Cluster allocation
            cluster_weights = {}
            for pos in positions:
                cluster = f"Cluster {pos.correlation_cluster}"
                cluster_weights[cluster] = cluster_weights.get(cluster, 0) + pos.final_weight
                
            fig_cluster = px.bar(
                x=list(cluster_weights.keys()),
                y=list(cluster_weights.values()),
                title="Allocation by Correlation Cluster",
                labels={'x': 'Cluster', 'y': 'Weight'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    with tab2:
        st.subheader("📊 Signal Bucket Analysis")
        
        # Signal scores heatmap
        signal_data = []
        for pos in positions:
            signal_data.append({
                'Symbol': pos.symbol,
                'Momentum': pos.momentum_score,
                'Mean Revert': pos.mean_revert_score,
                'Funding': pos.funding_score,
                'Sentiment': pos.sentiment_score,
                'Weight': pos.final_weight
            })
            
        df_signals = pd.DataFrame(signal_data)
        
        # Heatmap of signal scores
        fig_heatmap = px.imshow(
            df_signals[['Momentum', 'Mean Revert', 'Funding', 'Sentiment']].T,
            x=df_signals['Symbol'],
            y=['Momentum', 'Mean Revert', 'Funding', 'Sentiment'],
            color_continuous_scale='Viridis',
            title="Signal Scores Heatmap"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Signal contribution scatter
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                df_signals,
                x='Momentum',
                y='Mean Revert',
                size='Weight',
                color='Sentiment',
                hover_name='Symbol',
                title="Momentum vs Mean Reversion"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            fig_funding = px.scatter(
                df_signals,
                x='Funding',
                y='Sentiment',
                size='Weight',
                color='Weight',
                hover_name='Symbol',
                title="Funding vs Sentiment"
            )
            st.plotly_chart(fig_funding, use_container_width=True)

    with tab3:
        st.subheader("🔍 Universe Analysis")
        
        # Universe overview
        universe_data = []
        for coin in market_data['coins'][:20]:  # Top 20 for display
            universe_data.append({
                'Symbol': coin['symbol'],
                'Market Cap': coin['market_cap_usd'],
                'Volume 24h': coin['volume_24h_usd'],
                'Spread (bps)': coin['spread_bps'],
                'Depth 1%': coin['depth_1pct_usd'],
                'Price Change %': coin['price_change_24h_pct'],
                'RSI': coin['rsi_14'],
                'Sentiment': coin['sentiment_score'],
                'Tier': coin['tier']
            })
            
        df_universe = pd.DataFrame(universe_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume vs Market Cap
            fig_volume = px.scatter(
                df_universe,
                x='Market Cap',
                y='Volume 24h',
                color='Tier',
                hover_name='Symbol',
                title="Volume vs Market Cap",
                log_x=True,
                log_y=True
            )
            st.plotly_chart(fig_volume, use_container_width=True)
            
        with col2:
            # Liquidity quality
            fig_liquidity = px.scatter(
                df_universe,
                x='Spread (bps)',
                y='Depth 1%',
                color='Tier',
                size='Volume 24h',
                hover_name='Symbol',
                title="Liquidity Quality",
                log_y=True
            )
            st.plotly_chart(fig_liquidity, use_container_width=True)

    with tab4:
        st.subheader("⚖️ Performance Attribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attribution bar chart
            attr_data = pd.DataFrame([attribution]).T.reset_index()
            attr_data.columns = ['Factor', 'Contribution']
            
            fig_attr = px.bar(
                attr_data,
                x='Factor',
                y='Contribution',
                title="Factor Contribution to Portfolio"
            )
            fig_attr.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_attr, use_container_width=True)
            
        with col2:
            # Risk metrics
            st.markdown("### 📊 Portfolio Risk Metrics")
            
            avg_spread = sum(p.spread_bps * p.final_weight for p in positions)
            weighted_mcap = sum(p.market_cap_usd * p.final_weight for p in positions)
            
            risk_metrics = {
                "Weighted Avg Spread": f"{avg_spread:.1f} bps",
                "Weighted Market Cap": f"${weighted_mcap:,.0f}",
                "Position Count": len(positions),
                "Max Single Weight": f"{max(p.final_weight for p in positions):.1%}",
                "Portfolio Concentration": f"{sum(p.final_weight**2 for p in positions):.3f}"
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, value)

    with tab5:
        st.subheader("📈 Execution Analysis")
        
        # Execution quality metrics
        exec_data = []
        for pos in positions:
            exec_data.append({
                'Symbol': pos.symbol,
                'Weight': pos.final_weight,
                'Spread (bps)': pos.spread_bps,
                'Depth ($)': pos.depth_1pct_usd,
                'Execution Quality': pos.execution_quality,
                'Liquidity Rank': pos.liquidity_rank
            })
            
        df_exec = pd.DataFrame(exec_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Execution quality vs weight
            fig_exec = px.scatter(
                df_exec,
                x='Weight',
                y='Execution Quality',
                size='Depth ($)',
                color='Spread (bps)',
                hover_name='Symbol',
                title="Execution Quality vs Position Size"
            )
            st.plotly_chart(fig_exec, use_container_width=True)
            
        with col2:
            # Estimated transaction costs
            total_portfolio_value = 1_000_000  # $1M portfolio assumption
            
            cost_data = []
            total_cost = 0
            
            for pos in positions:
                position_value = pos.final_weight * total_portfolio_value
                spread_cost = position_value * (pos.spread_bps / 10000)
                total_cost += spread_cost
                
                cost_data.append({
                    'Symbol': pos.symbol,
                    'Position Value': position_value,
                    'Spread Cost': spread_cost,
                    'Cost %': (spread_cost / position_value) * 100
                })
                
            df_costs = pd.DataFrame(cost_data)
            
            st.markdown(f"### 💰 Estimated Transaction Costs")
            st.metric("Total Spread Cost", f"${total_cost:,.0f}")
            st.metric("Portfolio Impact", f"{(total_cost/total_portfolio_value)*100:.2f}%")
            
            # Cost breakdown
            fig_costs = px.bar(
                df_costs,
                x='Symbol',
                y='Spread Cost',
                title="Transaction Costs by Position"
            )
            fig_costs.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_costs, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Alpha Motor v2.0** | **CryptoSmartTrader**")

if __name__ == "__main__":
    main()