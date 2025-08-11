#!/usr/bin/env python3
"""
Market Overview Page - Real-time market data and analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dashboards.utils.cache_manager import cache_manager
from dashboards.utils.session_state import session_manager


def render():
    """Render market overview page"""
    
    st.header("🏪 Market Overview")
    
    # Market controls
    render_market_controls()
    
    # Market metrics summary
    render_market_metrics()
    
    # Main chart
    render_main_chart()
    
    # Market data table
    render_market_table()
    
    # Market analysis
    render_market_analysis()


def render_market_controls():
    """Render market control widgets"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Symbol selection
        available_symbols = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'XRP/USD', 'ADA/USD', 
                           'SOL/USD', 'DOGE/USD', 'DOT/USD', 'MATIC/USD', 'LTC/USD']
        
        selected_symbol = st.selectbox(
            "Primary Symbol",
            available_symbols,
            index=available_symbols.index(session_manager.get('selected_symbol', 'BTC/USD')),
            key="market_symbol_select"
        )
        session_manager.set('selected_symbol', selected_symbol)
    
    with col2:
        # Timeframe selection
        timeframes = ['1h', '4h', '1d', '1w']
        selected_timeframe = st.selectbox(
            "Timeframe",
            timeframes,
            index=timeframes.index(session_manager.get('selected_timeframe', '1h')),
            key="market_timeframe_select"
        )
        session_manager.set('selected_timeframe', selected_timeframe)
    
    with col3:
        # Exchange filter
        exchanges = ['All', 'Kraken', 'Binance', 'Coinbase']
        selected_exchange = st.selectbox(
            "Exchange",
            exchanges,
            key="market_exchange_filter"
        )
    
    with col4:
        # Market cap filter
        market_cap_filter = st.selectbox(
            "Market Cap",
            ['All', 'Large Cap (>$10B)', 'Mid Cap ($1B-$10B)', 'Small Cap (<$1B)'],
            key="market_cap_filter"
        )


def render_market_metrics():
    """Render key market metrics"""
    
    # Get market data
    symbols = session_manager.get('market_overview_symbols', ['BTC/USD', 'ETH/USD', 'BNB/USD'])
    market_data = cache_manager.get_market_data(symbols)
    
    if market_data.empty:
        st.warning("No market data available")
        return
    
    # Calculate market metrics
    total_market_cap = market_data['market_cap'].sum()
    avg_change_24h = market_data['change_24h'].mean()
    total_volume_24h = market_data['volume_24h'].sum()
    
    # Count gainers/losers
    gainers = len(market_data[market_data['change_24h'] > 0])
    losers = len(market_data[market_data['change_24h'] < 0])
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Market Cap",
            f"${total_market_cap/1e9:.2f}B",
            delta=f"{avg_change_24h:+.2f}%"
        )
    
    with col2:
        st.metric(
            "24h Volume",
            f"${total_volume_24h/1e9:.2f}B",
            delta=None
        )
    
    with col3:
        st.metric(
            "Market Trend",
            f"{avg_change_24h:+.2f}%",
            delta=f"{'📈' if avg_change_24h > 0 else '📉'}"
        )
    
    with col4:
        st.metric(
            "Gainers",
            gainers,
            delta=f"vs {losers} losers"
        )
    
    with col5:
        # Market fear & greed index (simulated)
        import random
        fear_greed = random.randint(20, 80)
        fear_greed_label = "Greed" if fear_greed > 50 else "Fear"
        
        st.metric(
            "Fear & Greed",
            f"{fear_greed} ({fear_greed_label})",
            delta=None
        )


def render_main_chart():
    """Render main price chart with technical indicators"""
    
    selected_symbol = session_manager.get('selected_symbol', 'BTC/USD')
    selected_timeframe = session_manager.get('selected_timeframe', '1h')
    
    # Get historical data
    historical_data = cache_manager.get_historical_data(selected_symbol, selected_timeframe)
    
    if historical_data.empty:
        st.warning(f"No historical data available for {selected_symbol}")
        return
    
    # Chart controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"{selected_symbol} Price Chart")
    
    with col2:
        show_volume = st.checkbox("Show Volume", value=True, key="show_volume")
        show_indicators = st.checkbox("Show Indicators", value=True, key="show_indicators")
    
    # Create subplot figure
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{selected_symbol} Price', 'Volume']
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_data['timestamp'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages if indicators enabled
    if show_indicators:
        # Simple moving averages
        historical_data['sma_20'] = historical_data['close'].rolling(window=20).mean()
        historical_data['sma_50'] = historical_data['close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['sma_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Add volume chart
    if show_volume:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(historical_data['close'], historical_data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=historical_data['timestamp'],
                y=historical_data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=session_manager.get('chart_height', 400),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)


def render_market_table():
    """Render market data table"""
    
    st.subheader("💹 Market Data")
    
    # Get market data
    symbols = session_manager.get('market_overview_symbols', ['BTC/USD', 'ETH/USD', 'BNB/USD'])
    market_data = cache_manager.get_market_data(symbols)
    
    if market_data.empty:
        st.warning("No market data available")
        return
    
    # Table controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ['market_cap', 'price', 'change_24h', 'volume_24h'],
            key="table_sort_by"
        )
    
    with col2:
        sort_ascending = st.checkbox("Ascending", value=False, key="table_sort_asc")
    
    with col3:
        show_only_gainers = st.checkbox("Only Gainers", value=False, key="show_only_gainers")
    
    # Filter data
    if show_only_gainers:
        market_data = market_data[market_data['change_24h'] > 0]
    
    # Sort data
    market_data = market_data.sort_values(sort_by, ascending=sort_ascending)
    
    # Format data for display
    display_data = market_data.copy()
    display_data['price'] = display_data['price'].apply(lambda x: f"${x:,.2f}")
    display_data['change_24h'] = display_data['change_24h'].apply(lambda x: f"{x:+.2f}%")
    display_data['volume_24h'] = display_data['volume_24h'].apply(lambda x: f"${x/1e6:.1f}M")
    display_data['market_cap'] = display_data['market_cap'].apply(lambda x: f"${x/1e9:.2f}B")
    
    # Rename columns for display
    display_data = display_data.rename(columns={
        'symbol': 'Symbol',
        'price': 'Price',
        'change_24h': '24h Change',
        'volume_24h': '24h Volume',
        'market_cap': 'Market Cap'
    })
    
    # Display table
    st.dataframe(
        display_data[['Symbol', 'Price', '24h Change', '24h Volume', 'Market Cap']],
        use_container_width=True,
        hide_index=True
    )


def render_market_analysis():
    """Render market analysis section"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Market Distribution")
        
        # Get market data for pie chart
        symbols = session_manager.get('market_overview_symbols', ['BTC/USD', 'ETH/USD', 'BNB/USD'])
        market_data = cache_manager.get_market_data(symbols)
        
        if not market_data.empty:
            # Create market cap distribution pie chart
            fig_pie = px.pie(
                market_data,
                values='market_cap',
                names='symbol',
                title='Market Cap Distribution'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top Movers")
        
        if not market_data.empty:
            # Get top gainers and losers
            top_gainers = market_data.nlargest(3, 'change_24h')[['symbol', 'change_24h']]
            top_losers = market_data.nsmallest(3, 'change_24h')[['symbol', 'change_24h']]
            
            st.markdown("**Top Gainers (24h)**")
            for _, row in top_gainers.iterrows():
                st.markdown(f"🟢 {row['symbol']}: +{row['change_24h']:.2f}%")
            
            st.markdown("**Top Losers (24h)**")
            for _, row in top_losers.iterrows():
                st.markdown(f"🔴 {row['symbol']}: {row['change_24h']:.2f}%")
    
    # Market insights
    st.subheader("🧠 Market Insights")
    
    # Generate insights based on data
    insights = generate_market_insights(market_data)
    
    for insight in insights:
        st.info(insight)


def generate_market_insights(market_data: pd.DataFrame) -> list:
    """Generate market insights based on current data"""
    
    if market_data.empty:
        return ["No market data available for analysis"]
    
    insights = []
    
    # Market trend analysis
    avg_change = market_data['change_24h'].mean()
    if avg_change > 2:
        insights.append("🚀 Strong bullish momentum detected across the market")
    elif avg_change < -2:
        insights.append("🐻 Bearish pressure evident in market movements")
    else:
        insights.append("⚖️ Market showing consolidation with mixed signals")
    
    # Volume analysis
    high_volume_count = len(market_data[market_data['volume_24h'] > market_data['volume_24h'].median() * 1.5])
    if high_volume_count > len(market_data) * 0.5:
        insights.append("📈 High trading activity suggests increased market interest")
    
    # Volatility analysis
    volatility = market_data['change_24h'].std()
    if volatility > 5:
        insights.append("⚡ High market volatility - exercise caution with position sizing")
    elif volatility < 2:
        insights.append("😴 Low volatility environment - potential for breakout moves")
    
    return insights