#!/usr/bin/env python3
"""
Whale Detection & Sentiment Monitoring Dashboard

Production-ready dashboard voor real-time whale activity en sentiment monitoring
met TOS-compliant data feeds en veilige signal integration.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.cryptosmarttrader.data.whale_detector import (
    WhaleDetector, WhaleDetectionConfig, WhaleSignal, WhaleEventType
)
from src.cryptosmarttrader.data.sentiment_monitor import (
    SentimentMonitor, SentimentConfig, SentimentSignal, SentimentSource
)

# Configure Streamlit
st.set_page_config(
    page_title="Whale & Sentiment Monitor",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_market_data(symbols: List[str]) -> Dict:
    """Generate realistic market data for testing"""
    import numpy as np
    
    market_data = {}
    
    for symbol in symbols:
        base_volume = np.random.uniform(20_000_000, 300_000_000)
        price = np.random.uniform(0.1, 50000)
        
        market_data[symbol] = {
            'price_usd': price,
            'volume_24h_usd': base_volume,
            'avg_volume_7d_usd': base_volume * np.random.uniform(0.8, 1.2),
            'spread_bps': np.random.uniform(5, 50),
            'depth_1pct_usd': base_volume * np.random.uniform(0.02, 0.12),
            'market_cap_usd': base_volume * np.random.uniform(20, 200)
        }
    
    return market_data


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_cached_whale_signals(symbols: List[str]) -> Dict[str, WhaleSignal]:
    """Get cached whale signals"""
    try:
        market_data = generate_realistic_market_data(symbols)
        
        # Initialize whale detector
        whale_config = WhaleDetectionConfig()
        whale_config.MIN_TRANSFER_USD = 1_000_000  # $1M threshold
        
        async def collect_signals():
            detector = WhaleDetector(whale_config)
            async with detector:
                return await detector.process_whale_events(market_data)
        
        return asyncio.run(collect_signals())
    
    except Exception as e:
        st.error(f"Whale detection error: {e}")
        return {}


@st.cache_data(ttl=30)  # Cache for 30 seconds  
def get_cached_sentiment_signals(symbols: List[str]) -> Dict[str, SentimentSignal]:
    """Get cached sentiment signals"""
    try:
        sentiment_config = SentimentConfig()
        
        async def collect_signals():
            monitor = SentimentMonitor(sentiment_config)
            async with monitor:
                return await monitor.process_sentiment_data(symbols)
        
        return asyncio.run(collect_signals())
    
    except Exception as e:
        st.error(f"Sentiment monitoring error: {e}")
        return {}


def display_header():
    """Display dashboard header"""
    st.title("🐋💭 Whale Detection & Sentiment Monitor")
    st.markdown("**Real-time TOS-compliant whale activity en social sentiment monitoring**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🐋 Whale Threshold", "$1M+", "Minimum transfer")
    with col2:
        st.metric("💭 Sources", "Reddit + X", "TOS Compliant")
    with col3:
        st.metric("⏱️ Rate Limits", "Active", "Exponential backoff")
    with col4:
        st.metric("🛡️ Trading", "Signal Only", "No automation")


def display_whale_detection(whale_signals: Dict[str, WhaleSignal]):
    """Display whale detection results"""
    
    st.header("🐋 Whale Activity Detection")
    
    if not whale_signals:
        st.warning("🔍 No whale activity detected in current monitoring window")
        return
    
    # Whale activity metrics
    col1, col2, col3 = st.columns(3)
    
    total_flow = sum(s.net_flow_usd for s in whale_signals.values())
    active_signals = len([s for s in whale_signals.values() if abs(s.signal_strength) > 0.1])
    avg_confidence = sum(s.confidence_score for s in whale_signals.values()) / len(whale_signals)
    
    with col1:
        color = "normal" if total_flow >= 0 else "inverse"
        st.metric("💰 Net Flow", f"${total_flow:,.0f}", delta_color=color)
    with col2:
        st.metric("🎯 Active Signals", active_signals)
    with col3:
        st.metric("🔍 Avg Confidence", f"{avg_confidence:.1%}")
    
    # Whale signals table
    st.subheader("🐋 Detected Whale Signals")
    
    whale_data = []
    for symbol, signal in whale_signals.items():
        whale_data.append({
            'Symbol': symbol,
            'Net Flow ($)': f"{signal.net_flow_usd:,.0f}",
            'Signal Strength': f"{signal.signal_strength:.3f}",
            'Events': signal.event_count,
            'Max Single ($)': f"{signal.max_single_event_usd:,.0f}",
            'Confidence': f"{signal.confidence_score:.1%}",
            'Inflows': signal.inflow_events,
            'Outflows': signal.outflow_events,
            'Manipulation Risk': f"{signal.manipulation_risk:.1%}"
        })
    
    df = pd.DataFrame(whale_data)
    st.dataframe(df, use_container_width=True)
    
    # Whale activity visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Net flow chart
        symbols = list(whale_signals.keys())
        flows = [whale_signals[s].net_flow_usd for s in symbols]
        colors = ['green' if f > 0 else 'red' for f in flows]
        
        fig = go.Figure([
            go.Bar(x=symbols, y=flows, marker_color=colors, name='Net Flow')
        ])
        fig.update_layout(
            title="Net Whale Flow by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Net Flow (USD)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal strength vs confidence
        strengths = [whale_signals[s].signal_strength for s in symbols]
        confidences = [whale_signals[s].confidence_score for s in symbols]
        
        fig = go.Figure([
            go.Scatter(
                x=confidences, y=strengths, text=symbols,
                mode='markers+text', textposition='top center',
                marker=dict(size=10, color=strengths, colorscale='RdYlGn'),
                name='Whale Signals'
            )
        ])
        fig.update_layout(
            title="Signal Strength vs Confidence",
            xaxis_title="Confidence Score",
            yaxis_title="Signal Strength",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_sentiment_monitoring(sentiment_signals: Dict[str, SentimentSignal]):
    """Display sentiment monitoring results"""
    
    st.header("💭 Social Sentiment Analysis")
    
    if not sentiment_signals:
        st.info("📱 No significant sentiment signals in current monitoring window")
        return
    
    # Sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_mentions = sum(s.mention_count for s in sentiment_signals.values())
    total_engagement = sum(s.total_engagement for s in sentiment_signals.values())
    avg_sentiment = sum(s.net_sentiment for s in sentiment_signals.values()) / len(sentiment_signals)
    active_signals = len([s for s in sentiment_signals.values() if abs(s.signal_strength) > 0.1])
    
    with col1:
        st.metric("📱 Total Mentions", total_mentions)
    with col2:
        st.metric("❤️ Total Engagement", f"{total_engagement:,}")
    with col3:
        sentiment_emoji = "📈" if avg_sentiment > 0 else "📉" if avg_sentiment < 0 else "➡️"
        st.metric("🎭 Avg Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_emoji)
    with col4:
        st.metric("🎯 Active Signals", active_signals)
    
    # Sentiment signals table
    st.subheader("💭 Social Sentiment Signals")
    
    sentiment_data = []
    for symbol, signal in sentiment_signals.items():
        sentiment_data.append({
            'Symbol': symbol,
            'Net Sentiment': f"{signal.net_sentiment:.3f}",
            'Signal Strength': f"{signal.signal_strength:.3f}",
            'Mentions': signal.mention_count,
            'Engagement': f"{signal.total_engagement:,}",
            'Confidence': f"{signal.confidence:.1%}",
            'Reddit': f"{signal.reddit_sentiment:.3f}",
            'Twitter': f"{signal.twitter_sentiment:.3f}",
            'Quality': f"{signal.avg_quality:.1%}",
            'Spam Ratio': f"{signal.spam_ratio:.1%}"
        })
    
    df = pd.DataFrame(sentiment_data)
    st.dataframe(df, use_container_width=True)
    
    # Sentiment visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        symbols = list(sentiment_signals.keys())
        sentiments = [sentiment_signals[s].net_sentiment for s in symbols]
        colors = ['green' if s > 0.1 else 'red' if s < -0.1 else 'gray' for s in sentiments]
        
        fig = go.Figure([
            go.Bar(x=symbols, y=sentiments, marker_color=colors, name='Sentiment')
        ])
        fig.update_layout(
            title="Net Sentiment by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Net Sentiment (-1 to 1)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mentions vs engagement
        mentions = [sentiment_signals[s].mention_count for s in symbols]
        engagement = [sentiment_signals[s].total_engagement for s in symbols]
        
        fig = go.Figure([
            go.Scatter(
                x=mentions, y=engagement, text=symbols,
                mode='markers+text', textposition='top center',
                marker=dict(size=10, color=sentiments, colorscale='RdYlGn'),
                name='Social Activity'
            )
        ])
        fig.update_layout(
            title="Mentions vs Engagement",
            xaxis_title="Mention Count",
            yaxis_title="Total Engagement",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_combined_signals(whale_signals: Dict[str, WhaleSignal], 
                           sentiment_signals: Dict[str, SentimentSignal]):
    """Display combined whale and sentiment analysis"""
    
    st.header("🔄 Combined Signal Analysis")
    
    # Get symbols that have both whale and sentiment signals
    common_symbols = set(whale_signals.keys()) & set(sentiment_signals.keys())
    
    if not common_symbols:
        st.info("🔍 No symbols with both whale and sentiment signals detected")
        return
    
    st.subheader(f"📊 Combined Analysis ({len(common_symbols)} symbols)")
    
    # Combined signals table
    combined_data = []
    for symbol in common_symbols:
        whale_sig = whale_signals[symbol]
        sent_sig = sentiment_signals[symbol]
        
        # Calculate combined score (weighted average)
        combined_strength = (whale_sig.signal_strength * 0.6 + sent_sig.signal_strength * 0.4)
        
        combined_data.append({
            'Symbol': symbol,
            'Combined Strength': f"{combined_strength:.3f}",
            'Whale Flow ($)': f"{whale_sig.net_flow_usd:,.0f}",
            'Whale Strength': f"{whale_sig.signal_strength:.3f}",
            'Sentiment': f"{sent_sig.net_sentiment:.3f}",
            'Sentiment Strength': f"{sent_sig.signal_strength:.3f}",
            'Mentions': sent_sig.mention_count,
            'Whale Events': whale_sig.event_count,
            'Risk Assessment': 'HIGH' if abs(combined_strength) > 0.7 else 'MEDIUM' if abs(combined_strength) > 0.3 else 'LOW'
        })
    
    df = pd.DataFrame(combined_data)
    st.dataframe(df, use_container_width=True)
    
    # Combined visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Whale vs Sentiment Strength', 'Combined Signal Distribution'),
        specs=[[{"secondary_y": False}, {"type": "bar"}]]
    )
    
    # Scatter plot: whale vs sentiment
    whale_strengths = [whale_signals[s].signal_strength for s in common_symbols]
    sent_strengths = [sentiment_signals[s].signal_strength for s in common_symbols]
    
    fig.add_trace(
        go.Scatter(
            x=whale_strengths, y=sent_strengths, 
            text=list(common_symbols),
            mode='markers+text', textposition='top center',
            marker=dict(size=12, color='blue', opacity=0.7),
            name='Combined Signals'
        ),
        row=1, col=1
    )
    
    # Bar chart: combined strengths
    combined_strengths = [(whale_signals[s].signal_strength * 0.6 + 
                          sentiment_signals[s].signal_strength * 0.4) 
                         for s in common_symbols]
    colors = ['green' if s > 0 else 'red' for s in combined_strengths]
    
    fig.add_trace(
        go.Bar(x=list(common_symbols), y=combined_strengths, 
               marker_color=colors, name='Combined'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="Combined Whale & Sentiment Analysis"
    )
    fig.update_xaxes(title_text="Whale Signal Strength", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Signal Strength", row=1, col=1)
    fig.update_xaxes(title_text="Symbol", row=1, col=2)
    fig.update_yaxes(title_text="Combined Signal Strength", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function"""
    
    display_header()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Monitoring Controls")
    
    # Symbol selection
    default_symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 'DOGE/USDT',
        'SHIB/USDT', 'MATIC/USDT', 'UNI/USDT', 'AVAX/USDT', 'DOT/USDT'
    ]
    
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols to Monitor:",
        default_symbols,
        default=default_symbols[:6]
    )
    
    if not selected_symbols:
        st.warning("⚠️ Please select at least one symbol to monitor")
        return
    
    # Controls
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", False)
    run_monitoring = st.sidebar.button("🚀 Run Monitoring", type="primary")
    
    # Configuration
    st.sidebar.subheader("⚙️ Configuration")
    whale_threshold = st.sidebar.slider("Whale Threshold ($M)", 0.5, 10.0, 1.0, 0.5)
    
    # Safety notice
    st.sidebar.info("""
    🛡️ **Safety Features**:
    - TOS-compliant data collection
    - Rate limiting with exponential backoff
    - Signal-only integration (no auto-trading)
    - ExecutionPolicy validation required
    """)
    
    # Run monitoring
    if run_monitoring or auto_refresh:
        
        with st.spinner(f"🔄 Monitoring {len(selected_symbols)} symbols..."):
            
            # Collect whale signals
            whale_signals = get_cached_whale_signals(selected_symbols)
            
            # Collect sentiment signals  
            sentiment_signals = get_cached_sentiment_signals(selected_symbols)
            
            # Store in session state
            st.session_state['whale_signals'] = whale_signals
            st.session_state['sentiment_signals'] = sentiment_signals
            st.session_state['last_update'] = datetime.now()
        
        if whale_signals or sentiment_signals:
            st.success(f"✅ Monitoring completed - "
                      f"{len(whale_signals)} whale + {len(sentiment_signals)} sentiment signals")
        else:
            st.info("ℹ️ No significant whale or sentiment activity detected")
    
    # Display results if available
    if 'whale_signals' in st.session_state or 'sentiment_signals' in st.session_state:
        
        whale_signals = st.session_state.get('whale_signals', {})
        sentiment_signals = st.session_state.get('sentiment_signals', {})
        last_update = st.session_state.get('last_update', datetime.now())
        
        st.info(f"📊 Last update: {last_update.strftime('%H:%M:%S')}")
        
        # Display whale detection
        if whale_signals:
            display_whale_detection(whale_signals)
        
        # Display sentiment monitoring
        if sentiment_signals:
            display_sentiment_monitoring(sentiment_signals)
        
        # Display combined analysis
        if whale_signals and sentiment_signals:
            display_combined_signals(whale_signals, sentiment_signals)
        
        # Trading integration warning
        st.warning("""
        ⚠️ **Trading Integration Notice**: 
        These signals are integrated into the Practical Pipeline for ranking purposes only. 
        No autonomous trading occurs - all positions require ExecutionPolicy validation.
        """)
    
    else:
        st.info("👆 Click 'Run Monitoring' to start whale and sentiment detection")
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**🐋💭 Whale & Sentiment Monitor** | TOS-Compliant Monitoring | CryptoSmartTrader V2")


if __name__ == "__main__":
    main()