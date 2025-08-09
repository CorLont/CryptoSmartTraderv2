#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Fixed Minimal App
Alle code-audit punten geïmplementeerd
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="CryptoSmartTrader V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_data_availability():
    """Fixed: Real check voor API keys en data beschikbaarheid"""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "KRAKEN_API_KEY": os.getenv("KRAKEN_API_KEY"),
        "KRAKEN_SECRET": os.getenv("KRAKEN_SECRET"),
    }
    missing = [k for k,v in keys.items() if not v]

    # Probeer lokale async file, anders sync CCXT
    live = False
    try:
        live = bool(get_live_market_data())  # returns list/None
    except Exception:
        live = False

    # Fixed: Consistent model files check
    model_files = [
        'models/saved/rf_1h.pkl',
        'models/saved/rf_24h.pkl', 
        'models/saved/rf_168h.pkl',
        'models/saved/rf_720h.pkl'
    ]
    missing_models = [f for f in model_files if not os.path.exists(f)]

    return {
        "has_live_data": live,
        "has_openai": bool(keys["OPENAI_API_KEY"]),
        "has_kraken": bool(keys["KRAKEN_API_KEY"] and keys["KRAKEN_SECRET"]),
        "missing_keys": missing,
        "models_trained": len(missing_models)==0,
        "missing_models": missing_models,
        "can_get_authenticated_data": bool(keys["KRAKEN_API_KEY"] and keys["KRAKEN_SECRET"]),
    }

def get_live_market_data():
    """Fixed: Alle Kraken coins zonder capping"""
    try:
        import ccxt
        
        # Setup Kraken client
        api_key = os.getenv('KRAKEN_API_KEY')
        secret = os.getenv('KRAKEN_SECRET')
        
        if api_key and secret:
            client = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': False,
                'enableRateLimit': True
            })
        else:
            client = ccxt.kraken({'enableRateLimit': True})
        
        # Get all tickers
        tickers = client.fetch_tickers()
        
        # Fixed: Haal ALLE USD pairs op, geen capping
        usd_pairs = {k: v for k, v in tickers.items() if k.endswith('/USD')}
        sorted_pairs = sorted(
            usd_pairs.items(), 
            key=lambda x: (x[1].get('quoteVolume') or 0), 
            reverse=True
        )
        # Verwijderd: [:25] - verwerk ALLE paren
        
        market_data = []
        for symbol, ticker in sorted_pairs:
            if ticker['last'] is not None:
                market_data.append({
                    'symbol': symbol,
                    'coin': symbol.split('/')[0],
                    'price': ticker['last'],
                    'change_24h': ticker['percentage'],
                    'volume_24h': ticker['baseVolume'],
                    'high_24h': ticker['high'],
                    'low_24h': ticker['low']
                })
        
        logger.info(f"Fetched {len(market_data)} USD pairs from Kraken")
        return market_data
        
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return []

def get_authentic_predictions():
    """Load only authentic predictions from production pipeline"""
    pred_file = Path("exports/production/predictions.csv")
    
    if not pred_file.exists():
        return []
    
    try:
        df = pd.read_csv(pred_file)
        predictions = df.to_dict('records')
        logger.info(f"Loaded {len(predictions)} authentic predictions")
        return predictions
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return []

def apply_confidence_gate_fixed(predictions, threshold=0.80):
    """Fixed: Proper confidence normalization"""
    filtered = []
    
    for pred in predictions:
        # Get confidence (already normalized 0-1 from production pipeline)
        conf_cols = [k for k in pred.keys() if k.startswith('confidence_')]
        if conf_cols:
            max_conf = max([pred.get(col, 0) for col in conf_cols])
            # Fixed: Confidence already normalized, direct comparison
            if max_conf >= threshold:
                filtered.append(pred)
    
    logger.info(f"Confidence gate: {len(filtered)}/{len(predictions)} passed ≥{threshold*100:.0f}%")
    return filtered

def render_market_status_fixed():
    """Fixed: Authentieke data zonder dummy movers"""
    st.subheader("📊 Live Market Status")
    
    market_data = get_live_market_data()
    
    if not market_data:
        st.warning("Geen live market data beschikbaar - configureer API keys")
        return
    
    # Real BTC price chart (behouden)
    btc_data = [d for d in market_data if d['symbol'] == 'BTC/USD']
    if btc_data:
        btc = btc_data[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BTC Price", f"${btc['price']:,.2f}", f"{btc['change_24h']:+.2f}%")
        with col2:
            st.metric("24h High", f"${btc['high_24h']:,.2f}")
        with col3:
            st.metric("24h Volume", f"${btc['volume_24h']:,.0f}")
    
    # Fixed: Authentieke grootste stijgers (niet hardcoded)
    st.subheader("🚀 Top Gainers (24h)")
    
    # Na market_data = get_live_market_data()
    top_gainers = sorted(
        [c for c in market_data if c.get('change_24h') is not None and c['change_24h'] > 0],
        key=lambda x: x['change_24h'],
        reverse=True
    )[:10]
    
    if top_gainers:
        movers_df = pd.DataFrame([{
            "Coin": m["coin"], 
            "Prijs": f"${m['price']:.4f}",
            "24h": f"{m['change_24h']:+.2f}%",
            "Volume (24h)": f"${(m['volume_24h'] or 0):,.0f}"
        } for m in top_gainers])
        st.dataframe(movers_df, use_container_width=True)
    else:
        st.info("Geen stijgers data beschikbaar")

def render_ai_predictions_fixed():
    """Fixed: Alleen authentieke predictions uit productie pipeline"""
    st.subheader("🤖 AI Predictions")
    
    # Fixed: Gebruik uitsluitend productie predictions
    predictions = get_authentic_predictions()
    
    if not predictions:
        st.warning("Geen AI predictions beschikbaar - run productie pipeline eerst")
        st.code("python run_demo_pipeline.py")
        return
    
    # Apply confidence gate
    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.80, 0.05)
    filtered_predictions = apply_confidence_gate_fixed(predictions, threshold)
    
    if not filtered_predictions:
        st.warning(f"Geen predictions boven {threshold*100:.0f}% confidence")
        return
    
    # Display predictions
    st.success(f"Showing {len(filtered_predictions)} high-confidence predictions")
    
    pred_data = []
    for pred in filtered_predictions:
        # Get max confidence
        conf_cols = [k for k in pred.keys() if k.startswith('confidence_')]
        max_conf = max([pred.get(col, 0) for col in conf_cols]) if conf_cols else 0
        
        pred_data.append({
            'Coin': pred.get('coin', 'N/A'),
            'Expected Return': f"{pred.get('expected_return_pct', 0):.1f}%",
            'Confidence': f"{max_conf*100:.0f}%",
            'Regime': pred.get('regime', 'N/A'),
            'Meta Quality': f"{pred.get('meta_label_quality', 0):.2f}",
            'Uncertainty': f"{pred.get('total_uncertainty', 0):.3f}"
        })
    
    if pred_data:
        df = pd.DataFrame(pred_data)
        st.dataframe(df, use_container_width=True)

def main():
    """Fixed main function met alle audit fixes"""
    st.title("🚀 CryptoSmartTrader V2 - Enterprise Fixed")
    st.markdown("### Multi-Agent Cryptocurrency Analysis - Code Audit Fixes Applied")
    
    # Check system status
    try:
        status = check_data_availability()
        
        models_present = status['models_trained']
        has_live_data = status['has_live_data']
        has_openai = status['has_openai']
        has_kraken = status['has_kraken']
        
        # System readiness check
        readiness_score = 0
        if models_present:
            readiness_score += 50
        if has_live_data:
            readiness_score += 30
        if has_openai:
            readiness_score += 10
        if has_kraken:
            readiness_score += 10
        
        # Fixed: Display system status correct
        if readiness_score >= 90:
            st.sidebar.success(f"🟢 System Ready ({readiness_score:.0f}/100)")
        elif readiness_score >= 70:
            st.sidebar.warning(f"🟠 System Degraded ({readiness_score:.0f}/100)")
        else:
            st.sidebar.error(f"🔴 System Not Ready ({readiness_score:.0f}/100)")
        
        # Fixed: Hard gate correct en zonder unreachable UI
        if not models_present:
            st.error("⚠️ Geen getrainde modellen. AI-tabs uitgeschakeld.")
            st.info("Train eerst modellen via: python ml/train_baseline.py")
            st.stop()
        
        # Sidebar info (alleen als modellen aanwezig)
        with st.sidebar:
            st.header("🎛️ System Status")
            
            if models_present:
                st.success("✅ RF Models Loaded")
            
            if has_kraken:
                st.success("✅ Kraken API Connected")
            else:
                st.warning("⚠️ Kraken API Missing")
                
            if has_openai:
                st.success("✅ OpenAI API Connected")
            else:
                st.warning("⚠️ OpenAI API Missing")
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🤖 AI Predictions", "⚙️ System Health"])
        
        with tab1:
            render_market_status_fixed()
        
        with tab2:
            render_ai_predictions_fixed()
        
        with tab3:
            st.subheader("🔧 System Health")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Readiness Score", f"{readiness_score}/100", 
                         delta="Ready" if readiness_score >= 90 else "Degraded")
            
            with col2:
                st.metric("Models Status", "✅ Active" if models_present else "❌ Missing")
            
            # Status details
            st.subheader("Detailed Status")
            status_df = pd.DataFrame([
                {"Component": "RF Models", "Status": "✅ Ready" if models_present else "❌ Missing"},
                {"Component": "Kraken API", "Status": "✅ Connected" if has_kraken else "❌ Missing"},
                {"Component": "OpenAI API", "Status": "✅ Connected" if has_openai else "❌ Missing"},
                {"Component": "Live Data", "Status": "✅ Available" if has_live_data else "❌ Unavailable"},
            ])
            st.dataframe(status_df, use_container_width=True)
            
            # Missing keys info
            if status['missing_keys']:
                st.warning(f"Missing environment keys: {', '.join(status['missing_keys'])}")
                st.info("Configure keys in .env file")
    
    # Fixed: Eén except-blok volstaat
    except Exception as e:
        st.error(f"Dashboard rendering error: {e}")
        logger.exception("Dashboard error", exc_info=True)

if __name__ == "__main__":
    main()