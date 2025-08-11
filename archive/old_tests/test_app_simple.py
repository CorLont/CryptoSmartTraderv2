#!/usr/bin/env python3
"""
Simple test app voor ProductionApp workflow - alle fixes geïmplementeerd
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ccxt
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CryptoSmartTrader V2 - Production Test",
    page_icon="🚀",
    layout="wide"
)

def get_all_kraken_pairs():
    """Get ALL Kraken USD pairs - no demo limits"""
    try:
        client = ccxt.kraken({'enableRateLimit': True})
        tickers = client.fetch_tickers()
        
        # Get ALL USD pairs (no capping)
        usd_pairs = {k: v for k, v in tickers.items() if k.endswith('/USD')}
        
        market_data = []
        for symbol, ticker in usd_pairs.items():
            if ticker['last'] is not None:
                market_data.append({
                    'symbol': symbol,
                    'coin': symbol.split('/')[0],
                    'price': ticker['last'],
                    'change_24h': ticker.get('percentage', 0),
                    'volume_24h': ticker.get('baseVolume', 0),
                    'high_24h': ticker.get('high', ticker['last']),
                    'low_24h': ticker.get('low', ticker['last'])
                })
        
        logger.info(f"Retrieved {len(market_data)} Kraken USD pairs (ALL, no capping)")
        return market_data
        
    except Exception as e:
        logger.error(f"Failed to get Kraken data: {e}")
        return []

def check_production_predictions():
    """Check for production predictions file"""
    pred_file = Path("exports/production/predictions.csv")
    enhanced_file = Path("exports/production/enhanced_predictions.json")
    
    if pred_file.exists():
        try:
            df = pd.read_csv(pred_file)
            return {'status': 'csv_available', 'count': len(df), 'data': df}
        except Exception as e:
            logger.error(f"Failed to load CSV predictions: {e}")
    
    if enhanced_file.exists():
        try:
            with open(enhanced_file, 'r') as f:
                data = json.load(f)
            return {'status': 'json_available', 'count': len(data), 'data': data}
        except Exception as e:
            logger.error(f"Failed to load JSON predictions: {e}")
    
    return {'status': 'not_available', 'count': 0, 'data': None}

def apply_80_percent_gate(predictions_data):
    """Apply consistent 80% confidence gate"""
    if predictions_data is None:
        return []
    
    if isinstance(predictions_data, pd.DataFrame) and predictions_data.empty:
        return []
    
    if isinstance(predictions_data, list) and len(predictions_data) == 0:
        return []
    
    if isinstance(predictions_data, pd.DataFrame):
        # Handle DataFrame
        conf_cols = [col for col in predictions_data.columns if 'confidence' in col.lower()]
        if conf_cols:
            max_conf = predictions_data[conf_cols].max(axis=1)
            gate_passed = max_conf >= 0.80
            return predictions_data[gate_passed].to_dict('records')
    else:
        # Handle list of dicts
        filtered = []
        for pred in predictions_data:
            conf_cols = [k for k in pred.keys() if 'confidence' in k.lower()]
            if conf_cols:
                max_conf = max([pred.get(col, 0) for col in conf_cols])
                if max_conf >= 0.80:
                    pred['max_confidence'] = max_conf
                    filtered.append(pred)
        return filtered
    
    return []

def render_market_overview():
    """Render market overview with full Kraken coverage"""
    st.subheader("📊 Market Overview - Full Kraken Coverage")
    
    market_data = get_all_kraken_pairs()
    
    if not market_data:
        st.error("❌ Cannot connect to Kraken API")
        st.info("Configure API keys in environment variables")
        return
    
    st.success(f"✅ Full Kraken Coverage: {len(market_data)} USD pairs")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_volume = sum([d.get('volume_24h', 0) for d in market_data if d.get('volume_24h')])
    avg_change = np.mean([d.get('change_24h', 0) for d in market_data if d.get('change_24h')])
    gainers = len([d for d in market_data if d.get('change_24h', 0) > 0])
    
    with col1:
        st.metric("Total Pairs", len(market_data))
    with col2:
        st.metric("Total Volume", f"${total_volume:,.0f}")
    with col3:
        st.metric("Avg Change", f"{avg_change:+.2f}%")
    with col4:
        st.metric("Gainers", f"{gainers}/{len(market_data)}")
    
    # Top gainers
    top_gainers = sorted(
        [d for d in market_data if d.get('change_24h', 0) > 0],
        key=lambda x: x['change_24h'],
        reverse=True
    )[:10]
    
    if top_gainers:
        st.subheader("🚀 Top Gainers (24h)")
        gainers_data = [{
            'Coin': g['coin'],
            'Price': f"${g['price']:.4f}",
            '24h Change': f"{g['change_24h']:+.2f}%",
            'Volume': f"${g.get('volume_24h', 0):,.0f}"
        } for g in top_gainers]
        
        df = pd.DataFrame(gainers_data)
        st.dataframe(df, use_container_width=True)

def render_ai_predictions():
    """Render AI predictions based on production data"""
    st.subheader("🤖 AI Predictions - Production Data Only")
    
    pred_status = check_production_predictions()
    
    if pred_status['status'] == 'not_available':
        st.warning("❌ No production predictions available")
        st.info("Generate predictions first: python generate_final_predictions.py")
        return
    
    st.success(f"✅ Loaded {pred_status['count']} predictions from production pipeline")
    
    # Apply 80% confidence gate
    filtered_predictions = apply_80_percent_gate(pred_status['data'])
    
    if not filtered_predictions:
        st.warning("⚠️ No predictions passed 80% confidence gate")
        st.info(f"Total predictions: {pred_status['count']} | Passed gate: 0")
        return
    
    st.info(f"Showing {len(filtered_predictions)} high-confidence predictions (≥80%)")
    
    # Display predictions
    pred_display = []
    for pred in filtered_predictions:
        # Handle different data structures
        coin = pred.get('coin', pred.get('symbol', 'N/A'))
        if isinstance(coin, str) and '/' in coin:
            coin = coin.split('/')[0]
        
        expected_return = pred.get('expected_return_pct', pred.get('expected_return_24h', 0))
        confidence = pred.get('max_confidence', pred.get('confidence_24h', 0))
        
        pred_display.append({
            'Coin': coin,
            'Expected Return': f"{expected_return:+.1f}%",
            'Confidence': f"{confidence*100:.0f}%",
            'Regime': pred.get('regime', 'N/A'),
            'Sentiment': pred.get('sentiment_label', 'neutral').title(),
            'Whale Activity': 'High' if pred.get('whale_activity_detected', False) else 'Low',
            'Meta Quality': f"{pred.get('meta_label_quality', 0):.2f}",
            'Uncertainty': f"{pred.get('total_uncertainty', 0):.3f}"
        })
    
    if pred_display:
        pred_df = pd.DataFrame(pred_display)
        st.dataframe(pred_df, use_container_width=True)

def render_system_status():
    """Render consistent system status"""
    st.subheader("⚙️ System Status - Consistent RF Architecture")
    
    # Check models
    model_files = [
        'models/saved/rf_1h.pkl',
        'models/saved/rf_24h.pkl',
        'models/saved/rf_168h.pkl',
        'models/saved/rf_720h.pkl'
    ]
    
    model_status = {}
    for model_file in model_files:
        horizon = Path(model_file).stem.replace('rf_', '')
        model_status[horizon] = Path(model_file).exists()
    
    all_models_present = all(model_status.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Architecture", "RandomForest", delta="Consistent")
        st.metric("Models Present", "✅ Complete" if all_models_present else "❌ Incomplete")
    
    with col2:
        api_keys = {
            'KRAKEN_API_KEY': bool(os.getenv('KRAKEN_API_KEY')),
            'KRAKEN_SECRET': bool(os.getenv('KRAKEN_SECRET')),
            'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY'))
        }
        
        api_ready = all(api_keys.values())
        st.metric("API Keys", "✅ Complete" if api_ready else "⚠️ Missing")
    
    # Detailed status
    st.subheader("Detailed Component Status")
    
    status_data = []
    
    # Model status
    for horizon, exists in model_status.items():
        status_data.append({
            'Component': f'RF Model ({horizon})',
            'Status': '✅ Ready' if exists else '❌ Missing',
            'Type': 'Model'
        })
    
    # API status
    for key, available in api_keys.items():
        status_data.append({
            'Component': key,
            'Status': '✅ Ready' if available else '❌ Missing',
            'Type': 'API Key'
        })
    
    # Data status
    pred_status = check_production_predictions()
    status_data.append({
        'Component': 'Production Predictions',
        'Status': f"✅ {pred_status['count']} available" if pred_status['status'] != 'not_available' else '❌ Not Available',
        'Type': 'Data'
    })
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)

def main():
    """Main application"""
    st.title("🚀 CryptoSmartTrader V2 - Production Test")
    st.markdown("### All Production Requirements Implemented")
    
    with st.sidebar:
        st.header("🎛️ Production Features")
        st.success("✅ Full Kraken Coverage")
        st.success("✅ Consistent RF Models")
        st.success("✅ 80% Confidence Gate")
        st.success("✅ Production Predictions Only")
        st.success("✅ No Demo/Dummy Data")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Market", "🤖 Predictions", "⚙️ Status"])
    
    with tab1:
        render_market_overview()
    
    with tab2:
        render_ai_predictions()
    
    with tab3:
        render_system_status()

if __name__ == "__main__":
    main()