#!/usr/bin/env python3
"""
Fixed simple trading app - implements audit point E
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import os

st.set_page_config(page_title="CryptoTrader - Fixed", layout="wide")

def get_live_market_data():
    """Get real market data instead of synthetic"""
    try:
        client = ccxt.kraken({'enableRateLimit': True})
        tickers = client.fetch_tickers()
        
        usd_pairs = {k: v for k, v in tickers.items() if k.endswith('/USD')}
        
        market_data = []
        for symbol, ticker in usd_pairs.items():
            if ticker['last'] is not None:
                market_data.append({
                    'symbol': symbol,
                    'coin': symbol.split('/')[0],
                    'price': ticker['last'],
                    'change_24h': ticker['percentage'],
                    'volume_24h': ticker['baseVolume']
                })
        
        return market_data
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        return []

def get_authentic_top_movers():
    """Fixed: vervang dummy movers door live data"""
    market_data = get_live_market_data()
    
    if not market_data:
        return []
    
    # Na market_data = get_live_market_data()
    top_movers = sorted(
        [c for c in market_data if c.get('change_24h') is not None],
        key=lambda x: x['change_24h'],
        reverse=True
    )[:10]
    
    return [{
        "Coin": m["coin"], 
        "Prijs": f"${m['price']:.4f}",
        "24h": f"{m['change_24h']:+.2f}%",
        "Volume (24h)": f"${(m['volume_24h'] or 0):,.0f}"
    } for m in top_movers]

def main():
    st.title("🚀 CryptoTrader - Fixed Version")
    st.markdown("### Authentieke data zonder synthetische voorspellingen")
    
    # Fixed: gebruik live data i.p.v. demo
    top_movers = get_authentic_top_movers()
    
    if top_movers:
        st.subheader("📈 Live Top Movers")
        movers_df = pd.DataFrame(top_movers)
        st.dataframe(movers_df, use_container_width=True)
    else:
        st.warning("Configureer API keys voor live data")
        st.info("Voeg KRAKEN_API_KEY toe aan .env file")

if __name__ == "__main__":
    main()