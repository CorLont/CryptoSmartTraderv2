# agents/data_collector.py - Real-time data collection agent
import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class DataCollectorAgent:
    """Real-time cryptocurrency data collection with 100% coverage monitoring"""
    
    def __init__(self):
        self.exchanges = {
            'kraken': ccxt.kraken(),
            'binance': ccxt.binance(),
        }
        self.target_coins = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'SOL/USD', 'AVAX/USD']
        self.horizons = ['1h', '24h', '168h', '720h']  # 1h, 1d, 7d, 30d
        
    async def collect_market_data(self):
        """Collect market data from multiple exchanges"""
        market_data = []
        
        for coin in self.target_coins:
            try:
                # Get from Kraken (primary)
                ticker = self.exchanges['kraken'].fetch_ticker(coin)
                ohlcv = self.exchanges['kraken'].fetch_ohlcv(coin, '1h', limit=720)  # 30 days
                
                # Calculate technical indicators
                closes = [candle[4] for candle in ohlcv[-100:]]  # Last 100 closes
                rsi = self._calculate_rsi(closes)
                macd = self._calculate_macd(closes)
                
                market_data.append({
                    'coin': coin.replace('/USD', ''),
                    'timestamp': datetime.utcnow(),
                    'price': ticker['last'],
                    'volume_24h': ticker['quoteVolume'],
                    'change_24h': ticker['percentage'],
                    'feat_rsi_14': rsi,
                    'feat_macd': macd,
                    'feat_vol_24h': ticker['quoteVolume'],
                    'feat_price_change_1h': self._calc_price_change(ohlcv, 1),
                    'feat_price_change_24h': ticker['percentage'] / 100,
                })
                
            except Exception as e:
                logger.error(f"Data collection failed for {coin}: {e}")
                
        return market_data
    
    def _calculate_rsi(self, closes, period=14):
        """Calculate RSI indicator"""
        if len(closes) < period + 1:
            return 50.0
            
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, closes):
        """Calculate MACD indicator"""
        if len(closes) < 26:
            return 0.0
            
        ema12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
        return ema12 - ema26
    
    def _calc_price_change(self, ohlcv, hours):
        """Calculate price change over specified hours"""
        if len(ohlcv) < hours + 1:
            return 0.0
        return (ohlcv[-1][4] - ohlcv[-hours-1][4]) / ohlcv[-hours-1][4]

    async def run_continuous(self):
        """Run data collection continuously"""
        while True:
            try:
                data = await self.collect_market_data()
                
                # Save to features file
                df = pd.DataFrame(data)
                Path("exports").mkdir(exist_ok=True)
                df.to_parquet("exports/features.parquet")
                
                # Update coverage metrics
                coverage = len(data) / len(self.target_coins) * 100
                
                metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'coverage_pct': coverage,
                    'coins_collected': len(data),
                    'target_coins': len(self.target_coins)
                }
                
                with open('logs/coverage_metrics.json', 'w') as f:
                    json.dump(metrics, f)
                
                logger.info(f"Data collection: {coverage:.1f}% coverage ({len(data)} coins)")
                
            except Exception as e:
                logger.error(f"Data collection cycle failed: {e}")
            
            await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    agent = DataCollectorAgent()
    asyncio.run(agent.run_continuous())