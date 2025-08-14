#!/usr/bin/env python3
"""
Market Data Simulator voor Alpha Motor Testing

Genereert realistische crypto market data voor alpha motor development.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


class MarketDataSimulator:
    """Simuleert realistische crypto market data"""
    
    def __init__(self):
        # Top crypto symbols met realistische parameters
        self.symbols = [
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'XRP/USD', 'ADA/USD',
            'SOL/USD', 'DOGE/USD', 'DOT/USD', 'AVAX/USD', 'SHIB/USD',
            'LINK/USD', 'UNI/USD', 'LTC/USD', 'ALGO/USD', 'VET/USD',
            'FIL/USD', 'TRX/USD', 'ETC/USD', 'XLM/USD', 'THETA/USD',
            'ATOM/USD', 'FTT/USD', 'NEAR/USD', 'HBAR/USD', 'SAND/USD',
            'MANA/USD', 'AXS/USD', 'ICP/USD', 'EGLD/USD', 'AAVE/USD',
            'GRT/USD', 'ENJ/USD', 'CHZ/USD', 'FLOW/USD', 'XTZ/USD'
        ]
        
        # Market cap tiers voor realistic volume/liquidity
        self.market_cap_tiers = {
            'large': (10_000_000_000, 500_000_000_000),  # $10B - $500B
            'mid': (1_000_000_000, 10_000_000_000),      # $1B - $10B  
            'small': (100_000_000, 1_000_000_000)        # $100M - $1B
        }

    def generate_market_snapshot(self) -> Dict[str, Any]:
        """Genereert complete market snapshot"""
        
        coins = []
        timestamp = datetime.now()
        
        for i, symbol in enumerate(self.symbols):
            # Determine tier based on position (top coins = large cap)
            if i < 10:
                tier = 'large'
            elif i < 25:
                tier = 'mid'
            else:
                tier = 'small'
                
            coin_data = self._generate_coin_data(symbol, tier)
            coins.append(coin_data)
            
        return {
            'timestamp': timestamp.isoformat(),
            'market_summary': {
                'total_market_cap': sum(c['market_cap_usd'] for c in coins),
                'total_volume_24h': sum(c['volume_24h_usd'] for c in coins),
                'coins_count': len(coins)
            },
            'coins': coins
        }

    def _generate_coin_data(self, symbol: str, tier: str) -> Dict[str, Any]:
        """Genereert realistische data voor één coin"""
        
        # Market cap based on tier
        min_cap, max_cap = self.market_cap_tiers[tier]
        market_cap = random.uniform(min_cap, max_cap)
        
        # Volume/turnover ratio (higher for smaller caps)
        if tier == 'large':
            turnover_ratio = random.uniform(0.05, 0.25)  # 5-25% daily turnover
        elif tier == 'mid':
            turnover_ratio = random.uniform(0.10, 0.40)  # 10-40%
        else:
            turnover_ratio = random.uniform(0.15, 0.60)  # 15-60%
            
        volume_24h = market_cap * turnover_ratio
        
        # Price based on market cap (fake price for calculation)
        supply = random.uniform(100_000_000, 10_000_000_000)  # 100M - 10B tokens
        price = market_cap / supply
        
        # Liquidity metrics (better for larger caps)
        if tier == 'large':
            spread_bps = random.uniform(2, 15)
            depth_multiplier = random.uniform(0.01, 0.05)  # 1-5% of volume
        elif tier == 'mid':
            spread_bps = random.uniform(5, 30)
            depth_multiplier = random.uniform(0.005, 0.03)
        else:
            spread_bps = random.uniform(10, 80)
            depth_multiplier = random.uniform(0.002, 0.02)
            
        depth_1pct = volume_24h * depth_multiplier
        
        # Technical indicators
        price_change_24h = random.gauss(0, 0.08)  # 8% daily vol
        rsi_14 = random.uniform(20, 80)
        volume_7d_avg = volume_24h * random.uniform(0.7, 1.5)
        
        # Derivatives data
        funding_rate_8h = random.gauss(0, 0.002)  # ~0.2% 8h funding
        oi_change_24h = random.gauss(0, 0.15)  # 15% OI volatility
        
        # Social sentiment
        mention_base = max(10, int(volume_24h / 1_000_000))  # More mentions for higher volume
        social_mentions_24h = max(1, int(random.gammavariate(2, mention_base)))
        sentiment_score = np.random.beta(2, 2) * 0.8 + 0.1  # 0.1 - 0.9 range, centered around 0.5
        
        return {
            'symbol': symbol,
            'price_usd': price,
            'market_cap_usd': market_cap,
            'volume_24h_usd': volume_24h,
            'volume_7d_avg': volume_7d_avg,
            'price_change_24h_pct': price_change_24h * 100,
            'spread_bps': spread_bps,
            'depth_1pct_usd': depth_1pct,
            'rsi_14': rsi_14,
            'funding_rate_8h_pct': funding_rate_8h * 100,
            'oi_change_24h_pct': oi_change_24h * 100,
            'social_mentions_24h': social_mentions_24h,
            'sentiment_score': sentiment_score,
            'tier': tier,
            'liquidity_score': min(100, depth_1pct / 1000),  # Liquidity score 0-100
            'momentum_proxy': abs(price_change_24h) * (volume_24h / volume_7d_avg)
        }

    def generate_time_series(self, symbol: str, days: int = 30) -> List[Dict]:
        """Genereert time series data voor backtesting"""
        
        series = []
        base_price = random.uniform(0.01, 50000)  # Wide price range
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Random walk met trend
            daily_return = random.gauss(0.001, 0.05)  # 5% daily vol
            base_price *= (1 + daily_return)
            
            # Volume pattern (higher on volatile days)
            base_volume = random.uniform(1_000_000, 100_000_000)
            volatility_factor = abs(daily_return) * 10
            volume = base_volume * (1 + volatility_factor)
            
            candle = {
                'timestamp': date.isoformat(),
                'symbol': symbol,
                'open': base_price * random.uniform(0.99, 1.01),
                'high': base_price * random.uniform(1.00, 1.05),
                'low': base_price * random.uniform(0.95, 1.00),
                'close': base_price,
                'volume': volume,
                'daily_return_pct': daily_return * 100
            }
            
            series.append(candle)
            
        return series

    def add_signal_scenarios(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Voegt specifieke signal scenarios toe voor testing"""
        
        coins = market_data['coins']
        
        # Scenario 1: Momentum breakout (top 3 coins)
        for i in range(3):
            coins[i]['price_change_24h_pct'] = random.uniform(15, 35)  # Strong momentum
            coins[i]['volume_24h_usd'] *= random.uniform(2, 5)  # Volume surge
            coins[i]['rsi_14'] = random.uniform(65, 80)  # Overbought
            coins[i]['social_mentions_24h'] *= random.randint(3, 8)  # Social buzz
            
        # Scenario 2: Mean reversion setup (next 3 coins)  
        for i in range(3, 6):
            coins[i]['price_change_24h_pct'] = random.uniform(-25, -10)  # Oversold
            coins[i]['rsi_14'] = random.uniform(15, 35)  # Oversold RSI
            coins[i]['sentiment_score'] = random.uniform(0.1, 0.3)  # Poor sentiment
            
        # Scenario 3: Funding rate arbitrage (next 2 coins)
        for i in range(6, 8):
            coins[i]['funding_rate_8h_pct'] = random.uniform(0.3, 0.8)  # High funding
            coins[i]['oi_change_24h_pct'] = random.uniform(20, 50)  # OI surge
            
        # Scenario 4: Social/event driven (next 2 coins)
        for i in range(8, 10):
            coins[i]['social_mentions_24h'] *= random.randint(10, 25)  # Viral
            coins[i]['sentiment_score'] = random.uniform(0.8, 0.95)  # Very positive
            coins[i]['volume_24h_usd'] *= random.uniform(3, 8)  # Volume spike
            
        return market_data

    def save_sample_data(self, filename: str = "sample_market_data.json"):
        """Slaat sample data op voor development"""
        
        market_data = self.generate_market_snapshot()
        market_data = self.add_signal_scenarios(market_data)
        
        with open(filename, 'w') as f:
            json.dump(market_data, f, indent=2)
            
        print(f"Sample data saved to {filename}")
        return market_data


if __name__ == "__main__":
    simulator = MarketDataSimulator()
    
    # Generate en save sample data
    data = simulator.save_sample_data()
    
    print(f"Generated {len(data['coins'])} coins")
    print(f"Total market cap: ${data['market_summary']['total_market_cap']:,.0f}")
    print(f"Total volume: ${data['market_summary']['total_volume_24h']:,.0f}")