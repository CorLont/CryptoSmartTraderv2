#!/usr/bin/env python3
"""
Test Data Generators - Deterministic test data generation
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class DeterministicDataGenerator:
    """
    Generate deterministic test data for unit and integration tests
    
    Uses fixed seeds to ensure reproducible test data across runs
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    def generate_market_data(
        self,
        symbols: List[str] = None,
        periods: int = 1000,
        start_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Generate realistic OHLCV market data"""
        
        if symbols is None:
            symbols = ["BTC/USD", "ETH/USD", "BNB/USD"]
        
        # Generate timestamps
        start = pd.to_datetime(start_date)
        timestamps = pd.date_range(start=start, periods=periods, freq='H')
        
        data = []
        
        for symbol in symbols:
            # Base price varies by symbol
            base_prices = {
                "BTC/USD": 50000,
                "ETH/USD": 3000,
                "BNB/USD": 400,
                "XRP/USD": 0.5,
                "ADA/USD": 1.2
            }
            
            base_price = base_prices.get(symbol, 100)
            
            # Generate price series with realistic patterns
            returns = np.random.normal(0, 0.02, periods)  # 2% hourly volatility
            
            # Add some trend and mean reversion
            trend = np.linspace(-0.1, 0.1, periods)
            mean_reversion = -0.001 * np.cumsum(returns)
            
            adjusted_returns = returns + trend + mean_reversion
            
            # Calculate price series
            prices = [base_price]
            for ret in adjusted_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate OHLC from close prices
            for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC spread
                spread = abs(np.random.normal(0, 0.005)) + 0.001  # 0.1-0.6% spread
                
                high = close_price * (1 + spread * np.random.uniform(0.5, 1.0))
                low = close_price * (1 - spread * np.random.uniform(0.5, 1.0))
                
                # Open price (previous close + gap)
                if i == 0:
                    open_price = close_price
                else:
                    gap = np.random.normal(0, 0.001)  # Small gap
                    open_price = prices[i-1] * (1 + gap)
                
                # Ensure OHLC relationships
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                # Volume correlated with volatility
                volatility = abs(adjusted_returns[i]) if i < len(adjusted_returns) else 0.01
                base_volume = 1000000
                volume = base_volume * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
                
                data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': round(volume),
                    'volume_usd': round(volume * close_price)
                })
        
        df = pd.DataFrame(data)
        
        # Save to parquet for efficient loading
        output_path = self.data_dir / "sample_market_data.parquet"
        df.to_parquet(output_path, index=False)
        
        return df
    
    def generate_portfolio_data(self) -> Dict[str, Any]:
        """Generate sample portfolio data"""
        
        positions = [
            {
                "symbol": "BTC/USD",
                "quantity": 2.5,
                "avg_price": 45000,
                "current_price": 47500,
                "unrealized_pnl": 6250,
                "realized_pnl": 1200,
                "entry_time": "2024-01-15T10:30:00Z",
                "last_update": "2024-01-20T14:25:00Z"
            },
            {
                "symbol": "ETH/USD",
                "quantity": 15.0,
                "avg_price": 3200,
                "current_price": 3350,
                "unrealized_pnl": 2250,
                "realized_pnl": 800,
                "entry_time": "2024-01-16T09:15:00Z",
                "last_update": "2024-01-20T14:25:00Z"
            },
            {
                "symbol": "BNB/USD",
                "quantity": 50.0,
                "avg_price": 380,
                "current_price": 395,
                "unrealized_pnl": 750,
                "realized_pnl": -200,
                "entry_time": "2024-01-18T11:45:00Z",
                "last_update": "2024-01-20T14:25:00Z"
            }
        ]
        
        portfolio = {
            "account_id": "test_account_123",
            "total_value": 142750.0,
            "cash_balance": 25000.0,
            "invested_value": 117750.0,
            "total_unrealized_pnl": 9250.0,
            "total_realized_pnl": 1800.0,
            "daily_pnl": 1250.0,
            "positions": positions,
            "last_update": "2024-01-20T14:25:00Z",
            "risk_metrics": {
                "total_exposure": 0.78,
                "max_position_size": 0.35,
                "correlation_risk": 0.45,
                "var_95": -5200.0
            }
        }
        
        # Save to JSON
        output_path = self.data_dir / "sample_portfolio.json"
        with open(output_path, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        return portfolio
    
    def generate_trading_signals(self, count: int = 100) -> pd.DataFrame:
        """Generate sample trading signals for backtesting"""
        
        start_date = pd.to_datetime("2024-01-01")
        
        signals = []
        
        for i in range(count):
            timestamp = start_date + timedelta(hours=i*6)  # Signal every 6 hours
            
            symbol = np.random.choice(['BTC/USD', 'ETH/USD', 'BNB/USD'], p=[0.5, 0.3, 0.2])
            signal_type = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])
            
            # Confidence with realistic distribution
            confidence = np.random.beta(3, 2)  # Skewed towards higher confidence
            confidence = max(0.5, min(0.95, confidence))  # Clamp to realistic range
            
            # Price at signal time
            base_prices = {"BTC/USD": 50000, "ETH/USD": 3000, "BNB/USD": 400}
            price_variation = np.random.uniform(-0.1, 0.1)
            signal_price = base_prices[symbol] * (1 + price_variation)
            
            signals.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': round(confidence, 3),
                'price': round(signal_price, 2),
                'agent': np.random.choice(['Technical', 'Sentiment', 'ML_Predictor']),
                'reason': f"Technical indicator signal ({signal_type})",
                'expected_move': np.random.uniform(0.02, 0.08),  # Expected 2-8% move
                'time_horizon': np.random.choice(['1h', '4h', '1d']),
                'risk_score': np.random.uniform(0.1, 0.7)
            })
        
        df = pd.DataFrame(signals)
        
        # Save to CSV
        output_path = self.data_dir / "sample_signals.csv"
        df.to_csv(output_path, index=False)
        
        return df
    
    def generate_news_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate sample news articles for sentiment analysis"""
        
        positive_keywords = [
            "surge", "rally", "breakthrough", "adoption", "partnership",
            "bullish", "momentum", "growth", "innovation", "upgrade"
        ]
        
        negative_keywords = [
            "crash", "decline", "bearish", "regulation", "ban",
            "concern", "volatility", "uncertainty", "risk", "pressure"
        ]
        
        neutral_keywords = [
            "analysis", "report", "update", "news", "market",
            "trading", "volume", "activity", "development", "trend"
        ]
        
        articles = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(count):
            # Random sentiment
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.2, 0.5])
            
            if sentiment == 'positive':
                keywords = positive_keywords
            elif sentiment == 'negative':
                keywords = negative_keywords
            else:
                keywords = neutral_keywords
            
            # Generate article
            symbol = np.random.choice(['Bitcoin', 'Ethereum', 'Binance Coin', 'Cryptocurrency'])
            keyword = np.random.choice(keywords)
            
            title = f"{symbol} {keyword}: Market Analysis and Implications"
            content = f"Recent developments in {symbol} show {keyword} patterns. " \
                     f"Market analysts are monitoring the situation closely. " \
                     f"This could impact trading strategies and investor sentiment."
            
            article = {
                'id': f"news_{i:03d}",
                'title': title,
                'content': content,
                'source': np.random.choice(['CryptoNews', 'BlockchainTimes', 'CoinDaily']),
                'author': f"Author {i % 10}",
                'published_at': (start_date + timedelta(hours=i*12)).isoformat(),
                'url': f"https://example.com/news/{i}",
                'sentiment_label': sentiment,
                'sentiment_score': {
                    'positive': 0.8 if sentiment == 'positive' else 0.1,
                    'negative': 0.8 if sentiment == 'negative' else 0.1,
                    'neutral': 0.8 if sentiment == 'neutral' else 0.1
                },
                'keywords': [keyword, symbol.lower()],
                'category': 'cryptocurrency',
                'relevance_score': np.random.uniform(0.6, 0.95)
            }
            
            articles.append(article)
        
        # Save to JSON
        output_path = self.data_dir / "sample_news.json"
        with open(output_path, 'w') as f:
            json.dump(articles, f, indent=2)
        
        return articles
    
    def generate_features_data(self, periods: int = 500) -> pd.DataFrame:
        """Generate sample engineered features"""
        
        start_date = pd.to_datetime("2024-01-01")
        timestamps = pd.date_range(start=start_date, periods=periods, freq='H')
        
        # Generate base price
        np.random.seed(self.seed)
        returns = np.random.normal(0, 0.02, periods)
        prices = [50000]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate technical indicators
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        # Moving averages
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        bb_std = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume'] = np.random.exponential(1000000, periods)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['price'].pct_change().rolling(20).std() * np.sqrt(24)
        
        # Support/Resistance levels
        df['support'] = df['price'].rolling(50).min()
        df['resistance'] = df['price'].rolling(50).max()
        df['support_distance'] = (df['price'] - df['support']) / df['support']
        df['resistance_distance'] = (df['resistance'] - df['price']) / df['price']
        
        # Market regime indicators
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['price']
        df['momentum'] = df['price'].pct_change(10)
        
        # Clean up NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        # Save to parquet
        output_path = self.data_dir / "sample_features.parquet"
        df.to_parquet(output_path, index=False)
        
        return df
    
    def generate_all_test_data(self):
        """Generate all test datasets"""
        print("Generating test data with deterministic seed...")
        
        market_data = self.generate_market_data()
        print(f"✓ Generated market data: {len(market_data)} records")
        
        portfolio_data = self.generate_portfolio_data()
        print(f"✓ Generated portfolio data: {len(portfolio_data['positions'])} positions")
        
        signals_data = self.generate_trading_signals()
        print(f"✓ Generated trading signals: {len(signals_data)} signals")
        
        news_data = self.generate_news_data()
        print(f"✓ Generated news data: {len(news_data)} articles")
        
        features_data = self.generate_features_data()
        print(f"✓ Generated features data: {len(features_data)} records")
        
        print("All test data generated successfully!")


if __name__ == "__main__":
    generator = DeterministicDataGenerator(seed=42)
    generator.generate_all_test_data()