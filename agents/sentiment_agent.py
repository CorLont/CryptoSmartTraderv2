import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from textblob import TextBlob
import os

# OpenAI integration for advanced sentiment analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SentimentAgent:
    """Sentiment Analysis Agent for social media and news sentiment"""
    
    def __init__(self, config_manager, data_manager, cache_manager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0
        
        # OpenAI client for advanced sentiment analysis
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
        
        # Sentiment data storage
        self.sentiment_data = {}
        self._lock = threading.Lock()
        
        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("sentiment", {}).get("enabled", True):
            self.start()
    
    def start(self):
        """Start the sentiment analysis agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Sentiment Agent started")
    
    def stop(self):
        """Stop the sentiment analysis agent"""
        self.active = False
        self.logger.info("Sentiment Agent stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = self.config_manager.get("agents", {}).get("sentiment", {}).get("update_interval", 300)
                
                # Perform sentiment analysis
                self._analyze_market_sentiment()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep until next analysis
                time.sleep(interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Sentiment analysis error: {str(e)}")
                time.sleep(60)  # Sleep on error
    
    def _analyze_market_sentiment(self):
        """Analyze market sentiment for all tracked cryptocurrencies"""
        try:
            # Get list of supported symbols
            symbols = self.data_manager.get_supported_symbols()
            
            for symbol in symbols[:50]:  # Limit to top 50 for efficiency
                try:
                    base_currency = symbol.split('/')[0] if '/' in symbol else symbol
                    sentiment_score = self._analyze_currency_sentiment(base_currency)
                    
                    if sentiment_score is not None:
                        self._store_sentiment_data(base_currency, sentiment_score)
                        self.processed_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Market sentiment analysis error: {str(e)}")
    
    def _analyze_currency_sentiment(self, currency: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment for a specific cryptocurrency"""
        # Check cache first
        cache_key = f"sentiment_{currency.lower()}"
        cached_sentiment = self.cache_manager.get(cache_key)
        
        if cached_sentiment is not None:
            return cached_sentiment
        
        try:
            # Generate mock sentiment data (in production, this would aggregate from multiple sources)
            sentiment_score = self._generate_sentiment_analysis(currency)
            
            # Cache the result
            self.cache_manager.set(cache_key, sentiment_score, ttl_minutes=30)
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Currency sentiment analysis error for {currency}: {str(e)}")
            return None
    
    def _generate_sentiment_analysis(self, currency: str) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis"""
        timestamp = datetime.now()
        
        # Mock sentiment sources (in production, these would be real data sources)
        news_sentiment = self._analyze_news_sentiment(currency)
        social_sentiment = self._analyze_social_sentiment(currency)
        market_sentiment = self._analyze_market_sentiment_indicators(currency)
        
        # Calculate weighted composite sentiment
        composite_score = (
            news_sentiment['score'] * 0.4 +
            social_sentiment['score'] * 0.4 +
            market_sentiment['score'] * 0.2
        )
        
        # Determine sentiment category
        if composite_score >= 0.6:
            category = "bullish"
        elif composite_score >= 0.4:
            category = "neutral"
        else:
            category = "bearish"
        
        return {
            'timestamp': timestamp.isoformat(),
            'currency': currency,
            'composite_score': composite_score,
            'category': category,
            'confidence': min(0.95, max(0.3, abs(composite_score - 0.5) * 2)),
            'components': {
                'news': news_sentiment,
                'social': social_sentiment,
                'market': market_sentiment
            },
            'volume_mentions': news_sentiment['mentions'] + social_sentiment['mentions'],
            'trend_direction': 'up' if composite_score > 0.5 else 'down',
            'strength': 'strong' if abs(composite_score - 0.5) > 0.3 else 'weak'
        }
    
    def _analyze_news_sentiment(self, currency: str) -> Dict[str, Any]:
        """Analyze news sentiment (mock implementation)"""
        import random
        
        # Mock news sentiment analysis
        # In production, this would scrape and analyze real news articles
        base_score = random.uniform(0.2, 0.8)
        mentions = random.randint(5, 50)
        
        return {
            'score': base_score,
            'mentions': mentions,
            'source_count': random.randint(3, 15),
            'avg_subjectivity': random.uniform(0.3, 0.7),
            'top_keywords': [f"{currency.lower()}_keyword_{i}" for i in range(3)]
        }
    
    def _analyze_social_sentiment(self, currency: str) -> Dict[str, Any]:
        """Analyze social media sentiment (mock implementation)"""
        import random
        
        # Mock social media sentiment analysis
        # In production, this would analyze Twitter, Reddit, Discord, etc.
        base_score = random.uniform(0.3, 0.7)
        mentions = random.randint(10, 200)
        
        return {
            'score': base_score,
            'mentions': mentions,
            'engagement_rate': random.uniform(0.05, 0.25),
            'platforms': {
                'twitter': {'score': random.uniform(0.2, 0.8), 'mentions': random.randint(5, 100)},
                'reddit': {'score': random.uniform(0.2, 0.8), 'mentions': random.randint(2, 50)},
                'discord': {'score': random.uniform(0.2, 0.8), 'mentions': random.randint(1, 30)}
            }
        }
    
    def _analyze_market_sentiment_indicators(self, currency: str) -> Dict[str, Any]:
        """Analyze market-based sentiment indicators"""
        import random
        
        # Get market data for the currency
        market_data = self.data_manager.get_market_data(symbol=f"{currency}/USD")
        
        if market_data is not None and not market_data.empty:
            latest = market_data.iloc[-1]
            
            # Calculate market sentiment based on price action and volume
            price_change = latest.get('change_percent', 0)
            volume_ratio = random.uniform(0.8, 1.5)  # Mock volume ratio
            
            # Convert price change to sentiment score
            sentiment_score = max(0, min(1, (price_change + 10) / 20))  # Normalize -10% to +10% range
            
        else:
            # Fallback if no market data
            sentiment_score = random.uniform(0.3, 0.7)
            volume_ratio = 1.0
            price_change = 0
        
        return {
            'score': sentiment_score,
            'mentions': 1,  # Market data is single source
            'price_momentum': 'positive' if price_change > 0 else 'negative',
            'volume_sentiment': 'high' if volume_ratio > 1.2 else 'normal',
            'technical_indicators': {
                'rsi_sentiment': random.uniform(0.3, 0.7),
                'ma_sentiment': random.uniform(0.3, 0.7)
            }
        }
    
    def _store_sentiment_data(self, currency: str, sentiment_data: Dict[str, Any]):
        """Store sentiment data"""
        with self._lock:
            if currency not in self.sentiment_data:
                self.sentiment_data[currency] = []
            
            self.sentiment_data[currency].append(sentiment_data)
            
            # Keep only last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.sentiment_data[currency] = [
                data for data in self.sentiment_data[currency]
                if datetime.fromisoformat(data['timestamp']) > cutoff_time
            ]
    
    def get_sentiment(self, currency: str) -> Optional[Dict[str, Any]]:
        """Get latest sentiment for a currency"""
        with self._lock:
            if currency in self.sentiment_data and self.sentiment_data[currency]:
                return self.sentiment_data[currency][-1]
            return None
    
    def get_sentiment_history(self, currency: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get sentiment history for a currency"""
        with self._lock:
            if currency not in self.sentiment_data:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                data for data in self.sentiment_data[currency]
                if datetime.fromisoformat(data['timestamp']) > cutoff_time
            ]
    
    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        with self._lock:
            if not self.sentiment_data:
                return {
                    'overall_sentiment': 'neutral',
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'total_currencies': 0
                }
            
            latest_sentiments = {}
            for currency, data_list in self.sentiment_data.items():
                if data_list:
                    latest_sentiments[currency] = data_list[-1]
            
            bullish_count = sum(1 for data in latest_sentiments.values() if data['category'] == 'bullish')
            bearish_count = sum(1 for data in latest_sentiments.values() if data['category'] == 'bearish')
            neutral_count = len(latest_sentiments) - bullish_count - bearish_count
            
            # Determine overall market sentiment
            if bullish_count > bearish_count * 1.5:
                overall_sentiment = 'bullish'
            elif bearish_count > bullish_count * 1.5:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'overall_sentiment': overall_sentiment,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'total_currencies': len(latest_sentiments),
                'avg_sentiment_score': sum(data['composite_score'] for data in latest_sentiments.values()) / len(latest_sentiments) if latest_sentiments else 0.5,
                'high_confidence_count': sum(1 for data in latest_sentiments.values() if data['confidence'] > 0.7)
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'tracked_currencies': len(self.sentiment_data),
            'openai_available': self.openai_client is not None
        }
