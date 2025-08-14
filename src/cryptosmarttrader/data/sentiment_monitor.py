#!/usr/bin/env python3
"""
Sentiment Monitoring System - TOS-compliant Reddit/X data collection

Deze module implementeert veilige social media sentiment monitoring met:
- Rate-limited clients met exponential backoff
- User-agent rotation en session management
- Officiële APIs waar mogelijk (Reddit API, Twitter API v2)
- Entity mapping naar crypto tickers met whitelist
- Noise filtering en sentiment aggregation
- Cache management met laatste IDs voor efficiency

SECURITY: Volledig TOS-compliant, geen scraping zonder toestemming.
"""

import asyncio
import logging
import time
import json
import random
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import hashlib
from collections import defaultdict, deque
import re
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Supported sentiment data sources"""
    REDDIT = "reddit"
    TWITTER = "twitter"
    TELEGRAM = "telegram"  # Future support
    DISCORD = "discord"   # Future support


@dataclass
class SentimentMention:
    """Individual mention from social media"""
    mention_id: str
    symbol: str
    source: SentimentSource
    content: str
    author: str
    timestamp: datetime
    
    # Engagement metrics
    upvotes: int = 0
    comments: int = 0
    shares: int = 0
    likes: int = 0
    
    # Sentiment analysis
    sentiment_score: float = 0.0  # -1 (bearish) to 1 (bullish)
    confidence: float = 0.0       # 0-1 confidence in sentiment
    keywords: Optional[List[str]] = None
    
    # Quality indicators
    is_spam: bool = False
    is_bot: bool = False
    quality_score: float = 0.5  # 0-1, higher = better quality


@dataclass  
class SentimentSignal:
    """Aggregated sentiment signal per symbol"""
    symbol: str
    net_sentiment: float  # -1 to 1, aggregated sentiment
    mention_count: int
    total_engagement: int
    confidence: float
    signal_strength: float  # Final trading signal strength
    last_updated: datetime
    
    # Source breakdown
    reddit_sentiment: float = 0.0
    twitter_sentiment: float = 0.0
    
    # Quality metrics
    spam_ratio: float = 0.0
    bot_ratio: float = 0.0
    avg_quality: float = 0.0


class SentimentConfig:
    """Configuration for sentiment monitoring"""
    
    # API Rate limits (TOS compliant)
    REDDIT_REQUESTS_PER_MINUTE = 60  # Reddit API limit
    TWITTER_REQUESTS_PER_MINUTE = 300  # Twitter API v2 limit
    
    # Exponential backoff settings
    BASE_BACKOFF_SECONDS = 1
    MAX_BACKOFF_SECONDS = 300
    BACKOFF_MULTIPLIER = 2
    
    # Data collection
    MAX_MENTIONS_PER_REQUEST = 100
    LOOKBACK_HOURS = 24  # How far back to search
    
    # Content filtering
    MIN_CONTENT_LENGTH = 10
    MAX_CONTENT_LENGTH = 1000
    MIN_ENGAGEMENT_THRESHOLD = 1  # Minimum upvotes/likes
    
    # Sentiment processing
    SIGNAL_DECAY_HOURS = 6  # Signal strength decays over 6 hours
    MIN_MENTIONS_FOR_SIGNAL = 5  # Need at least 5 mentions for signal
    CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for signals


class CryptoEntityMatcher:
    """Maps social media mentions to crypto tickers with whitelist"""
    
    def __init__(self):
        # Crypto ticker whitelist with common variations
        self.ticker_mapping = {
            # Major cryptocurrencies
            'BTC': ['bitcoin', 'btc', '$btc', '#bitcoin', '#btc'],
            'ETH': ['ethereum', 'eth', '$eth', '#ethereum', '#eth', 'ether'],
            'BNB': ['binance coin', 'bnb', '$bnb', '#bnb'],
            'XRP': ['ripple', 'xrp', '$xrp', '#xrp', '#ripple'],
            'ADA': ['cardano', 'ada', '$ada', '#cardano', '#ada'],
            'SOL': ['solana', 'sol', '$sol', '#solana', '#sol'],
            'DOT': ['polkadot', 'dot', '$dot', '#polkadot', '#dot'],
            'DOGE': ['dogecoin', 'doge', '$doge', '#dogecoin', '#doge'],
            'SHIB': ['shiba inu', 'shib', '$shib', '#shibainu', '#shib'],
            'MATIC': ['polygon', 'matic', '$matic', '#polygon', '#matic'],
            'LINK': ['chainlink', 'link', '$link', '#chainlink', '#link'],
            'UNI': ['uniswap', 'uni', '$uni', '#uniswap', '#uni'],
            'AVAX': ['avalanche', 'avax', '$avax', '#avalanche', '#avax'],
            'LTC': ['litecoin', 'ltc', '$ltc', '#litecoin', '#ltc'],
            'ATOM': ['cosmos', 'atom', '$atom', '#cosmos', '#atom'],
            
            # Add more as needed
        }
        
        # Build reverse mapping for fast lookup
        self.mention_to_ticker = {}
        for ticker, mentions in self.ticker_mapping.items():
            for mention in mentions:
                self.mention_to_ticker[mention.lower()] = ticker
        
        # Noise filtering patterns
        self.noise_patterns = [
            r'\b(buy|sell|hodl|moon|lambo|rocket|diamond hands|paper hands)\b',
            r'\b(pump|dump|manipulation|scam|rugpull)\b',
            r'\b(technical analysis|ta|chart|pattern|support|resistance)\b'
        ]
        
        self.noise_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.noise_patterns]
    
    def extract_tickers(self, content: str) -> List[Tuple[str, float]]:
        """Extract crypto tickers from content with confidence scores"""
        
        tickers = []
        content_lower = content.lower()
        
        # Find all ticker mentions
        for mention, ticker in self.mention_to_ticker.items():
            if mention in content_lower:
                # Calculate confidence based on context
                confidence = self._calculate_mention_confidence(content, mention)
                if confidence > 0.3:  # Only include high confidence matches
                    tickers.append((ticker, confidence))
        
        # Remove duplicates, keep highest confidence
        ticker_dict = {}
        for ticker, conf in tickers:
            if ticker not in ticker_dict or conf > ticker_dict[ticker]:
                ticker_dict[ticker] = conf
        
        return [(ticker, conf) for ticker, conf in ticker_dict.items()]
    
    def _calculate_mention_confidence(self, content: str, mention: str) -> float:
        """Calculate confidence that mention refers to actual crypto"""
        
        confidence = 0.5  # Base confidence
        content_lower = content.lower()
        
        # Boost confidence for financial context
        financial_keywords = ['price', 'buy', 'sell', 'trade', 'invest', 'profit', 'loss']
        for keyword in financial_keywords:
            if keyword in content_lower:
                confidence += 0.1
        
        # Boost for crypto-specific terms
        crypto_keywords = ['wallet', 'exchange', 'blockchain', 'defi', 'nft']
        for keyword in crypto_keywords:
            if keyword in content_lower:
                confidence += 0.15
        
        # Reduce confidence for obvious noise
        for noise_pattern in self.noise_regex:
            if noise_pattern.search(content):
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def is_noise_content(self, content: str) -> bool:
        """Check if content is likely noise/spam"""
        
        content_lower = content.lower()
        
        # Check for spam indicators
        spam_indicators = [
            'follow me', 'dm me', 'telegram group', 'discord server',
            'pump group', 'signal group', 'guaranteed profit',
            '100x gains', 'easy money', 'get rich quick'
        ]
        
        for indicator in spam_indicators:
            if indicator in content_lower:
                return True
        
        # Check for excessive emojis or caps
        emoji_count = len(re.findall(r'[🚀💎🌙💰📈📉]', content))
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        
        if emoji_count > 10 or caps_ratio > 0.7:
            return True
        
        return False


class SentimentAnalyzer:
    """Analyzes sentiment from social media content"""
    
    def __init__(self):
        # Simple keyword-based sentiment analysis
        # In production, use proper NLP models
        
        self.bullish_keywords = {
            'bullish': 0.8, 'moon': 0.9, 'rocket': 0.7, 'pump': 0.6,
            'buy': 0.5, 'long': 0.6, 'hodl': 0.4, 'diamond hands': 0.8,
            'to the moon': 0.9, 'all in': 0.7, 'accumulate': 0.6,
            'strong support': 0.7, 'breakout': 0.6, 'golden cross': 0.8
        }
        
        self.bearish_keywords = {
            'bearish': -0.8, 'dump': -0.7, 'crash': -0.9, 'sell': -0.5,
            'short': -0.6, 'paper hands': -0.6, 'dead cat bounce': -0.7,
            'resistance': -0.4, 'overbought': -0.5, 'death cross': -0.8,
            'rug pull': -0.9, 'scam': -0.9, 'ponzi': -0.9
        }
    
    def analyze_sentiment(self, content: str) -> Tuple[float, float]:
        """Analyze sentiment returning (score, confidence)"""
        
        content_lower = content.lower()
        scores = []
        
        # Check bullish keywords
        for keyword, score in self.bullish_keywords.items():
            if keyword in content_lower:
                scores.append(score)
        
        # Check bearish keywords  
        for keyword, score in self.bearish_keywords.items():
            if keyword in content_lower:
                scores.append(score)
        
        if not scores:
            return 0.0, 0.0  # Neutral, no confidence
        
        # Calculate average sentiment
        avg_sentiment = sum(scores) / len(scores)
        
        # Confidence based on number of sentiment indicators
        confidence = min(1.0, len(scores) * 0.3)
        
        return avg_sentiment, confidence


class SentimentMonitor:
    """Main sentiment monitoring system"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Components
        self.entity_matcher = CryptoEntityMatcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Rate limiting
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.backoff_until: Dict[str, datetime] = {}
        
        # Data storage
        self.recent_mentions: Dict[str, List[SentimentMention]] = defaultdict(list)
        self.current_signals: Dict[str, SentimentSignal] = {}
        self.last_ids: Dict[str, str] = {}  # Track last processed IDs
        
        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
        
        # Metrics
        self.total_mentions_collected = 0
        self.total_signals_generated = 0
    
    async def __aenter__(self):
        """Async context manager setup"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup"""
        if self.session:
            await self.session.close()
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent for requests"""
        return random.choice(self.user_agents)
    
    async def _check_rate_limit(self, source: str) -> bool:
        """Check and enforce rate limits"""
        
        now = datetime.now()
        
        # Check backoff period
        if source in self.backoff_until and now < self.backoff_until[source]:
            return False
        
        # Clean old requests (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        while (self.request_counts[source] and 
               self.request_counts[source][0] < minute_ago):
            self.request_counts[source].popleft()
        
        # Check rate limit
        max_requests = (self.config.REDDIT_REQUESTS_PER_MINUTE if source == 'reddit' 
                       else self.config.TWITTER_REQUESTS_PER_MINUTE)
        
        if len(self.request_counts[source]) >= max_requests:
            # Calculate exponential backoff
            backoff_count = len([t for t in self.backoff_until.values() if t > now])
            backoff_seconds = min(
                self.config.BASE_BACKOFF_SECONDS * (self.config.BACKOFF_MULTIPLIER ** backoff_count),
                self.config.MAX_BACKOFF_SECONDS
            )
            
            self.backoff_until[source] = now + timedelta(seconds=backoff_seconds)
            logger.warning(f"Rate limit hit for {source}, backing off {backoff_seconds}s")
            return False
        
        # Record request
        self.request_counts[source].append(now)
        return True
    
    async def collect_reddit_mentions(self, subreddits: List[str]) -> List[SentimentMention]:
        """Collect mentions from Reddit using official API (mock implementation)"""
        
        mentions = []
        
        # Mock implementation - replace with actual Reddit API calls
        # Use PRAW (Python Reddit API Wrapper) or direct API calls
        
        if not await self._check_rate_limit('reddit'):
            return mentions
        
        try:
            # Simulate Reddit API call
            for subreddit in subreddits:
                
                # Mock data generation for testing
                import random
                
                for i in range(random.randint(1, 5)):
                    
                    # Generate mock Reddit post
                    content_samples = [
                        "Bitcoin looking strong with recent price action and volume",
                        "Ethereum upgrade bringing major improvements to network",
                        "DOGE to the moon! Diamond hands only!",
                        "Technical analysis shows SOL breaking resistance levels",
                        "Market sentiment turning bullish on major altcoins"
                    ]
                    
                    content = random.choice(content_samples)
                    tickers = self.entity_matcher.extract_tickers(content)
                    
                    if tickers and not self.entity_matcher.is_noise_content(content):
                        
                        sentiment_score, confidence = self.sentiment_analyzer.analyze_sentiment(content)
                        
                        for ticker, ticker_confidence in tickers:
                            mention = SentimentMention(
                                mention_id=f"reddit_{subreddit}_{int(time.time())}_{i}",
                                symbol=ticker,
                                source=SentimentSource.REDDIT,
                                content=content,
                                author=f"user_{i}",
                                timestamp=datetime.now(),
                                upvotes=random.randint(1, 100),
                                comments=random.randint(0, 50),
                                sentiment_score=sentiment_score,
                                confidence=confidence * ticker_confidence,
                                quality_score=random.uniform(0.5, 0.9)
                            )
                            
                            mentions.append(mention)
                            self.total_mentions_collected += 1
        
        except Exception as e:
            logger.error(f"Error collecting Reddit mentions: {e}")
        
        return mentions
    
    async def collect_twitter_mentions(self, query_terms: List[str]) -> List[SentimentMention]:
        """Collect mentions from Twitter using official API v2 (mock implementation)"""
        
        mentions = []
        
        # Mock implementation - replace with actual Twitter API v2 calls
        # Use tweepy or direct API calls with Bearer token
        
        if not await self._check_rate_limit('twitter'):
            return mentions
        
        try:
            # Simulate Twitter API call
            for query in query_terms:
                
                # Mock data generation for testing
                import random
                
                for i in range(random.randint(1, 3)):
                    
                    content_samples = [
                        f"{query} breaking key resistance levels #crypto",
                        f"Bullish on {query} fundamentals looking strong",
                        f"Taking profits on {query} after recent pump",
                        f"{query} technical analysis showing reversal pattern",
                        f"Long term holder of {query} through all market cycles"
                    ]
                    
                    content = random.choice(content_samples)
                    tickers = self.entity_matcher.extract_tickers(content)
                    
                    if tickers and not self.entity_matcher.is_noise_content(content):
                        
                        sentiment_score, confidence = self.sentiment_analyzer.analyze_sentiment(content)
                        
                        for ticker, ticker_confidence in tickers:
                            mention = SentimentMention(
                                mention_id=f"twitter_{query}_{int(time.time())}_{i}",
                                symbol=ticker,
                                source=SentimentSource.TWITTER,
                                content=content,
                                author=f"@user_{i}",
                                timestamp=datetime.now(),
                                likes=random.randint(1, 200),
                                shares=random.randint(0, 50),
                                sentiment_score=sentiment_score,
                                confidence=confidence * ticker_confidence,
                                quality_score=random.uniform(0.4, 0.8)
                            )
                            
                            mentions.append(mention)
                            self.total_mentions_collected += 1
        
        except Exception as e:
            logger.error(f"Error collecting Twitter mentions: {e}")
        
        return mentions
    
    async def process_sentiment_data(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """Main sentiment processing pipeline"""
        
        all_mentions = []
        
        # Collect from different sources
        reddit_mentions = await self.collect_reddit_mentions(['cryptocurrency', 'bitcoin', 'ethtrader'])
        twitter_mentions = await self.collect_twitter_mentions(symbols[:10])  # Limit query terms
        
        all_mentions.extend(reddit_mentions)
        all_mentions.extend(twitter_mentions)
        
        # Store mentions
        for mention in all_mentions:
            self.recent_mentions[mention.symbol].append(mention)
        
        # Clean old mentions (older than lookback period)
        cutoff_time = datetime.now() - timedelta(hours=self.config.LOOKBACK_HOURS)
        for symbol in self.recent_mentions:
            self.recent_mentions[symbol] = [
                m for m in self.recent_mentions[symbol] if m.timestamp > cutoff_time
            ]
        
        # Generate signals
        signals = self._generate_sentiment_signals()
        self.current_signals.update(signals)
        self.total_signals_generated += len(signals)
        
        return self.current_signals
    
    def _generate_sentiment_signals(self) -> Dict[str, SentimentSignal]:
        """Generate aggregated sentiment signals"""
        
        signals = {}
        
        for symbol, mentions in self.recent_mentions.items():
            
            if len(mentions) < self.config.MIN_MENTIONS_FOR_SIGNAL:
                continue
            
            # Calculate aggregated metrics
            total_sentiment = 0.0
            total_confidence = 0.0
            total_engagement = 0
            reddit_sentiment = 0.0
            twitter_sentiment = 0.0
            reddit_count = 0
            twitter_count = 0
            spam_count = 0
            
            for mention in mentions:
                weight = mention.confidence * mention.quality_score
                total_sentiment += mention.sentiment_score * weight
                total_confidence += mention.confidence
                
                # Engagement metrics
                engagement = mention.upvotes + mention.likes + mention.comments + mention.shares
                total_engagement += engagement
                
                # Source breakdown
                if mention.source == SentimentSource.REDDIT:
                    reddit_sentiment += mention.sentiment_score * weight
                    reddit_count += 1
                elif mention.source == SentimentSource.TWITTER:
                    twitter_sentiment += mention.sentiment_score * weight
                    twitter_count += 1
                
                if mention.is_spam:
                    spam_count += 1
            
            # Calculate normalized values
            net_sentiment = total_sentiment / len(mentions) if mentions else 0.0
            avg_confidence = total_confidence / len(mentions) if mentions else 0.0
            
            if avg_confidence < self.config.CONFIDENCE_THRESHOLD:
                continue
            
            # Source-specific sentiment
            reddit_avg = reddit_sentiment / reddit_count if reddit_count > 0 else 0.0
            twitter_avg = twitter_sentiment / twitter_count if twitter_count > 0 else 0.0
            
            # Quality metrics
            spam_ratio = spam_count / len(mentions) if mentions else 0.0
            avg_quality = sum(m.quality_score for m in mentions) / len(mentions) if mentions else 0.0
            
            # Calculate final signal strength
            # Weight by engagement and quality
            engagement_weight = min(1.0, total_engagement / 1000)  # Cap at 1000 engagement
            quality_weight = max(0.1, avg_quality - spam_ratio)
            
            signal_strength = net_sentiment * avg_confidence * engagement_weight * quality_weight
            
            # Create signal
            signal = SentimentSignal(
                symbol=symbol,
                net_sentiment=net_sentiment,
                mention_count=len(mentions),
                total_engagement=total_engagement,
                confidence=avg_confidence,
                signal_strength=signal_strength,
                last_updated=datetime.now(),
                reddit_sentiment=reddit_avg,
                twitter_sentiment=twitter_avg,
                spam_ratio=spam_ratio,
                avg_quality=avg_quality
            )
            
            signals[symbol] = signal
        
        return signals
    
    def get_signal_for_symbol(self, symbol: str) -> Optional[SentimentSignal]:
        """Get current sentiment signal for symbol with time decay"""
        
        signal = self.current_signals.get(symbol)
        
        if signal:
            # Apply time decay
            hours_since_update = (datetime.now() - signal.last_updated).total_seconds() / 3600
            decay_factor = max(0.0, 1.0 - (hours_since_update / self.config.SIGNAL_DECAY_HOURS))
            
            # Apply decay to signal strength and confidence
            signal.signal_strength *= decay_factor
            signal.confidence *= decay_factor
        
        return signal
    
    def get_monitoring_metrics(self) -> dict:
        """Get sentiment monitoring metrics"""
        
        active_signals = len([s for s in self.current_signals.values() 
                            if abs(s.signal_strength) > 0.1])
        
        total_mentions = sum(len(mentions) for mentions in self.recent_mentions.values())
        
        return {
            "total_mentions_collected": self.total_mentions_collected,
            "total_signals_generated": self.total_signals_generated,
            "active_signals": active_signals,
            "current_mentions_stored": total_mentions,
            "symbols_tracked": len(self.recent_mentions),
            "rate_limit_backoffs": len(self.backoff_until)
        }


# Test function
async def test_sentiment_monitor():
    """Test sentiment monitoring system"""
    
    symbols = ['BTC', 'ETH', 'SOL', 'LINK', 'DOGE']
    
    async with SentimentMonitor() as monitor:
        
        logger.info("Testing sentiment monitoring system...")
        
        # Process sentiment data
        signals = await monitor.process_sentiment_data(symbols)
        
        # Display results
        logger.info(f"Generated {len(signals)} sentiment signals")
        
        for symbol, signal in signals.items():
            logger.info(f"{symbol}: Sentiment={signal.net_sentiment:.3f}, "
                       f"Strength={signal.signal_strength:.3f}, "
                       f"Mentions={signal.mention_count}, "
                       f"Engagement={signal.total_engagement}")
        
        # Show metrics
        metrics = monitor.get_monitoring_metrics()
        logger.info(f"Monitoring metrics: {metrics}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_sentiment_monitor())