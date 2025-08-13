"""
News Speed Advantage Agent voor Milliseconde Trading Reacties

Ultra-fast nieuws monitoring en analyse systeem dat binnen milliseconden
trading signalen genereert op basis van breaking news, sentiment shifts,
en markt-bewegende gebeurtenissen.
"""

import asyncio
import aiohttp
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import re
import hashlib
import concurrent.futures
from urllib.parse import quote_plus

try:
    import feedparser

    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logging.warning("Feedparser not available for RSS feeds")

try:
    import websocket

    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    logging.warning("Websocket not available for real-time feeds")

try:
    from textblob import TextBlob

    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logging.warning("TextBlob not available for sentiment analysis")

logger = logging.getLogger(__name__)


class NewsSource(Enum):
    """Nieuws bronnen voor monitoring"""

    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"
    DECRYPT = "decrypt"
    THE_BLOCK = "theblock"
    REUTERS_CRYPTO = "reuters_crypto"
    BLOOMBERG_CRYPTO = "bloomberg_crypto"
    TWITTER_CRYPTO = "twitter_crypto"
    REDDIT_CRYPTO = "reddit_crypto"
    TELEGRAM_CHANNELS = "telegram_channels"
    OFFICIAL_ANNOUNCEMENTS = "official_announcements"


class NewsImpact(Enum):
    """Impact level van nieuws"""

    CRITICAL = "critical"  # Extreme market impact verwacht
    HIGH = "high"  # Significante impact verwacht
    MEDIUM = "medium"  # Matige impact verwacht
    LOW = "low"  # Minimale impact verwacht
    NOISE = "noise"  # Geen relevante impact


class TradingDirection(Enum):
    """Trading richting op basis van nieuws"""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class NewsEvent:
    """Individuele nieuws gebeurtenis"""

    event_id: str
    timestamp: datetime
    source: NewsSource
    title: str
    content: str
    url: str

    # Sentiment analyse
    sentiment_score: float  # -1 (zeer negatief) tot +1 (zeer positief)
    sentiment_confidence: float  # 0-1 confidence in sentiment
    emotion_scores: Dict[str, float]  # angst, hebzucht, etc.

    # Impact assessment
    impact_level: NewsImpact
    affected_symbols: List[str]  # Crypto symbols die beïnvloed worden
    impact_timeframe: str  # "immediate", "short_term", "long_term"

    # Content analysis
    keywords: List[str]
    entities: List[str]  # Bedrijven, personen, etc.
    topics: List[str]  # "regulation", "adoption", "hack", etc.

    # Trading signals
    trading_direction: TradingDirection
    confidence_score: float  # 0-1 trading confidence
    urgency_score: float  # 0-1 urgency for action

    # Speed metrics
    processing_time_ms: float  # Tijd om nieuws te verwerken
    detection_latency_ms: float  # Tijd sinds publicatie

    # Market context
    market_correlation: float  # Correlatie met eerdere soortgelijke nieuws
    historical_accuracy: float  # Historische accuracy van soortgelijke signalen


@dataclass
class SpeedSignal:
    """Ultra-fast trading signaal gebaseerd op nieuws"""

    signal_id: str
    timestamp: datetime
    generated_in_ms: float  # Tijd om signaal te genereren

    # Source nieuws
    triggering_events: List[NewsEvent]
    signal_strength: float  # 0-1 sterkte van het signaal

    # Trading recommendation
    symbol: str
    direction: TradingDirection
    recommended_size: float  # % van portfolio
    time_horizon: str  # "1m", "5m", "15m", "1h"

    # Risk parameters
    stop_loss: Optional[float]
    take_profit: Optional[float]
    max_hold_time: timedelta

    # Speed advantage
    estimated_competitor_delay: float  # Geschatte vertraging van concurrenten
    news_propagation_speed: float  # Snelheid van nieuwsverspreiding

    # Validation
    cross_source_confirmation: bool  # Bevestiging van meerdere bronnen
    sentiment_consistency: float  # Consistentie van sentiment


class NewsSpeedAgent:
    """
    News Speed Advantage Agent voor Milliseconde Trading Reacties

    Monitort real-time nieuwsfeeds en genereert binnen milliseconden
    trading signalen voor crypto assets.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Agent state
        self.active = False
        self.last_update = None
        self.signals_generated = 0
        self.error_count = 0

        # News tracking
        self.processed_news: Dict[str, NewsEvent] = {}
        self.recent_signals: deque = deque(maxlen=1000)
        self.news_queue: asyncio.Queue = None

        # Performance tracking
        self.speed_metrics = {
            "average_processing_time_ms": 0.0,
            "fastest_signal_ms": float("inf"),
            "total_signals_generated": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "average_accuracy": 0.0,
            "speed_advantage_vs_market": 0.0,
        }

        # Configuration
        self.monitoring_interval = 0.1  # 100ms polling
        self.max_processing_time_ms = 50  # Max 50ms processing time
        self.min_confidence_threshold = 0.7
        self.max_signals_per_minute = 10

        # News sources configuration
        self.news_sources = {
            NewsSource.COINDESK: {
                "rss_url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "priority": 0.9,
                "reliability": 0.95,
                "average_delay_ms": 500,
            },
            NewsSource.COINTELEGRAPH: {
                "rss_url": "https://cointelegraph.com/rss",
                "priority": 0.8,
                "reliability": 0.90,
                "average_delay_ms": 300,
            },
            NewsSource.DECRYPT: {
                "rss_url": "https://decrypt.co/feed",
                "priority": 0.7,
                "reliability": 0.85,
                "average_delay_ms": 400,
            },
            NewsSource.THE_BLOCK: {
                "rss_url": "https://www.theblockcrypto.com/rss.xml",
                "priority": 0.95,
                "reliability": 0.98,
                "average_delay_ms": 200,
            },
        }

        # Keywords voor impact assessment
        self.critical_keywords = [
            "hack",
            "exploit",
            "breach",
            "stolen",
            "attack",
            "regulation",
            "ban",
            "illegal",
            "investigation",
            "sec",
            "cftc",
            "fed",
            "central bank",
            "adoption",
            "partnership",
            "integration",
            "listing",
            "delisting",
            "exchange",
        ]

        self.bullish_keywords = [
            "adoption",
            "partnership",
            "integration",
            "investment",
            "bullish",
            "positive",
            "growth",
            "surge",
            "rally",
            "breakthrough",
            "milestone",
            "approval",
            "green light",
        ]

        self.bearish_keywords = [
            "hack",
            "exploit",
            "crash",
            "plunge",
            "dump",
            "bearish",
            "negative",
            "decline",
            "fall",
            "drop",
            "ban",
            "illegal",
            "investigation",
            "concern",
        ]

        # Symbol mapping voor impact detection
        self.symbol_keywords = {
            "BTC/USD": ["bitcoin", "btc", "satoshi"],
            "ETH/USD": ["ethereum", "eth", "ether", "vitalik"],
            "BNB/USD": ["binance", "bnb", "cz"],
            "XRP/USD": ["ripple", "xrp", "brad garlinghouse"],
            "ADA/USD": ["cardano", "ada", "charles hoskinson"],
            "SOL/USD": ["solana", "sol", "anatoly yakovenko"],
            "AVAX/USD": ["avalanche", "avax"],
            "DOT/USD": ["polkadot", "dot", "gavin wood"],
            "MATIC/USD": ["polygon", "matic"],
            "LINK/USD": ["chainlink", "link", "sergey nazarov"],
        }

        # Thread safety
        self._lock = threading.RLock()

        # Data directory
        self.data_path = Path("data/news_speed")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize async components
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info("News Speed Agent initialized")

    def start(self):
        """Start de news speed agent"""
        if not self.active:
            self.active = True
            self.monitoring_thread = threading.Thread(
                target=self._start_async_monitoring, daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("News Speed Agent started")

    def stop(self):
        """Stop de news speed agent"""
        self.active = False
        if self.session:
            asyncio.create_task(self.session.close())
        self.logger.info("News Speed Agent stopped")

    def _start_async_monitoring(self):
        """Start async monitoring in eigen thread"""
        asyncio.run(self._async_monitoring_loop())

    async def _async_monitoring_loop(self):
        """Main async monitoring loop"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5), connector=aiohttp.TCPConnector(limit=100)
        )

        self.news_queue = asyncio.Queue(maxsize=1000)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._monitor_news_sources()),
            asyncio.create_task(self._process_news_queue()),
            asyncio.create_task(self._cleanup_old_news()),
        ]

        try:
            while self.active:
                await asyncio.sleep(0.1)  # 100ms loop

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"News monitoring error: {e}")
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            await self.session.close()

    async def _monitor_news_sources(self):
        """Monitor alle nieuws bronnen"""

        while self.active:
            try:
                # Monitor elke bron parallel
                monitoring_tasks = []

                for source, config in self.news_sources.items():
                    if config.get("rss_url"):
                        task = asyncio.create_task(self._monitor_rss_feed(source, config))
                        monitoring_tasks.append(task)

                # Wacht op alle monitoring tasks (met timeout)
                if monitoring_tasks:
                    await asyncio.wait_for(
                        asyncio.gather(*monitoring_tasks, return_exceptions=True),
                        timeout=self.monitoring_interval,
                    )

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"News source monitoring error: {e}")
                await asyncio.sleep(1)

    async def _monitor_rss_feed(self, source: NewsSource, config: Dict):
        """Monitor RSS feed van specifieke bron"""

        try:
            rss_url = config["rss_url"]

            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse RSS feed
                    if HAS_FEEDPARSER:
                        feed = feedparser.parse(content)

                        for entry in feed.entries:
                            await self._process_rss_entry(source, entry, config)

        except Exception as e:
            self.logger.error(f"RSS monitoring error for {source.value}: {e}")

    async def _process_rss_entry(self, source: NewsSource, entry: Any, config: Dict):
        """Process RSS entry en genereer nieuwsevent"""

        try:
            # Extract basic info
            title = getattr(entry, "title", "")
            content = getattr(entry, "summary", "") or getattr(entry, "description", "")
            url = getattr(entry, "link", "")
            published = getattr(entry, "published_parsed", None)

            if not title or not url:
                return

            # Create unique ID
            event_id = hashlib.md5(f"{source.value}_{url}".encode()).hexdigest()

            # Check if already processed
            if event_id in self.processed_news:
                return

            # Parse timestamp
            if published:
                timestamp = datetime(*published[:6])
            else:
                timestamp = datetime.now()

            # Calculate detection latency
            detection_latency = (datetime.now() - timestamp).total_seconds() * 1000

            # Start high-speed processing
            processing_start = time.perf_counter()

            # Fast sentiment analysis
            sentiment_score, sentiment_confidence = self._fast_sentiment_analysis(
                title + " " + content
            )

            # Quick impact assessment
            impact_level, affected_symbols = self._fast_impact_assessment(title, content)

            # Extract keywords and entities
            keywords = self._extract_keywords(title + " " + content)
            entities = self._extract_entities(title + " " + content)
            topics = self._classify_topics(title + " " + content)

            # Generate trading direction
            trading_direction, confidence_score = self._generate_trading_signal(
                sentiment_score, impact_level, keywords
            )

            processing_time = (time.perf_counter() - processing_start) * 1000

            # Create news event
            news_event = NewsEvent(
                event_id=event_id,
                timestamp=timestamp,
                source=source,
                title=title,
                content=content,
                url=url,
                sentiment_score=sentiment_score,
                sentiment_confidence=sentiment_confidence,
                emotion_scores=self._analyze_emotions(content),
                impact_level=impact_level,
                affected_symbols=affected_symbols,
                impact_timeframe=self._determine_timeframe(impact_level),
                keywords=keywords,
                entities=entities,
                topics=topics,
                trading_direction=trading_direction,
                confidence_score=confidence_score,
                urgency_score=self._calculate_urgency(impact_level, detection_latency),
                processing_time_ms=processing_time,
                detection_latency_ms=detection_latency,
                market_correlation=0.7,  # Would be calculated from historical data
                historical_accuracy=0.75,
            )

            # Add to queue for signal generation
            await self.news_queue.put(news_event)

            # Store processed news
            with self._lock:
                self.processed_news[event_id] = news_event

            self.logger.info(
                f"NEWS PROCESSED: {title[:50]}... - "
                f"Impact: {impact_level.value} - "
                f"Sentiment: {sentiment_score:.2f} - "
                f"Processed in: {processing_time:.1f}ms"
            )

        except Exception as e:
            self.logger.error(f"RSS entry processing error: {e}")

    async def _process_news_queue(self):
        """Process nieuws queue en genereer speed signals"""

        while self.active:
            try:
                # Get news event from queue (with timeout)
                news_event = await asyncio.wait_for(self.news_queue.get(), timeout=1.0)

                # Check if signal should be generated
                if self._should_generate_signal(news_event):
                    signal_start = time.perf_counter()

                    # Generate speed signal voor elk affected symbol
                    for symbol in news_event.affected_symbols:
                        speed_signal = self._generate_speed_signal(news_event, symbol)

                        if speed_signal:
                            with self._lock:
                                self.recent_signals.append(speed_signal)
                                self.signals_generated += 1

                            signal_time = (time.perf_counter() - signal_start) * 1000

                            self.logger.info(
                                f"SPEED SIGNAL: {symbol} {speed_signal.direction.value} - "
                                f"Strength: {speed_signal.signal_strength:.2f} - "
                                f"Generated in: {signal_time:.1f}ms"
                            )

                            # Update speed metrics
                            self._update_speed_metrics(signal_time)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"News queue processing error: {e}")

    def _fast_sentiment_analysis(self, text: str) -> Tuple[float, float]:
        """Ultra-fast sentiment analysis"""

        if not text:
            return 0.0, 0.0

        text_lower = text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        negative_count = sum(1 for word in self.bearish_keywords if word in text_lower)

        total_count = positive_count + negative_count

        if total_count == 0:
            return 0.0, 0.1  # Neutral with low confidence

        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_count

        # Calculate confidence based on keyword density
        word_count = len(text_lower.split())
        keyword_density = total_count / max(word_count, 1)
        confidence = min(0.95, keyword_density * 10)  # Scale density to confidence

        return sentiment_score, confidence

    def _fast_impact_assessment(self, title: str, content: str) -> Tuple[NewsImpact, List[str]]:
        """Snelle impact assessment"""

        text = (title + " " + content).lower()
        affected_symbols = []

        # Check voor kritieke keywords
        critical_count = sum(1 for word in self.critical_keywords if word in text)

        # Determine impact level
        if critical_count >= 3:
            impact_level = NewsImpact.CRITICAL
        elif critical_count >= 2:
            impact_level = NewsImpact.HIGH
        elif critical_count >= 1:
            impact_level = NewsImpact.MEDIUM
        else:
            # Check for any relevant keywords
            all_keywords = self.bullish_keywords + self.bearish_keywords
            keyword_count = sum(1 for word in all_keywords if word in text)

            if keyword_count >= 3:
                impact_level = NewsImpact.MEDIUM
            elif keyword_count >= 1:
                impact_level = NewsImpact.LOW
            else:
                impact_level = NewsImpact.NOISE

        # Determine affected symbols
        for symbol, symbol_keywords in self.symbol_keywords.items():
            for keyword in symbol_keywords:
                if keyword in text:
                    affected_symbols.append(symbol)
                    break

        # If no specific symbols, assume BTC/ETH for high impact news
        if not affected_symbols and impact_level in [NewsImpact.CRITICAL, NewsImpact.HIGH]:
            affected_symbols = ["BTC/USD", "ETH/USD"]

        return impact_level, affected_symbols

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevante keywords"""

        text_lower = text.lower()
        keywords = []

        # Check all defined keywords
        all_keywords = self.critical_keywords + self.bullish_keywords + self.bearish_keywords

        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)

        return keywords

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (bedrijven, personen)"""

        # Simple entity extraction (in production, use NER)
        entities = []

        # Common crypto entities
        crypto_entities = [
            "bitcoin",
            "ethereum",
            "binance",
            "coinbase",
            "kraken",
            "tesla",
            "microstrategy",
            "paypal",
            "visa",
            "mastercard",
            "sec",
            "cftc",
            "fed",
            "ecb",
            "biden",
            "yellen",
        ]

        text_lower = text.lower()
        for entity in crypto_entities:
            if entity in text_lower:
                entities.append(entity)

        return entities

    def _classify_topics(self, text: str) -> List[str]:
        """Classificeer nieuws topics"""

        text_lower = text.lower()
        topics = []

        topic_keywords = {
            "regulation": ["regulation", "sec", "cftc", "ban", "legal", "compliance"],
            "adoption": ["adoption", "partnership", "integration", "payment", "accept"],
            "technology": ["upgrade", "fork", "protocol", "development", "innovation"],
            "security": ["hack", "exploit", "breach", "security", "attack"],
            "market": ["price", "trading", "volume", "market", "exchange"],
            "institutional": ["institutional", "fund", "investment", "bank", "corporate"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _generate_trading_signal(
        self, sentiment_score: float, impact_level: NewsImpact, keywords: List[str]
    ) -> Tuple[TradingDirection, float]:
        """Genereer trading richting en confidence"""

        # Base confidence op impact level
        impact_confidence = {
            NewsImpact.CRITICAL: 0.9,
            NewsImpact.HIGH: 0.8,
            NewsImpact.MEDIUM: 0.6,
            NewsImpact.LOW: 0.4,
            NewsImpact.NOISE: 0.1,
        }

        base_confidence = impact_confidence[impact_level]

        # Adjust confidence based on sentiment strength
        sentiment_confidence = abs(sentiment_score) * base_confidence

        # Determine direction
        if sentiment_score > 0.6:
            direction = TradingDirection.STRONG_BUY
        elif sentiment_score > 0.2:
            direction = TradingDirection.BUY
        elif sentiment_score < -0.6:
            direction = TradingDirection.STRONG_SELL
        elif sentiment_score < -0.2:
            direction = TradingDirection.SELL
        else:
            direction = TradingDirection.NEUTRAL

        return direction, sentiment_confidence

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyseer emoties in tekst"""

        text_lower = text.lower()

        emotion_keywords = {
            "fear": ["fear", "scared", "worry", "panic", "concern", "anxious"],
            "greed": ["pump", "moon", "profit", "gains", "rich", "wealth"],
            "excitement": ["excited", "amazing", "incredible", "breakthrough"],
            "anger": ["angry", "outrage", "furious", "disappointed"],
        }

        emotions = {}
        total_words = len(text_lower.split())

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = count / max(total_words, 1)

        return emotions

    def _determine_timeframe(self, impact_level: NewsImpact) -> str:
        """Bepaal impact timeframe"""

        timeframe_map = {
            NewsImpact.CRITICAL: "immediate",
            NewsImpact.HIGH: "immediate",
            NewsImpact.MEDIUM: "short_term",
            NewsImpact.LOW: "short_term",
            NewsImpact.NOISE: "long_term",
        }

        return timeframe_map[impact_level]

    def _calculate_urgency(self, impact_level: NewsImpact, detection_latency: float) -> float:
        """Bereken urgency score"""

        # Base urgency op impact
        impact_urgency = {
            NewsImpact.CRITICAL: 1.0,
            NewsImpact.HIGH: 0.8,
            NewsImpact.MEDIUM: 0.6,
            NewsImpact.LOW: 0.3,
            NewsImpact.NOISE: 0.1,
        }

        base_urgency = impact_urgency[impact_level]

        # Reduce urgency for old news
        if detection_latency > 60000:  # > 1 minute
            latency_penalty = 0.5
        elif detection_latency > 10000:  # > 10 seconds
            latency_penalty = 0.8
        else:
            latency_penalty = 1.0

        return base_urgency * latency_penalty

    def _should_generate_signal(self, news_event: NewsEvent) -> bool:
        """Check of signal gegenereerd moet worden"""

        # Minimum confidence check
        if news_event.confidence_score < self.min_confidence_threshold:
            return False

        # Check rate limiting
        recent_signals_count = len(
            [s for s in self.recent_signals if (datetime.now() - s.timestamp).total_seconds() < 60]
        )

        if recent_signals_count >= self.max_signals_per_minute:
            return False

        # Only generate for meaningful impact
        if news_event.impact_level in [NewsImpact.NOISE, NewsImpact.LOW]:
            return False

        # Must have affected symbols
        if not news_event.affected_symbols:
            return False

        return True

    def _generate_speed_signal(self, news_event: NewsEvent, symbol: str) -> Optional[SpeedSignal]:
        """Genereer speed signal voor specifiek symbol"""

        try:
            signal_start = time.perf_counter()

            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(news_event)

            # Determine position size based on confidence and impact
            recommended_size = self._calculate_position_size(news_event, signal_strength)

            # Set time horizon based on impact
            time_horizon = self._get_time_horizon(news_event.impact_level)

            # Calculate risk parameters
            stop_loss, take_profit = self._calculate_risk_params(
                news_event.trading_direction, signal_strength
            )

            # Estimate competitor delay
            competitor_delay = self._estimate_competitor_delay(news_event)

            generation_time = (time.perf_counter() - signal_start) * 1000

            speed_signal = SpeedSignal(
                signal_id=f"speed_{symbol}_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                generated_in_ms=generation_time,
                triggering_events=[news_event],
                signal_strength=signal_strength,
                symbol=symbol,
                direction=news_event.trading_direction,
                recommended_size=recommended_size,
                time_horizon=time_horizon,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_hold_time=timedelta(hours=1),
                estimated_competitor_delay=competitor_delay,
                news_propagation_speed=news_event.detection_latency_ms,
                cross_source_confirmation=False,  # Would check multiple sources
                sentiment_consistency=news_event.sentiment_confidence,
            )

            return speed_signal

        except Exception as e:
            self.logger.error(f"Speed signal generation error: {e}")
            return None

    def _calculate_signal_strength(self, news_event: NewsEvent) -> float:
        """Bereken signal strength"""

        # Combine multiple factors
        factors = [
            news_event.confidence_score,
            abs(news_event.sentiment_score),
            news_event.urgency_score,
            news_event.sentiment_confidence,
        ]

        # Weight based on impact level
        impact_weight = {
            NewsImpact.CRITICAL: 1.0,
            NewsImpact.HIGH: 0.8,
            NewsImpact.MEDIUM: 0.6,
            NewsImpact.LOW: 0.4,
            NewsImpact.NOISE: 0.1,
        }

        base_strength = sum(factors) / len(factors)
        weighted_strength = base_strength * impact_weight[news_event.impact_level]

        return min(1.0, weighted_strength)

    def _calculate_position_size(self, news_event: NewsEvent, signal_strength: float) -> float:
        """Bereken aanbevolen positie grootte"""

        # Base size op signal strength
        base_size = signal_strength * 0.1  # Max 10% voor strongest signals

        # Adjust voor impact level
        impact_multiplier = {
            NewsImpact.CRITICAL: 1.0,
            NewsImpact.HIGH: 0.8,
            NewsImpact.MEDIUM: 0.5,
            NewsImpact.LOW: 0.2,
            NewsImpact.NOISE: 0.0,
        }

        adjusted_size = base_size * impact_multiplier[news_event.impact_level]

        return min(0.05, adjusted_size)  # Max 5% per signal

    def _get_time_horizon(self, impact_level: NewsImpact) -> str:
        """Get time horizon voor signal"""

        horizon_map = {
            NewsImpact.CRITICAL: "1m",
            NewsImpact.HIGH: "5m",
            NewsImpact.MEDIUM: "15m",
            NewsImpact.LOW: "1h",
            NewsImpact.NOISE: "1h",
        }

        return horizon_map[impact_level]

    def _calculate_risk_params(
        self, direction: TradingDirection, signal_strength: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Bereken stop loss en take profit"""

        if direction in [TradingDirection.STRONG_BUY, TradingDirection.BUY]:
            # Long position
            stop_loss = -0.02 * signal_strength  # 2% max loss for strong signals
            take_profit = 0.05 * signal_strength  # 5% max profit
        elif direction in [TradingDirection.STRONG_SELL, TradingDirection.SELL]:
            # Short position
            stop_loss = 0.02 * signal_strength
            take_profit = -0.05 * signal_strength
        else:
            return None, None

        return stop_loss, take_profit

    def _estimate_competitor_delay(self, news_event: NewsEvent) -> float:
        """Schat vertraging van concurrenten"""

        # Base delay op news source
        source_delays = {
            NewsSource.THE_BLOCK: 200,  # Fastest
            NewsSource.COINDESK: 500,  # Medium
            NewsSource.COINTELEGRAPH: 800,  # Slower
            NewsSource.DECRYPT: 1000,  # Slowest
        }

        base_delay = source_delays.get(news_event.source, 1000)

        # Add processing delay estimate
        processing_delay = 1000  # Assume 1 second for typical processing

        # Add human reaction delay
        human_delay = 5000  # 5 seconds for manual trading

        total_competitor_delay = base_delay + processing_delay + human_delay

        return total_competitor_delay

    async def _cleanup_old_news(self):
        """Cleanup oude nieuws events"""

        while self.active:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)

                with self._lock:
                    # Remove old news events
                    old_events = [
                        event_id
                        for event_id, event in self.processed_news.items()
                        if event.timestamp < cutoff_time
                    ]

                    for event_id in old_events:
                        del self.processed_news[event_id]

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

    def _update_speed_metrics(self, signal_time_ms: float):
        """Update speed performance metrics"""

        with self._lock:
            # Update average processing time
            current_avg = self.speed_metrics["average_processing_time_ms"]
            total_signals = self.speed_metrics["total_signals_generated"]

            new_avg = (current_avg * total_signals + signal_time_ms) / (total_signals + 1)
            self.speed_metrics["average_processing_time_ms"] = new_avg

            # Update fastest signal
            if signal_time_ms < self.speed_metrics["fastest_signal_ms"]:
                self.speed_metrics["fastest_signal_ms"] = signal_time_ms

            self.speed_metrics["total_signals_generated"] += 1

    def get_recent_signals(self, limit: int = 10) -> List[SpeedSignal]:
        """Get recente speed signals"""

        with self._lock:
            return list(self.recent_signals)[-limit:]

    def get_speed_summary(self) -> Dict[str, Any]:
        """Get speed performance summary"""

        with self._lock:
            recent_signals = list(self.recent_signals)[-100:]  # Last 100 signals

            if recent_signals:
                avg_generation_time = sum(s.generated_in_ms for s in recent_signals) / len(
                    recent_signals
                )

                # Count by direction
                direction_counts = {}
                for signal in recent_signals:
                    direction = signal.direction.value
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1

                return {
                    "total_signals_generated": len(recent_signals),
                    "average_generation_time_ms": avg_generation_time,
                    "fastest_signal_ms": self.speed_metrics["fastest_signal_ms"],
                    "signals_last_hour": len(
                        [
                            s
                            for s in recent_signals
                            if (datetime.now() - s.timestamp).total_seconds() < 3600
                        ]
                    ),
                    "direction_distribution": direction_counts,
                    "news_sources_active": len(self.news_sources),
                    "processed_news_count": len(self.processed_news),
                }
            else:
                return {"total_signals_generated": 0, "message": "No signals generated yet"}

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "active": self.active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "signals_generated": self.signals_generated,
            "error_count": self.error_count,
            "news_sources_configured": len(self.news_sources),
            "processed_news_count": len(self.processed_news),
            "recent_signals_count": len(self.recent_signals),
            "has_feedparser": HAS_FEEDPARSER,
            "has_websocket": HAS_WEBSOCKET,
            "has_textblob": HAS_TEXTBLOB,
            "speed_metrics": self.speed_metrics,
            "monitoring_interval_ms": self.monitoring_interval * 1000,
            "max_processing_time_ms": self.max_processing_time_ms,
        }
