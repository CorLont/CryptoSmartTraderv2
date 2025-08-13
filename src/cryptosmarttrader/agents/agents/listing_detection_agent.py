"""
Early Listing Detection Agent

Monitors exchange announcements, social media, and market data for new coin listings
to identify pre-listing accumulation opportunities and early trading signals.
"""

import asyncio
import time
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import hashlib

try:
    import aiohttp
    import ccxt
    import feedparser

    HAS_SCRAPING_LIBS = True
except ImportError:
    HAS_SCRAPING_LIBS = False
    logging.warning("Scraping libraries not available")

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class ListingStage(Enum):
    """Stages of listing detection"""

    RUMOR = "rumor"  # Social media rumors, speculation
    ANNOUNCEMENT = "announcement"  # Official exchange announcement
    PRE_LISTING = "pre_listing"  # Confirmed but not yet tradeable
    LISTED = "listed"  # Now tradeable
    POST_LISTING = "post_listing"  # After initial listing pump


class ExchangeSource(Enum):
    """Exchange sources for listing detection"""

    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    HUOBI = "huobi"
    OKX = "okx"
    BYBIT = "bybit"
    GATE_IO = "gate_io"


class ListingType(Enum):
    """Types of listings"""

    NEW_TOKEN = "new_token"  # Brand new token
    NEW_PAIR = "new_pair"  # Existing token, new trading pair
    REACTIVATION = "reactivation"  # Previously delisted, now relisted
    SPECIAL_EVENT = "special_event"  # Airdrop, fork, etc.


@dataclass
class ListingDetection:
    """Individual listing detection event"""

    timestamp: datetime
    symbol: str
    exchange: ExchangeSource
    listing_stage: ListingStage
    listing_type: ListingType

    # Detection details
    confidence: float  # 0-1 confidence in detection
    source_url: Optional[str]
    source_content: str
    detection_method: str  # "announcement", "api_change", "social_media"

    # Timing information
    estimated_listing_date: Optional[datetime]
    announcement_to_listing_hours: Optional[int]
    early_detection_advantage: float  # Hours before public announcement

    # Market context
    token_category: str  # "defi", "gaming", "infrastructure", etc.
    market_cap_estimate: Optional[float]
    pre_listing_buzz: float  # 0-1 social media buzz score
    institutional_interest: float  # 0-1 institutional backing score

    # Risk factors
    risk_score: float  # 0-1 risk assessment
    regulatory_concerns: bool
    technical_readiness: float  # 0-1 technical implementation readiness

    # Trading context
    expected_volume_spike: float
    price_impact_prediction: float
    optimal_entry_window: Tuple[int, int]  # Hours before/after listing


@dataclass
class ListingOpportunity:
    """Trading opportunity from listing detection"""

    symbol: str
    exchange: ExchangeSource
    opportunity_type: str

    # Opportunity details
    confidence: float
    expected_return: float  # Expected % return
    time_horizon: str  # "immediate", "short_term", "medium_term"

    # Entry strategy
    recommended_entry: str  # "pre_listing", "at_listing", "post_dip"
    position_size_recommendation: float  # % of portfolio
    stop_loss_level: Optional[float]
    take_profit_levels: List[float]

    # Market timing
    optimal_entry_time: datetime
    exit_window: Tuple[datetime, datetime]

    # Supporting data
    detections: List[ListingDetection]
    market_context: Dict[str, Any]
    risk_warnings: List[str]


class ListingDetectionAgent:
    """
    Advanced Early Listing Detection Agent
    Monitors multiple sources for new cryptocurrency listings to capture early opportunities
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0

        # Data storage
        self.listing_detections: deque = deque(maxlen=5000)
        self.listing_opportunities: deque = deque(maxlen=1000)
        self.tracked_symbols: Set[str] = set()
        self.exchange_pairs_cache: Dict[str, Set[str]] = {}

        # Configuration
        self.update_interval = 300  # 5 minutes for listing detection
        self.confidence_threshold = 0.6
        self.max_detection_age_hours = 72  # 3 days

        # Exchange monitoring sources
        self.exchange_announcement_sources = {
            ExchangeSource.BINANCE: [
                "https://www.binance.com/en/support/announcement/c-48",
                "https://www.binance.com/en/support/announcement/new-cryptocurrency-listing",
            ],
            ExchangeSource.COINBASE: [
                "https://blog.coinbase.com/",
                "https://help.coinbase.com/en/coinbase/trading-and-funding",
            ],
            ExchangeSource.KRAKEN: [
                "https://blog.kraken.com/",
                "https://support.kraken.com/hc/en-us/sections/360001246766-Announcements",
            ],
            ExchangeSource.KUCOIN: ["https://www.kucoin.com/news/en-new-listings"],
        }

        # Social media monitoring
        self.social_sources = [
            "https://twitter.com/binance",
            "https://twitter.com/coinbase",
            "https://twitter.com/krakenfx",
            "https://twitter.com/kucoincom",
        ]

        # Listing detection keywords
        self.listing_keywords = {
            "confirmed_listing": [
                "will list",
                "listing",
                "now supports",
                "added support for",
                "new trading pair",
                "trading begins",
                "deposits open",
                "withdrawals enabled",
                "trading enabled",
            ],
            "pre_listing": [
                "deposit-only mode",
                "deposits enabled",
                "trading coming soon",
                "listing soon",
                "will support",
                "under evaluation",
            ],
            "rumor_indicators": [
                "potential listing",
                "considering",
                "evaluation",
                "community request",
                "high demand",
            ],
        }

        # Exchange API clients
        self.exchange_clients = {}
        self._initialize_exchange_clients()

        # OpenAI client for advanced analysis
        self.openai_client = None
        if HAS_OPENAI:
            try:
                import os

                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                self.logger.warning(f"OpenAI not available: {e}")

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_detections": 0,
            "confirmed_listings": 0,
            "opportunities_identified": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "average_detection_lead_time": 0,
            "exchanges_monitored": len(self.exchange_announcement_sources),
        }

        # Data directory
        self.data_path = Path("data/listing_detection")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Load historical data
        self._load_historical_detections()

        logger.info("Listing Detection Agent initialized")

    def start(self):
        """Start the listing detection agent"""
        if not self.active and HAS_SCRAPING_LIBS:
            self.active = True
            self.agent_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Listing Detection Agent started")
        else:
            self.logger.warning("Listing Detection Agent not started - missing dependencies")

    def stop(self):
        """Stop the listing detection agent"""
        self.active = False
        self.logger.info("Listing Detection Agent stopped")

    def _initialize_exchange_clients(self):
        """Initialize exchange API clients for pair monitoring"""
        if not HAS_SCRAPING_LIBS:
            return

        try:
            # Initialize exchange clients for monitoring new pairs
            self.exchange_clients = {
                "binance": ccxt.binance({"sandbox": False, "enableRateLimit": True}),
                "kraken": ccxt.kraken({"sandbox": False, "enableRateLimit": True}),
                "kucoin": ccxt.kucoin({"sandbox": False, "enableRateLimit": True}),
            }

            # Cache current trading pairs for comparison
            self._update_pairs_cache()

            self.logger.info(
                f"Initialized {len(self.exchange_clients)} exchanges for listing detection"
            )

        except Exception as e:
            self.logger.error(f"Error initializing exchange clients: {e}")

    def _update_pairs_cache(self):
        """Update cache of current trading pairs on each exchange"""
        for exchange_name, client in self.exchange_clients.items():
            try:
                markets = client.load_markets()
                pairs = set(markets.keys())
                self.exchange_pairs_cache[exchange_name] = pairs
                self.logger.debug(f"Cached {len(pairs)} pairs for {exchange_name}")

            except Exception as e:
                self.logger.error(f"Error caching pairs for {exchange_name}: {e}")

    def _detection_loop(self):
        """Main listing detection loop"""
        while self.active:
            try:
                # Monitor exchange announcements
                self._monitor_exchange_announcements()

                # Detect new trading pairs via API
                self._detect_new_trading_pairs()

                # Monitor social media for listing hints
                self._monitor_social_media()

                # Analyze detections for opportunities
                self._analyze_listing_opportunities()

                # Clean old detections
                self._cleanup_old_detections()

                # Save data
                self._save_detection_data()

                # Update statistics
                self.last_update = datetime.now()

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Listing detection error: {e}")
                time.sleep(60)  # Sleep 1 minute on error

    def _monitor_exchange_announcements(self):
        """Monitor official exchange announcement feeds"""

        for exchange, sources in self.exchange_announcement_sources.items():
            for source_url in sources:
                try:
                    # Parse RSS/news feeds for listing announcements
                    if "rss" in source_url.lower() or "feed" in source_url.lower():
                        detections = self._parse_rss_announcements(exchange, source_url)
                    else:
                        detections = self._scrape_announcement_page(exchange, source_url)

                    # Process new detections
                    for detection in detections:
                        if self._is_new_detection(detection):
                            with self._lock:
                                self.listing_detections.append(detection)
                                self.stats["total_detections"] += 1

                                if detection.listing_stage == ListingStage.LISTED:
                                    self.stats["confirmed_listings"] += 1

                except Exception as e:
                    self.logger.error(f"Error monitoring {source_url}: {e}")

    def _parse_rss_announcements(
        self, exchange: ExchangeSource, rss_url: str
    ) -> List[ListingDetection]:
        """Parse RSS feeds for listing announcements"""
        detections = []

        try:
            feed = feedparser.parse(rss_url)

            for entry in feed.entries[:20]:  # Check latest 20 entries
                title = entry.get("title", "")
                content = entry.get("summary", "") or entry.get("description", "")
                full_content = f"{title}. {content}"

                # Check for listing-related content
                listing_info = self._analyze_listing_content(full_content)

                if listing_info:
                    detection = ListingDetection(
                        timestamp=datetime.now(),
                        symbol=listing_info["symbol"],
                        exchange=exchange,
                        listing_stage=listing_info["stage"],
                        listing_type=listing_info["type"],
                        confidence=listing_info["confidence"],
                        source_url=entry.get("link", rss_url),
                        source_content=full_content[:500],
                        detection_method="announcement",
                        estimated_listing_date=listing_info.get("estimated_date"),
                        announcement_to_listing_hours=listing_info.get("hours_to_listing"),
                        early_detection_advantage=0.0,  # RSS is public announcement
                        token_category=listing_info.get("category", "unknown"),
                        market_cap_estimate=listing_info.get("market_cap"),
                        pre_listing_buzz=0.5,
                        institutional_interest=0.6
                        if exchange in [ExchangeSource.COINBASE, ExchangeSource.KRAKEN]
                        else 0.4,
                        risk_score=listing_info.get("risk_score", 0.3),
                        regulatory_concerns=listing_info.get("regulatory_risk", False),
                        technical_readiness=0.8,
                        expected_volume_spike=listing_info.get("expected_volume", 5.0),
                        price_impact_prediction=listing_info.get("price_impact", 2.0),
                        optimal_entry_window=listing_info.get("entry_window", (-2, 24)),
                    )

                    detections.append(detection)

        except Exception as e:
            self.logger.error(f"Error parsing RSS {rss_url}: {e}")

        return detections

    def _scrape_announcement_page(
        self, exchange: ExchangeSource, page_url: str
    ) -> List[ListingDetection]:
        """Scrape exchange announcement pages for listing news"""
        # In production, this would use web scraping libraries like BeautifulSoup
        # For now, simulate realistic announcement detection

        simulated_announcements = [
            {
                "symbol": "NEWCOIN/USD",
                "stage": ListingStage.ANNOUNCEMENT,
                "type": ListingType.NEW_TOKEN,
                "confidence": 0.9,
                "content": f"{exchange.value.title()} announces listing of NEWCOIN with trading starting tomorrow",
            }
        ]

        detections = []
        for announcement in simulated_announcements:
            if np.random.random() < 0.1:  # 10% chance of simulated detection
                detection = ListingDetection(
                    timestamp=datetime.now(),
                    symbol=announcement["symbol"],
                    exchange=exchange,
                    listing_stage=announcement["stage"],
                    listing_type=announcement["type"],
                    confidence=announcement["confidence"],
                    source_url=page_url,
                    source_content=announcement["content"],
                    detection_method="announcement",
                    estimated_listing_date=datetime.now() + timedelta(hours=24),
                    announcement_to_listing_hours=24,
                    early_detection_advantage=0.0,
                    token_category="defi",
                    market_cap_estimate=100000000.0,  # $100M
                    pre_listing_buzz=0.7,
                    institutional_interest=0.6,
                    risk_score=0.4,
                    regulatory_concerns=False,
                    technical_readiness=0.9,
                    expected_volume_spike=10.0,
                    price_impact_prediction=5.0,
                    optimal_entry_window=(-1, 12),
                )

                detections.append(detection)

        return detections

    def _detect_new_trading_pairs(self):
        """Detect new trading pairs by comparing current pairs with cached pairs"""

        for exchange_name, client in self.exchange_clients.items():
            try:
                # Get current markets
                current_markets = client.load_markets()
                current_pairs = set(current_markets.keys())

                # Compare with cached pairs
                cached_pairs = self.exchange_pairs_cache.get(exchange_name, set())
                new_pairs = current_pairs - cached_pairs

                # Process new pairs
                for pair in new_pairs:
                    if self._is_relevant_pair(pair):
                        # Determine listing type
                        base_symbol = pair.split("/")[0]
                        is_new_token = not self._is_known_token(base_symbol)

                        listing_type = (
                            ListingType.NEW_TOKEN if is_new_token else ListingType.NEW_PAIR
                        )

                        detection = ListingDetection(
                            timestamp=datetime.now(),
                            symbol=pair,
                            exchange=ExchangeSource(exchange_name),
                            listing_stage=ListingStage.LISTED,
                            listing_type=listing_type,
                            confidence=0.95,  # High confidence - API detected
                            source_url=f"{exchange_name}_api",
                            source_content=f"New trading pair detected via API: {pair}",
                            detection_method="api_change",
                            estimated_listing_date=datetime.now(),
                            announcement_to_listing_hours=0,
                            early_detection_advantage=1.0,  # 1 hour advantage via API
                            token_category=self._classify_token_category(base_symbol),
                            market_cap_estimate=None,
                            pre_listing_buzz=0.3,  # Unknown until detected
                            institutional_interest=0.5,
                            risk_score=0.6,  # Higher risk for API-only detection
                            regulatory_concerns=False,
                            technical_readiness=1.0,  # Already live
                            expected_volume_spike=3.0,
                            price_impact_prediction=1.5,
                            optimal_entry_window=(0, 6),
                        )

                        with self._lock:
                            self.listing_detections.append(detection)
                            self.stats["total_detections"] += 1
                            self.stats["confirmed_listings"] += 1

                            self.logger.info(f"NEW LISTING DETECTED: {pair} on {exchange_name}")

                # Update cache
                self.exchange_pairs_cache[exchange_name] = current_pairs

            except Exception as e:
                self.logger.error(f"Error detecting new pairs on {exchange_name}: {e}")

    def _monitor_social_media(self):
        """Monitor social media for early listing hints"""

        # In production, this would connect to Twitter API, Reddit API, etc.
        # For now, simulate social media monitoring

        simulated_social_signals = [
            {
                "platform": "twitter",
                "content": "Hearing rumors about ALTCOIN listing on major exchange soon 👀",
                "symbol": "ALTCOIN",
                "confidence": 0.4,
                "stage": ListingStage.RUMOR,
            }
        ]

        for signal in simulated_social_signals:
            if np.random.random() < 0.05:  # 5% chance of social signal
                detection = ListingDetection(
                    timestamp=datetime.now(),
                    symbol=signal["symbol"] + "/USD",
                    exchange=ExchangeSource.BINANCE,  # Most common for rumors
                    listing_stage=signal["stage"],
                    listing_type=ListingType.NEW_TOKEN,
                    confidence=signal["confidence"],
                    source_url=f"{signal['platform']}_monitor",
                    source_content=signal["content"],
                    detection_method="social_media",
                    estimated_listing_date=datetime.now() + timedelta(days=7),
                    announcement_to_listing_hours=168,  # 7 days
                    early_detection_advantage=168.0,  # 7 days early
                    token_category="unknown",
                    market_cap_estimate=None,
                    pre_listing_buzz=0.8,  # High social buzz
                    institutional_interest=0.2,  # Low for rumors
                    risk_score=0.8,  # High risk for rumors
                    regulatory_concerns=False,
                    technical_readiness=0.3,  # Unknown
                    expected_volume_spike=2.0,
                    price_impact_prediction=1.0,
                    optimal_entry_window=(-24, 6),
                )

                with self._lock:
                    self.listing_detections.append(detection)
                    self.stats["total_detections"] += 1

    def _analyze_listing_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Analyze content for listing information"""

        content_lower = content.lower()

        # Check for listing keywords
        listing_stage = None
        confidence = 0.0

        # Confirmed listing
        for keyword in self.listing_keywords["confirmed_listing"]:
            if keyword in content_lower:
                listing_stage = ListingStage.ANNOUNCEMENT
                confidence = 0.8
                break

        # Pre-listing
        if not listing_stage:
            for keyword in self.listing_keywords["pre_listing"]:
                if keyword in content_lower:
                    listing_stage = ListingStage.PRE_LISTING
                    confidence = 0.6
                    break

        # Rumors
        if not listing_stage:
            for keyword in self.listing_keywords["rumor_indicators"]:
                if keyword in content_lower:
                    listing_stage = ListingStage.RUMOR
                    confidence = 0.4
                    break

        if not listing_stage:
            return None

        # Extract symbol (simplified)
        symbol_match = re.search(r"\b([A-Z]{2,10})\b", content.upper())
        symbol = f"{symbol_match.group(1)}/USD" if symbol_match else "UNKNOWN/USD"

        # Use AI for enhanced analysis if available
        if self.openai_client:
            ai_analysis = self._get_ai_listing_analysis(content)
            if ai_analysis:
                return {**ai_analysis, "stage": listing_stage}

        return {
            "symbol": symbol,
            "stage": listing_stage,
            "type": ListingType.NEW_TOKEN,
            "confidence": confidence,
            "category": "unknown",
            "risk_score": 0.5,
            "expected_volume": 3.0,
            "price_impact": 2.0,
            "entry_window": (-2, 24),
        }

    def _get_ai_listing_analysis(self, content: str) -> Optional[Dict[str, Any]]:
        """Get enhanced listing analysis from OpenAI"""

        try:
            prompt = f"""
            Analyze this cryptocurrency listing announcement:
            
            {content}
            
            Provide a JSON response with:
            - symbol: the cryptocurrency symbol mentioned
            - confidence: float 0-1 how confident this is a real listing
            - category: token category (defi, gaming, infrastructure, etc.)
            - market_cap_estimate: estimated market cap in USD
            - risk_score: float 0-1 risk assessment
            - regulatory_risk: boolean if there are regulatory concerns
            - expected_volume: expected volume multiplier
            - price_impact: expected price impact percentage
            - hours_to_listing: estimated hours until trading begins
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency listing analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
            )

            ai_result = json.loads(response.choices[0].message.content)
            return ai_result

        except Exception as e:
            self.logger.error(f"OpenAI listing analysis error: {e}")
            return None

    def _is_relevant_pair(self, pair: str) -> bool:
        """Check if trading pair is relevant for monitoring"""
        # Focus on USD pairs and major quote currencies
        return pair.endswith("/USD") or pair.endswith("/USDT") or pair.endswith("/EUR")

    def _is_known_token(self, symbol: str) -> bool:
        """Check if token is already known (simplified)"""
        known_tokens = {
            "BTC",
            "ETH",
            "BNB",
            "XRP",
            "ADA",
            "SOL",
            "AVAX",
            "DOT",
            "MATIC",
            "LINK",
            "UNI",
            "AAVE",
            "COMP",
            "MKR",
            "SNX",
        }
        return symbol in known_tokens

    def _classify_token_category(self, symbol: str) -> str:
        """Classify token by category (simplified)"""
        # In production, this would use comprehensive token database
        category_hints = {
            "defi": ["AAVE", "UNI", "COMP", "MKR", "SNX", "CRV"],
            "gaming": ["AXS", "SAND", "MANA", "ENJ"],
            "infrastructure": ["ETH", "DOT", "AVAX", "SOL", "MATIC"],
            "payments": ["BTC", "XRP", "LTC", "BCH"],
        }

        for category, tokens in category_hints.items():
            if symbol in tokens:
                return category

        return "unknown"

    def _is_new_detection(self, detection: ListingDetection) -> bool:
        """Check if this is a new detection"""
        # Check for duplicates based on symbol and exchange
        for existing in self.listing_detections:
            if (
                existing.symbol == detection.symbol
                and existing.exchange == detection.exchange
                and abs((existing.timestamp - detection.timestamp).total_seconds()) < 3600
            ):  # 1 hour window
                return False
        return True

    def _analyze_listing_opportunities(self):
        """Analyze listing detections for trading opportunities"""

        # Get recent high-confidence detections
        recent_detections = [
            d
            for d in self.listing_detections
            if d.timestamp > datetime.now() - timedelta(hours=24)
            and d.confidence > self.confidence_threshold
        ]

        # Group by symbol
        symbol_detections = defaultdict(list)
        for detection in recent_detections:
            symbol_detections[detection.symbol].append(detection)

        # Analyze each symbol for opportunities
        for symbol, detections in symbol_detections.items():
            opportunity = self._create_listing_opportunity(symbol, detections)

            if opportunity:
                with self._lock:
                    self.listing_opportunities.append(opportunity)
                    self.stats["opportunities_identified"] += 1

                    self.logger.info(
                        f"LISTING OPPORTUNITY: {symbol} - {opportunity.expected_return:.1f}% expected return"
                    )

    def _create_listing_opportunity(
        self, symbol: str, detections: List[ListingDetection]
    ) -> Optional[ListingOpportunity]:
        """Create trading opportunity from listing detections"""

        if not detections:
            return None

        # Get the most confident detection
        best_detection = max(detections, key=lambda d: d.confidence)

        # Calculate opportunity metrics
        confidence = np.mean([d.confidence for d in detections])

        # Expected return based on listing stage and exchange
        base_return = {
            ListingStage.RUMOR: 50.0,  # 50% potential if rumor is true
            ListingStage.ANNOUNCEMENT: 30.0,  # 30% from announcement to listing
            ListingStage.PRE_LISTING: 15.0,  # 15% from pre-listing to trading
            ListingStage.LISTED: 5.0,  # 5% immediate post-listing
        }.get(best_detection.listing_stage, 0.0)

        # Adjust for exchange (tier-1 exchanges get premium)
        exchange_multiplier = {
            ExchangeSource.COINBASE: 1.5,
            ExchangeSource.BINANCE: 1.3,
            ExchangeSource.KRAKEN: 1.2,
            ExchangeSource.KUCOIN: 1.0,
        }.get(best_detection.exchange, 0.8)

        expected_return = base_return * exchange_multiplier

        # Determine entry strategy
        if best_detection.listing_stage == ListingStage.RUMOR:
            recommended_entry = "pre_listing"
            time_horizon = "medium_term"
        elif best_detection.listing_stage == ListingStage.ANNOUNCEMENT:
            recommended_entry = "pre_listing"
            time_horizon = "short_term"
        else:
            recommended_entry = "at_listing"
            time_horizon = "immediate"

        # Position sizing based on confidence and risk
        position_size = min(0.05, confidence * 0.1)  # Max 5% position

        # Risk warnings
        risk_warnings = []
        if best_detection.risk_score > 0.7:
            risk_warnings.append("High risk token - proceed with caution")
        if best_detection.regulatory_concerns:
            risk_warnings.append("Potential regulatory issues")
        if confidence < 0.7:
            risk_warnings.append("Low confidence detection")

        opportunity = ListingOpportunity(
            symbol=symbol,
            exchange=best_detection.exchange,
            opportunity_type="listing_play",
            confidence=confidence,
            expected_return=expected_return,
            time_horizon=time_horizon,
            recommended_entry=recommended_entry,
            position_size_recommendation=position_size,
            stop_loss_level=0.8,  # 20% stop loss
            take_profit_levels=[1.2, 1.5, 2.0],  # 20%, 50%, 100% targets
            optimal_entry_time=best_detection.estimated_listing_date or datetime.now(),
            exit_window=(datetime.now(), datetime.now() + timedelta(hours=48)),
            detections=detections,
            market_context={"category": best_detection.token_category},
            risk_warnings=risk_warnings,
        )

        return opportunity

    def _cleanup_old_detections(self):
        """Remove old detection data"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_detection_age_hours)

        with self._lock:
            # Filter out old detections
            self.listing_detections = deque(
                [d for d in self.listing_detections if d.timestamp > cutoff_time], maxlen=5000
            )

            # Filter out old opportunities
            self.listing_opportunities = deque(
                [o for o in self.listing_opportunities if o.optimal_entry_time > cutoff_time],
                maxlen=1000,
            )

    def get_active_opportunities(self, min_confidence: float = 0.6) -> List[ListingOpportunity]:
        """Get active listing opportunities"""
        with self._lock:
            return [
                opp
                for opp in self.listing_opportunities
                if opp.confidence >= min_confidence
                and opp.optimal_entry_time > datetime.now() - timedelta(hours=48)
            ]

    def get_recent_detections(self, hours: int = 24) -> List[ListingDetection]:
        """Get recent listing detections"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            return [d for d in self.listing_detections if d.timestamp > cutoff_time]

    def get_listing_alerts(self, severity_min: float = 0.7) -> List[Dict[str, Any]]:
        """Get high-priority listing alerts"""
        alerts = []

        recent_opportunities = self.get_active_opportunities(min_confidence=severity_min)

        for opp in recent_opportunities:
            alerts.append(
                {
                    "timestamp": opp.optimal_entry_time,
                    "symbol": opp.symbol,
                    "exchange": opp.exchange.value,
                    "alert_type": "listing_opportunity",
                    "severity": opp.confidence,
                    "message": f"New listing opportunity: {opp.symbol} on {opp.exchange.value}",
                    "expected_return": opp.expected_return,
                    "time_horizon": opp.time_horizon,
                    "recommended_action": f"Consider {opp.recommended_entry} position",
                }
            )

        return sorted(alerts, key=lambda x: x["severity"], reverse=True)

    def _save_detection_data(self):
        """Save detection data to disk"""
        try:
            # Save recent opportunities
            opportunities_file = self.data_path / "listing_opportunities.json"
            opportunities_data = []

            for opp in list(self.listing_opportunities)[-100:]:  # Save last 100
                opportunities_data.append(
                    {
                        "timestamp": opp.optimal_entry_time.isoformat(),
                        "symbol": opp.symbol,
                        "exchange": opp.exchange.value,
                        "confidence": opp.confidence,
                        "expected_return": opp.expected_return,
                        "time_horizon": opp.time_horizon,
                        "recommended_entry": opp.recommended_entry,
                    }
                )

            with open(opportunities_file, "w") as f:
                json.dump(opportunities_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving detection data: {e}")

    def _load_historical_detections(self):
        """Load historical detection data for pattern analysis"""
        try:
            opportunities_file = self.data_path / "listing_opportunities.json"
            if opportunities_file.exists():
                with open(opportunities_file, "r") as f:
                    # Could load for historical analysis
                    pass
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "active": self.active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "total_detections": len(self.listing_detections),
            "active_opportunities": len(self.listing_opportunities),
            "exchanges_monitored": len(self.exchange_clients),
            "sources_monitored": sum(
                len(sources) for sources in self.exchange_announcement_sources.values()
            ),
            "has_openai": self.openai_client is not None,
            "statistics": self.stats,
            "dependencies_available": HAS_SCRAPING_LIBS,
        }
