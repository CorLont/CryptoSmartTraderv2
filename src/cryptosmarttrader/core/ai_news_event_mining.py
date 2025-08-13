#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - AI-Powered News & Event Mining Engine
Advanced LLM-based news interpretation and event detection for crypto markets
"""

import asyncio
import aiohttp
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import deque
import hashlib
import time

# OpenAI for advanced analysis
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# TextBlob for basic sentiment
try:
    from textblob import TextBlob

    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Web scraping
try:
    import trafilatura

    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


@dataclass
class NewsEvent:
    """Structured news event data"""

    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    symbols_mentioned: List[str]
    sentiment_score: float
    importance_score: float
    event_type: str
    ai_analysis: Dict[str, Any]
    impact_prediction: str
    confidence: float


@dataclass
class EventMiningConfig:
    """Configuration for AI news/event mining"""

    # OpenAI settings
    openai_model: str = "gpt-4o"  # Latest model as per blueprint
    max_tokens: int = 500
    temperature: float = 0.3

    # News sources
    news_sources: List[str] = None
    social_sources: List[str] = None

    # Analysis settings
    importance_threshold: float = 0.6
    sentiment_window_hours: int = 24
    max_events_per_hour: int = 50

    # Rate limiting
    requests_per_minute: int = 20
    max_concurrent_requests: int = 5

    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = [
                "cointelegraph.com",
                "coindesk.com",
                "decrypt.co",
                "theblock.co",
                "bitcoinmagazine.com",
            ]

        if self.social_sources is None:
            self.social_sources = [
                "reddit.com/r/cryptocurrency",
                "reddit.com/r/bitcoin",
                "reddit.com/r/ethereum",
                "twitter.com/crypto",
                "github.com",
            ]


class AINewsAnalyzer:
    """Advanced AI-powered news analysis using LLMs"""

    def __init__(self, config: EventMiningConfig):
        self.config = config
        self.client = None

        if HAS_OPENAI:
            try:
                import os

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    logging.info("OpenAI client initialized for AI news analysis")
                else:
                    logging.warning("OPENAI_API_KEY not found, AI analysis disabled")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")

    async def analyze_news_content(self, title: str, content: str, url: str) -> Dict[str, Any]:
        """Analyze news content using AI"""

        if not self.client:
            return self._fallback_analysis(title, content)

        try:
            # Construct analysis prompt
            prompt = self._build_analysis_prompt(title, content, url)

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst. Analyze news content and provide structured insights about market impact, sentiment, and trading implications.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result = json.loads(response.choices[0].message.content)
            return self._validate_ai_response(result)

        except Exception as e:
            logging.error(f"AI news analysis failed: {e}")
            return self._fallback_analysis(title, content)

    def _build_analysis_prompt(self, title: str, content: str, url: str) -> str:
        """Build comprehensive analysis prompt"""

        # Truncate content if too long
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        prompt = f"""
Analyze this cryptocurrency news article and provide a structured JSON response:

Title: {title}
URL: {url}
Content: {content}

Provide analysis in this exact JSON format:
{{
    "sentiment_score": <float between -1.0 and 1.0>,
    "importance_score": <float between 0.0 and 1.0>,
    "event_type": "<string: regulation, adoption, technology, market, partnership, security, etc.>",
    "symbols_mentioned": ["<list of cryptocurrency symbols found>"],
    "impact_prediction": "<string: bullish, bearish, neutral>",
    "confidence": <float between 0.0 and 1.0>,
    "key_points": ["<list of 3-5 key insights>"],
    "market_implications": "<string: brief description of potential market impact>",
    "time_horizon": "<string: immediate, short-term, medium-term, long-term>",
    "affected_sectors": ["<list of crypto sectors affected: DeFi, NFT, Layer1, etc.>"]
}}

Focus on:
1. Accurate sentiment analysis (-1.0 = very bearish, +1.0 = very bullish)
2. Importance scoring (0.0 = irrelevant, 1.0 = market-moving news)
3. Specific cryptocurrency symbols mentioned (use standard symbols like BTC, ETH, etc.)
4. Clear impact prediction with reasoning
5. Confidence in your analysis
"""
        return prompt

    def _validate_ai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize AI response"""

        # Ensure required fields exist with defaults
        validated = {
            "sentiment_score": max(-1.0, min(1.0, response.get("sentiment_score", 0.0))),
            "importance_score": max(0.0, min(1.0, response.get("importance_score", 0.5))),
            "event_type": response.get("event_type", "general"),
            "symbols_mentioned": response.get("symbols_mentioned", []),
            "impact_prediction": response.get("impact_prediction", "neutral"),
            "confidence": max(0.0, min(1.0, response.get("confidence", 0.5))),
            "key_points": response.get("key_points", []),
            "market_implications": response.get("market_implications", ""),
            "time_horizon": response.get("time_horizon", "unknown"),
            "affected_sectors": response.get("affected_sectors", []),
        }

        # Sanitize symbols list
        if isinstance(validated["symbols_mentioned"], list):
            validated["symbols_mentioned"] = [
                symbol.upper().replace("/", "").replace("-", "")
                for symbol in validated["symbols_mentioned"]
                if isinstance(symbol, str) and len(symbol) <= 10
            ]
        else:
            validated["symbols_mentioned"] = []

        return validated

    def _fallback_analysis(self, title: str, content: str) -> Dict[str, Any]:
        """Fallback analysis without AI"""

        # Basic sentiment analysis
        sentiment_score = 0.0
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(title + " " + content)
                sentiment_score = blob.sentiment.polarity
            except Exception:
                pass

        # Simple keyword-based analysis
        bullish_keywords = [
            "pump",
            "moon",
            "bullish",
            "buy",
            "surge",
            "rally",
            "gain",
            "up",
            "rise",
            "adoption",
            "partnership",
        ]
        bearish_keywords = [
            "dump",
            "crash",
            "bearish",
            "sell",
            "drop",
            "fall",
            "down",
            "hack",
            "ban",
            "regulation",
        ]

        text_lower = (title + " " + content).lower()

        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)

        if bullish_count > bearish_count:
            impact_prediction = "bullish"
        elif bearish_count > bullish_count:
            impact_prediction = "bearish"
        else:
            impact_prediction = "neutral"

        # Extract potential symbols (very basic)
        symbols = []
        crypto_patterns = [
            r"\b(BTC|Bitcoin)\b",
            r"\b(ETH|Ethereum)\b",
            r"\b(ADA|Cardano)\b",
            r"\b(SOL|Solana)\b",
            r"\b(MATIC|Polygon)\b",
        ]

        for pattern in crypto_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.upper() not in symbols:
                    symbols.append(match.upper())

        return {
            "sentiment_score": sentiment_score,
            "importance_score": 0.5,  # Default medium importance
            "event_type": "general",
            "symbols_mentioned": symbols,
            "impact_prediction": impact_prediction,
            "confidence": 0.3,  # Low confidence for fallback
            "key_points": [title],
            "market_implications": "Basic analysis - limited insights available",
            "time_horizon": "unknown",
            "affected_sectors": [],
        }


class WebScraper:
    """Advanced web scraping for news content"""

    def __init__(self, config: EventMiningConfig):
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(config.max_concurrent_requests)
        self.last_request_times = deque(maxlen=config.requests_per_minute)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def scrape_url(self, url: str) -> Optional[Dict[str, str]]:
        """Scrape content from a URL"""

        await self._rate_limit()

        try:
            async with self.rate_limiter:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._extract_content(html, url)
                    else:
                        logging.warning(f"Failed to fetch {url}: {response.status}")
                        return None

        except Exception as e:
            logging.error(f"Scraping failed for {url}: {e}")
            return None

    def _extract_content(self, html: str, url: str) -> Dict[str, str]:
        """Extract structured content from HTML"""

        if HAS_TRAFILATURA:
            try:
                # Use trafilatura for clean content extraction
                extracted = trafilatura.extract(
                    html, include_comments=False, include_tables=False, include_formatting=False
                )

                if extracted:
                    # Try to extract title
                    title = ""
                    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()

                    return {
                        "title": title or "No title",
                        "content": extracted,
                        "source": self._extract_domain(url),
                        "url": url,
                    }
            except Exception as e:
                logging.error(f"Trafilatura extraction failed: {e}")

        # Fallback: basic HTML parsing
        return self._basic_html_extraction(html, url)

    def _basic_html_extraction(self, html: str, url: str) -> Dict[str, str]:
        """Basic HTML content extraction"""

        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "No title"

        # Remove HTML tags and extract text
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        # Take first 1000 characters as content
        content = text[:1000] + "..." if len(text) > 1000 else text

        return {"title": title, "content": content, "source": self._extract_domain(url), "url": url}

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc
        except Exception:
            return url

    async def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()

        # Remove old timestamps
        while self.last_request_times and now - self.last_request_times[0] > 60:
            self.last_request_times.popleft()

        # Wait if we're at the limit
        if len(self.last_request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self.last_request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.last_request_times.append(now)


class EventDetector:
    """Detect and classify market events"""

    def __init__(self, config: EventMiningConfig):
        self.config = config
        self.event_cache = {}
        self.cache_lock = threading.Lock()

    def detect_events(self, news_events: List[NewsEvent]) -> List[Dict[str, Any]]:
        """Detect market events from news"""

        events = []

        for news in news_events:
            if news.importance_score >= self.config.importance_threshold:
                event = self._classify_event(news)
                if event:
                    events.append(event)

        # Group related events
        grouped_events = self._group_related_events(events)

        return grouped_events

    def _classify_event(self, news: NewsEvent) -> Optional[Dict[str, Any]]:
        """Classify a news item as a market event"""

        event_types = {
            "regulatory": ["regulation", "sec", "government", "legal", "compliance", "ban"],
            "adoption": ["adoption", "partnership", "integration", "mainstream", "institutional"],
            "technical": ["upgrade", "fork", "protocol", "network", "consensus", "blockchain"],
            "market": ["price", "volume", "trading", "exchange", "listing", "delisting"],
            "security": ["hack", "exploit", "vulnerability", "security", "breach", "attack"],
            "development": ["development", "release", "update", "roadmap", "github", "code"],
        }

        content_lower = (news.title + " " + news.content).lower()

        # Find best matching event type
        best_type = "general"
        best_score = 0

        for event_type, keywords in event_types.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > best_score:
                best_score = score
                best_type = event_type

        # Create event if we have a classification
        if best_score > 0:
            return {
                "event_id": self._generate_event_id(news),
                "event_type": best_type,
                "title": news.title,
                "description": news.ai_analysis.get("market_implications", ""),
                "importance": news.importance_score,
                "sentiment": news.sentiment_score,
                "symbols_affected": news.symbols_mentioned,
                "timestamp": news.timestamp,
                "source": news.source,
                "url": news.url,
                "prediction": news.impact_prediction,
                "confidence": news.confidence,
                "time_horizon": news.ai_analysis.get("time_horizon", "unknown"),
                "affected_sectors": news.ai_analysis.get("affected_sectors", []),
            }

        return None

    def _group_related_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related events together"""

        grouped = []
        processed = set()

        for i, event in enumerate(events):
            if i in processed:
                continue

            # Find related events
            related = [event]
            processed.add(i)

            for j, other_event in enumerate(events[i + 1 :], i + 1):
                if j in processed:
                    continue

                if self._are_events_related(event, other_event):
                    related.append(other_event)
                    processed.add(j)

            # Create grouped event
            if len(related) > 1:
                grouped_event = self._merge_events(related)
                grouped.append(grouped_event)
            else:
                grouped.append(event)

        return grouped

    def _are_events_related(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Check if two events are related"""

        # Same event type
        if event1["event_type"] == event2["event_type"]:
            # Within 6 hours
            time_diff = abs((event1["timestamp"] - event2["timestamp"]).total_seconds())
            if time_diff < 6 * 3600:
                # Share symbols or similar content
                shared_symbols = set(event1["symbols_affected"]) & set(event2["symbols_affected"])
                if shared_symbols:
                    return True

        return False

    def _merge_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge related events into a single event"""

        # Use the most important event as base
        base_event = max(events, key=lambda e: e["importance"])

        # Combine data
        all_symbols = set()
        all_sectors = set()
        total_importance = 0
        avg_sentiment = 0
        sources = []

        for event in events:
            all_symbols.update(event["symbols_affected"])
            all_sectors.update(event.get("affected_sectors", []))
            total_importance += event["importance"]
            avg_sentiment += event["sentiment"]
            sources.append(event["source"])

        # Create merged event
        merged = base_event.copy()
        merged.update(
            {
                "title": f"Multiple {base_event['event_type']} events",
                "symbols_affected": list(all_symbols),
                "affected_sectors": list(all_sectors),
                "importance": min(1.0, total_importance / len(events)),
                "sentiment": avg_sentiment / len(events),
                "sources": sources,
                "event_count": len(events),
                "description": f"Grouped {len(events)} related {base_event['event_type']} events",
            }
        )

        return merged

    def _generate_event_id(self, news: NewsEvent) -> str:
        """Generate unique event ID"""
        content_hash = hashlib.md5(
            (news.title + news.url + str(news.timestamp)).encode()
        ).hexdigest()[:12]

        return f"event_{content_hash}"


class AINewsEventMiningCoordinator:
    """Main coordinator for AI-powered news and event mining"""

    def __init__(self, config: EventMiningConfig = None):
        self.config = config or EventMiningConfig()

        # Initialize components
        self.ai_analyzer = AINewsAnalyzer(self.config)
        self.event_detector = EventDetector(self.config)

        # Data storage
        self.recent_events = deque(maxlen=1000)
        self.processed_urls = set()

        self._lock = threading.Lock()

        logging.info("AI News & Event Mining Coordinator initialized")

    async def mine_news_events(self, urls: List[str]) -> Dict[str, Any]:
        """Mine and analyze news events from URLs"""

        try:
            news_events = []

            async with WebScraper(self.config) as scraper:
                # Process URLs concurrently
                tasks = []
                for url in urls:
                    if url not in self.processed_urls:
                        tasks.append(self._process_single_url(scraper, url))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, NewsEvent):
                        news_events.append(result)
                        with self._lock:
                            self.processed_urls.add(result.url)

            # Detect market events
            market_events = self.event_detector.detect_events(news_events)

            # Store recent events
            with self._lock:
                self.recent_events.extend(market_events)

            return {
                "success": True,
                "news_processed": len(news_events),
                "events_detected": len(market_events),
                "high_importance_events": len([e for e in market_events if e["importance"] > 0.8]),
                "events": market_events[:10],  # Return top 10 events
                "processing_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logging.error(f"News event mining failed: {e}")
            return {"error": str(e)}

    async def _process_single_url(self, scraper: WebScraper, url: str) -> Optional[NewsEvent]:
        """Process a single URL into a NewsEvent"""

        try:
            # Scrape content
            scraped_content = await scraper.scrape_url(url)
            if not scraped_content:
                return None

            # AI analysis
            ai_analysis = await self.ai_analyzer.analyze_news_content(
                scraped_content["title"], scraped_content["content"], url
            )

            # Create NewsEvent object
            news_event = NewsEvent(
                title=scraped_content["title"],
                content=scraped_content["content"],
                source=scraped_content["source"],
                url=url,
                timestamp=datetime.now(),
                symbols_mentioned=ai_analysis["symbols_mentioned"],
                sentiment_score=ai_analysis["sentiment_score"],
                importance_score=ai_analysis["importance_score"],
                event_type=ai_analysis["event_type"],
                ai_analysis=ai_analysis,
                impact_prediction=ai_analysis["impact_prediction"],
                confidence=ai_analysis["confidence"],
            )

            return news_event

        except Exception as e:
            logging.error(f"Failed to process URL {url}: {e}")
            return None

    def get_recent_events(
        self, hours: int = 24, min_importance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get recent events filtered by time and importance"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            filtered_events = [
                event
                for event in self.recent_events
                if (
                    event.get("timestamp", datetime.min) > cutoff_time
                    and event.get("importance", 0) >= min_importance
                )
            ]

        # Sort by importance and timestamp
        filtered_events.sort(
            key=lambda e: (e.get("importance", 0), e.get("timestamp", datetime.min)), reverse=True
        )

        return filtered_events

    def get_events_for_symbol(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get events affecting a specific symbol"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            symbol_events = [
                event
                for event in self.recent_events
                if (
                    symbol in event.get("symbols_affected", [])
                    and event.get("timestamp", datetime.min) > cutoff_time
                )
            ]

        return sorted(symbol_events, key=lambda e: e.get("timestamp", datetime.min), reverse=True)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""

        with self._lock:
            recent_count = len(
                [
                    e
                    for e in self.recent_events
                    if (datetime.now() - e.get("timestamp", datetime.min)).total_seconds() < 3600
                ]
            )

        return {
            "ai_analyzer_ready": self.ai_analyzer.client is not None,
            "openai_available": HAS_OPENAI,
            "trafilatura_available": HAS_TRAFILATURA,
            "textblob_available": HAS_TEXTBLOB,
            "recent_events_count": len(self.recent_events),
            "events_last_hour": recent_count,
            "processed_urls_count": len(self.processed_urls),
            "config": {
                "importance_threshold": self.config.importance_threshold,
                "max_events_per_hour": self.config.max_events_per_hour,
                "requests_per_minute": self.config.requests_per_minute,
            },
        }


# Singleton coordinator instance
_ai_news_coordinator = None
_coordinator_lock = threading.Lock()


def get_ai_news_event_mining_coordinator(
    config: EventMiningConfig = None,
) -> AINewsEventMiningCoordinator:
    """Get the singleton AI news event mining coordinator"""
    global _ai_news_coordinator

    with _coordinator_lock:
        if _ai_news_coordinator is None:
            _ai_news_coordinator = AINewsEventMiningCoordinator(config)

        return _ai_news_coordinator


# Test function
async def test_ai_news_mining():
    """Test the AI news mining system"""
    print("Testing AI News & Event Mining...")

    config = EventMiningConfig(importance_threshold=0.5)
    coordinator = get_ai_news_event_mining_coordinator(config)

    # Test URLs (use real crypto news URLs for testing)
    test_urls = [
        "https://cointelegraph.com/news/bitcoin-price-analysis",
        "https://coindesk.com/markets/ethereum-update",
    ]

    # Test mining
    result = await coordinator.mine_news_events(test_urls)
    print(f"Mining result: {result}")

    # Test status
    status = coordinator.get_system_status()
    print(f"System status: {status}")

    print("AI News Mining test completed!")


if __name__ == "__main__":
    asyncio.run(test_ai_news_mining())
