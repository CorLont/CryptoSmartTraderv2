#!/usr/bin/env python3
"""
Event Impact Scoring System
LLM-powered news analysis with impact scoring and half-life decay modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import json
import re
from openai import OpenAI
import os
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NewsEvent:
    """News event with impact analysis"""
    timestamp: datetime
    headline: str
    content: str
    source: str
    symbols_mentioned: List[str]
    event_type: str  # 'listing', 'partnership', 'unlock', 'regulatory', 'technical', 'market'
    impact_direction: str  # 'bullish', 'bearish', 'neutral'
    impact_magnitude: float  # 0-1 scale
    confidence: float  # 0-1 confidence in analysis
    affected_market_cap: float  # Total market cap of affected coins
    half_life_hours: float  # Expected impact decay half-life
    decay_model: str  # 'exponential', 'linear', 'step'

@dataclass
class EventImpactScore:
    """Calculated impact score for a symbol"""
    symbol: str
    timestamp: datetime
    total_impact_score: float
    bullish_events_score: float
    bearish_events_score: float
    net_sentiment_score: float
    event_count_24h: int
    major_events: List[str]  # Headlines of major events
    decay_adjusted_score: float
    confidence: float

class NewsCollector:
    """Collects crypto news from multiple sources"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sources = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'cryptonews': 'https://cryptonews.net/news/rss.xml'
        }

    async def collect_latest_news(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect latest crypto news from multiple sources"""

        all_news = []

        # For now, simulate news collection with realistic examples
        # In production, this would parse actual RSS feeds

        simulated_news = self._generate_simulated_news(hours_back)
        all_news.extend(simulated_news)

        return all_news

    def _generate_simulated_news(self, hours_back: int) -> List[Dict[str, Any]]:
        """Generate realistic simulated news for testing"""

        now = datetime.utcnow()

        simulated_events = [
            {
                'timestamp': now - timedelta(hours=2),
                'headline': 'Coinbase Announces ADA Trading Support for Institutional Clients',
                'content': 'Major US exchange Coinbase has announced institutional trading support for Cardano (ADA), expanding access for professional traders and institutional investors.',
                'source': 'coindesk',
                'symbols': ['ADA']
            },
            {
                'timestamp': now - timedelta(hours=6),
                'headline': 'Ethereum Foundation Announces Major Token Unlock Schedule',
                'content': 'The Ethereum Foundation has published its token release schedule, with 500,000 ETH tokens set to unlock over the next 6 months.',
                'source': 'cointelegraph',
                'symbols': ['ETH']
            },
            {
                'timestamp': now - timedelta(hours=12),
                'headline': 'Solana Network Experiences Partial Outage, Recovery Underway',
                'content': 'The Solana blockchain experienced a partial network outage affecting transaction processing. Validators are working on recovery measures.',
                'source': 'cryptonews',
                'symbols': ['SOL']
            },
            {
                'timestamp': now - timedelta(hours=18),
                'headline': 'BlackRock Files for Bitcoin ETF with Enhanced Security Features',
                'content': 'Asset management giant BlackRock has filed for a Bitcoin ETF featuring enhanced security protocols and institutional-grade custody solutions.',
                'source': 'coindesk',
                'symbols': ['BTC']
            },
            {
                'timestamp': now - timedelta(hours=3),
                'headline': 'Binance Partners with Major European Bank for Crypto Services',
                'content': 'Binance has announced a strategic partnership with a leading European bank to provide cryptocurrency services to retail and institutional clients.',
                'source': 'cointelegraph',
                'symbols': ['BNB', 'BTC', 'ETH']
            },
            {
                'timestamp': now - timedelta(hours=8),
                'headline': 'Polygon Announces zkEVM Mainnet Launch Date',
                'content': 'Polygon has set a firm launch date for its zkEVM mainnet, promising Ethereum-compatible zero-knowledge rollup capabilities.',
                'source': 'cryptonews',
                'symbols': ['MATIC']
            }
        ]

        return simulated_events

class LLMEventAnalyzer:
    """Uses LLM to analyze news events for impact scoring"""

    def __init__(self):
        self.client = None
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client if API key available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.logger.warning("OpenAI API key not found, using fallback analysis")

    async def analyze_event_impact(
        self,
        headline: str,
        content: str,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Analyze news event for crypto impact using LLM"""

        if self.client:
            return await self._llm_analysis(headline, content, symbols)
        else:
            return self._fallback_analysis(headline, content, symbols)

    async def _llm_analysis(
        self,
        headline: str,
        content: str,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Perform LLM-based analysis using OpenAI"""

        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(headline, content, symbols)

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Latest OpenAI model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency market analyst. Analyze news events for their potential price impact on mentioned cryptocurrencies. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(headline, content, symbols)

    def _create_analysis_prompt(
        self,
        headline: str,
        content: str,
        symbols: List[str]
    ) -> str:
        """Create analysis prompt for LLM"""

        return f"""
Analyze the following cryptocurrency news event:

HEADLINE: {headline}
CONTENT: {content}
SYMBOLS: {', '.join(symbols)}

Provide analysis in this JSON format:
{{
    "event_type": "listing|partnership|unlock|regulatory|technical|market",
    "impact_direction": "bullish|bearish|neutral",
    "impact_magnitude": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of analysis",
    "affected_symbols": ["SYM1", "SYM2"],
    "half_life_hours": 1-168,
    "decay_model": "exponential|linear|step",
    "market_impact_factors": {{
        "liquidity_effect": 0.0-1.0,
        "adoption_potential": 0.0-1.0,
        "regulatory_risk": 0.0-1.0,
        "technical_significance": 0.0-1.0
    }}
}}

Analysis guidelines:
- Listings/partnerships: Usually bullish, 12-48h half-life
- Unlocks: Usually bearish, 168h+ half-life
- Technical issues: Bearish, 24-72h half-life
- Regulatory: Variable direction, 72-168h half-life
- Consider market cap, trading volume, and ecosystem effects
- Magnitude: 0.1=minor, 0.5=moderate, 0.8+=major impact
"""

    def _fallback_analysis(
        self,
        headline: str,
        content: str,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Fallback analysis using keyword matching"""

        text = (headline + " " + content).lower()

        # Event type classification
        event_type = "market"  # default
        if any(word in text for word in ['list', 'listing', 'trading', 'exchange']):
            event_type = "listing"
        elif any(word in text for word in ['partner', 'partnership', 'collaborate', 'integration']):
            event_type = "partnership"
        elif any(word in text for word in ['unlock', 'release', 'vesting', 'emission']):
            event_type = "unlock"
        elif any(word in text for word in ['regulation', 'regulatory', 'sec', 'legal', 'ban']):
            event_type = "regulatory"
        elif any(word in text for word in ['outage', 'bug', 'hack', 'exploit', 'technical']):
            event_type = "technical"

        # Impact direction
        bullish_words = ['partnership', 'listing', 'adoption', 'launch', 'upgrade', 'positive', 'growth']
        bearish_words = ['outage', 'hack', 'ban', 'regulatory', 'unlock', 'negative', 'down']

        bullish_count = sum(1 for word in bullish_words if word in text)
        bearish_count = sum(1 for word in bearish_words if word in text)

        if bullish_count > bearish_count:
            impact_direction = "bullish"
            magnitude = min(0.8, (bullish_count - bearish_count) * 0.2 + 0.2)
        elif bearish_count > bullish_count:
            impact_direction = "bearish"
            magnitude = min(0.8, (bearish_count - bullish_count) * 0.2 + 0.2)
        else:
            impact_direction = "neutral"
            magnitude = 0.1

        # Half-life based on event type
        half_life_map = {
            'listing': 24,
            'partnership': 48,
            'unlock': 168,
            'regulatory': 72,
            'technical': 48,
            'market': 12
        }

        return {
            'event_type': event_type,
            'impact_direction': impact_direction,
            'impact_magnitude': magnitude,
            'confidence': 0.6,  # Lower confidence for fallback
            'reasoning': f"Keyword-based analysis: {event_type} event with {impact_direction} sentiment",
            'affected_symbols': symbols,
            'half_life_hours': half_life_map[event_type],
            'decay_model': 'exponential',
            'market_impact_factors': {
                'liquidity_effect': 0.5,
                'adoption_potential': 0.5,
                'regulatory_risk': 0.3,
                'technical_significance': 0.4
            }
        }

class EventImpactCalculator:
    """Calculates time-decayed impact scores from events"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.event_history = {}  # symbol -> list of events

    def add_event(self, event: NewsEvent):
        """Add event to history for impact calculation"""

        for symbol in event.symbols_mentioned:
            if symbol not in self.event_history:
                self.event_history[symbol] = []

            self.event_history[symbol].append(event)

    def calculate_current_impact(self, symbol: str, current_time: datetime = None) -> EventImpactScore:
        """Calculate current impact score for symbol"""

        if current_time is None:
            current_time = datetime.utcnow()

        if symbol not in self.event_history:
            return EventImpactScore(
                symbol=symbol,
                timestamp=current_time,
                total_impact_score=0,
                bullish_events_score=0,
                bearish_events_score=0,
                net_sentiment_score=0,
                event_count_24h=0,
                major_events=[],
                decay_adjusted_score=0,
                confidence=0
            )

        events = self.event_history[symbol]

        # Filter events within reasonable timeframe (7 days)
        cutoff_time = current_time - timedelta(days=7)
        recent_events = [e for e in events if e.timestamp >= cutoff_time]

        # Calculate time-decayed impacts
        bullish_score = 0
        bearish_score = 0
        total_confidence = 0
        events_24h = 0
        major_events = []

        for event in recent_events:
            # Calculate time decay
            hours_elapsed = (current_time - event.timestamp).total_seconds() / 3600
            decay_factor = self._calculate_decay(hours_elapsed, event.half_life_hours, event.decay_model)

            # Calculate decayed impact
            decayed_impact = event.impact_magnitude * decay_factor * event.confidence

            # Add to appropriate bucket
            if event.impact_direction == 'bullish':
                bullish_score += decayed_impact
            elif event.impact_direction == 'bearish':
                bearish_score += decayed_impact

            total_confidence += event.confidence * decay_factor

            # Count recent events
            if hours_elapsed <= 24:
                events_24h += 1

            # Track major events
            if event.impact_magnitude >= 0.6:
                major_events.append(event.headline)

        # Calculate net sentiment and total impact
        net_sentiment = bullish_score - bearish_score
        total_impact = bullish_score + bearish_score

        # Normalize confidence
        avg_confidence = total_confidence / len(recent_events) if recent_events else 0

        return EventImpactScore(
            symbol=symbol,
            timestamp=current_time,
            total_impact_score=total_impact,
            bullish_events_score=bullish_score,
            bearish_events_score=bearish_score,
            net_sentiment_score=net_sentiment,
            event_count_24h=events_24h,
            major_events=major_events[:5],  # Top 5 major events
            decay_adjusted_score=net_sentiment,
            confidence=avg_confidence
        )

    def _calculate_decay(self, hours_elapsed: float, half_life_hours: float, decay_model: str) -> float:
        """Calculate time decay factor"""

        if hours_elapsed <= 0:
            return 1.0

        if decay_model == 'exponential':
            # Exponential decay: impact halves every half_life_hours
            return 0.5 ** (hours_elapsed / half_life_hours)

        elif decay_model == 'linear':
            # Linear decay: impact reaches zero at 2 * half_life_hours
            fade_time = 2 * half_life_hours
            return max(0, 1 - hours_elapsed / fade_time)

        elif decay_model == 'step':
            # Step decay: full impact until half_life, then halves
            if hours_elapsed <= half_life_hours:
                return 1.0
            elif hours_elapsed <= 2 * half_life_hours:
                return 0.5
            else:
                return 0.1

        else:
            # Default to exponential
            return 0.5 ** (hours_elapsed / half_life_hours)

    def get_top_impact_symbols(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get symbols with highest current impact scores"""

        current_time = datetime.utcnow()
        symbol_scores = []

        for symbol in self.event_history.keys():
            impact_score = self.calculate_current_impact(symbol, current_time)
            symbol_scores.append((symbol, abs(impact_score.decay_adjusted_score)))

        # Sort by absolute impact (highest first)
        symbol_scores.sort(key=lambda x: x[1], reverse=True)

        return symbol_scores[:n]

class EventImpactSystem:
    """Complete event impact scoring system"""

    def __init__(self, update_interval_hours: int = 1):
        self.news_collector = NewsCollector()
        self.llm_analyzer = LLMEventAnalyzer()
        self.impact_calculator = EventImpactCalculator()
        self.update_interval = update_interval_hours
        self.logger = logging.getLogger(__name__)

        # System state
        self.last_update = None
        self.processed_events = set()

    async def update_event_impacts(self) -> Dict[str, Any]:
        """Update event impacts with latest news"""

        # Collect latest news
        hours_back = 24 if self.last_update is None else self.update_interval + 1
        news_items = await self.news_collector.collect_latest_news(hours_back)

        new_events_processed = 0

        for news_item in news_items:
            # Skip if already processed
            news_id = f"{news_item['headline']}_{news_item['timestamp']}"
            if news_id in self.processed_events:
                continue

            # Analyze event impact
            analysis = await self.llm_analyzer.analyze_event_impact(
                news_item['headline'],
                news_item['content'],
                news_item['symbols']
            )

            # Create NewsEvent object
            event = NewsEvent(
                timestamp=news_item['timestamp'],
                headline=news_item['headline'],
                content=news_item['content'],
                source=news_item['source'],
                symbols_mentioned=analysis['affected_symbols'],
                event_type=analysis['event_type'],
                impact_direction=analysis['impact_direction'],
                impact_magnitude=analysis['impact_magnitude'],
                confidence=analysis['confidence'],
                affected_market_cap=0,  # Would be calculated from real market data
                half_life_hours=analysis['half_life_hours'],
                decay_model=analysis['decay_model']
            )

            # Add to impact calculator
            self.impact_calculator.add_event(event)
            self.processed_events.add(news_id)
            new_events_processed += 1

        self.last_update = datetime.utcnow()

        return {
            'news_items_collected': len(news_items),
            'new_events_processed': new_events_processed,
            'total_processed_events': len(self.processed_events),
            'last_update': self.last_update
        }

    def get_symbol_impact_scores(self, symbols: List[str]) -> Dict[str, EventImpactScore]:
        """Get current impact scores for specified symbols"""

        impact_scores = {}

        for symbol in symbols:
            impact_scores[symbol] = self.impact_calculator.calculate_current_impact(symbol)

        return impact_scores

    def create_event_features(self, symbols: List[str]) -> pd.DataFrame:
        """Create ML features from event impact scores"""

        impact_scores = self.get_symbol_impact_scores(symbols)
        features_data = []

        for symbol, impact_score in impact_scores.items():
            features = {
                'symbol': symbol,
                'event_total_impact': impact_score.total_impact_score,
                'event_bullish_score': impact_score.bullish_events_score,
                'event_bearish_score': impact_score.bearish_events_score,
                'event_net_sentiment': impact_score.net_sentiment_score,
                'event_count_24h': impact_score.event_count_24h,
                'event_decay_adjusted': impact_score.decay_adjusted_score,
                'event_confidence': impact_score.confidence,
                'has_major_events': len(impact_score.major_events) > 0,
                'event_intensity': impact_score.event_count_24h / 24,  # Events per hour
                'event_sentiment_ratio': (
                    impact_score.bullish_events_score / (impact_score.bearish_events_score + 0.001)
                    if impact_score.bearish_events_score > 0 else
                    impact_score.bullish_events_score
                )
            }

            features_data.append(features)

        return pd.DataFrame(features_data)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""

        top_impact = self.impact_calculator.get_top_impact_symbols(5)

        return {
            'last_update': self.last_update,
            'total_events_processed': len(self.impact_calculator.event_history),
            'symbols_with_events': len(self.impact_calculator.event_history),
            'top_impact_symbols': [{'symbol': s, 'impact': round(i, 3)} for s, i in top_impact],
            'llm_available': self.llm_analyzer.client is not None
        }

async def main():
    """Test event impact system"""

    system = EventImpactSystem()

    # Update with latest events
    update_result = await system.update_event_impacts()
    print(f"Update result: {update_result}")

    # Get impact scores for test symbols
    test_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC']
    impact_scores = system.get_symbol_impact_scores(test_symbols)

    print("\\nImpact Scores:")
    for symbol, score in impact_scores.items():
        print(f"{symbol}: Net={score.net_sentiment_score:.3f}, "
              f"Events={score.event_count_24h}, "
              f"Confidence={score.confidence:.3f}")

    # Create features
    features_df = system.create_event_features(test_symbols)
    print(f"\\nGenerated features DataFrame with {len(features_df)} rows")
    print(features_df[['symbol', 'event_net_sentiment', 'event_count_24h', 'event_confidence']].to_string())

    # System status
    status = system.get_system_status()
    print(f"\\nSystem status: {status}")

if __name__ == "__main__":
    asyncio.run(main())
