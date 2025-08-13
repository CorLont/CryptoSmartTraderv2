#!/usr/bin/env python3
"""
LLM-Powered News Impact Scoring System
Uses GPT-4o for structured analysis of crypto news impact with magnitude and half-life estimation
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from openai import OpenAI

from core.structured_logger import get_structured_logger

@dataclass
class NewsImpactAnalysis:
    """Structured news impact analysis result"""
    sentiment: str  # "bullish", "bearish", "neutral"
    magnitude: float  # 0.0 to 1.0 scale
    half_life_hours: float  # Expected impact duration half-life
    confidence: float  # AI confidence in assessment
    key_factors: List[str]  # Main impact drivers
    affected_tokens: List[str]  # Specific tokens mentioned
    market_sectors: List[str]  # Affected market sectors
    impact_timeline: str  # "immediate", "short_term", "medium_term", "long_term"

class LLMNewsImpactScorer:
    """Advanced news impact scoring using GPT-4o with structured output"""

    def __init__(self):
        self.logger = get_structured_logger("LLMNewsImpactScorer")

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        self.client = OpenAI(api_key=api_key)

        # Impact scoring parameters
        self.impact_categories = {
            "regulatory": {"base_magnitude": 0.8, "half_life_hours": 168},  # 1 week
            "adoption": {"base_magnitude": 0.6, "half_life_hours": 72},     # 3 days
            "technical": {"base_magnitude": 0.4, "half_life_hours": 24},    # 1 day
            "market": {"base_magnitude": 0.5, "half_life_hours": 48},       # 2 days
            "partnership": {"base_magnitude": 0.3, "half_life_hours": 36},  # 1.5 days
            "security": {"base_magnitude": 0.9, "half_life_hours": 120}     # 5 days
        }

    async def analyze_news_impact(self, news_items: List[Dict[str, Any]]) -> List[NewsImpactAnalysis]:
        """Analyze impact of multiple news items"""

        self.logger.info(f"Analyzing impact of {len(news_items)} news items")

        try:
            analyses = []

            # Process news items in batches to respect rate limits
            batch_size = 5
            for i in range(0, len(news_items), batch_size):
                batch = news_items[i:i + batch_size]
                batch_results = await self._process_news_batch(batch)
                analyses.extend(batch_results)

                # Rate limiting
                if i + batch_size < len(news_items):
                    await asyncio.sleep(1)

            self.logger.info(f"Completed analysis of {len(analyses)} news items")
            return analyses

        except Exception as e:
            self.logger.error(f"News impact analysis failed: {e}")
            return []

    async def _process_news_batch(self, news_batch: List[Dict[str, Any]]) -> List[NewsImpactAnalysis]:
        """Process a batch of news items"""

        analyses = []

        for news_item in news_batch:
            try:
                analysis = await self._analyze_single_news_item(news_item)
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Failed to analyze news item: {e}")
                continue

        return analyses

    async def _analyze_single_news_item(self, news_item: Dict[str, Any]) -> Optional[NewsImpactAnalysis]:
        """Analyze impact of a single news item using GPT-4o"""

        try:
            # Extract news content
            title = news_item.get('title', '')
            content = news_item.get('content', news_item.get('summary', ''))
            source = news_item.get('source', 'unknown')
            timestamp = news_item.get('timestamp', datetime.utcnow().isoformat())

            # Create structured prompt for GPT-4o
            prompt = self._create_analysis_prompt(title, content, source)

            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Je bent een expert cryptocurrency marktanalist die nieuwsimpact analyseert. "
                        "Geef gestructureerde output in JSON formaat zoals gevraagd."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent analysis
            )

            # Parse response
            analysis_json = json.loads(response.choices[0].message.content)

            # Create structured analysis object
            analysis = NewsImpactAnalysis(
                sentiment=analysis_json.get('sentiment', 'neutral'),
                magnitude=min(1.0, max(0.0, analysis_json.get('magnitude', 0.0))),
                half_life_hours=max(1.0, analysis_json.get('half_life_hours', 24.0)),
                confidence=min(1.0, max(0.0, analysis_json.get('confidence', 0.5))),
                key_factors=analysis_json.get('key_factors', []),
                affected_tokens=analysis_json.get('affected_tokens', []),
                market_sectors=analysis_json.get('market_sectors', []),
                impact_timeline=analysis_json.get('impact_timeline', 'short_term')
            )

            self.logger.info(f"Analyzed news: {analysis.sentiment} sentiment, {analysis.magnitude:.2f} magnitude")
            return analysis

        except Exception as e:
            self.logger.error(f"Single news analysis failed: {e}")
            return None

    def _create_analysis_prompt(self, title: str, content: str, source: str) -> str:
        """Create structured prompt for news impact analysis"""

        prompt = f"""
Analyseer de impact van dit cryptocurrency nieuws op de markt:

TITEL: {title}
INHOUD: {content}
BRON: {source}

Geef je analyse in JSON formaat met de volgende structuur:

{{
    "sentiment": "bullish/bearish/neutral",
    "magnitude": 0.0-1.0 (waar 1.0 = extreme impact),
    "half_life_hours": aantal uren voordat impact halveert,
    "confidence": 0.0-1.0 (vertrouwen in deze analyse),
    "key_factors": ["factor1", "factor2", ...],
    "affected_tokens": ["BTC", "ETH", ...],
    "market_sectors": ["DeFi", "NFT", "Layer1", ...],
    "impact_timeline": "immediate/short_term/medium_term/long_term"
}}

RICHTLIJNEN:
- magnitude: 0.1-0.3 = klein nieuws, 0.4-0.6 = gemiddeld, 0.7-0.9 = groot, 1.0 = extreem
- half_life_hours: hoelang voordat marktimpact halveert (1-720 uren)
- immediate = <1 uur, short_term = 1-24 uur, medium_term = 1-7 dagen, long_term = >1 week
- Overweeg: nieuwsbron geloofwaardigheid, marktomvang, regulatoire impact, technische significantie
- Voor Nederlandse context: let op EU regelgeving, AFM uitspraken, Nederlandse crypto bedrijven

Wees conservatief met magnitude scores tenzij het echt significant nieuws is.
"""

        return prompt

    def calculate_aggregate_sentiment(self, analyses: List[NewsImpactAnalysis],
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """Calculate aggregate market sentiment from multiple news analyses"""

        try:
            if not analyses:
                return self._get_default_sentiment()

            # Filter recent analyses
            current_time = datetime.utcnow()
            recent_analyses = []

            for analysis in analyses:
                # Assume analyses are recent if no timestamp filtering needed
                recent_analyses.append(analysis)

            if not recent_analyses:
                return self._get_default_sentiment()

            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0

            for analysis in recent_analyses:
                # Weight by magnitude and confidence
                weight = analysis.magnitude * analysis.confidence

                # Convert sentiment to numeric
                if analysis.sentiment == 'bullish':
                    sentiment_value = 1.0
                elif analysis.sentiment == 'bearish':
                    sentiment_value = -1.0
                else:
                    sentiment_value = 0.0

                weighted_sentiment += sentiment_value * weight
                total_weight += weight

            # Calculate final metrics
            if total_weight > 0:
                final_sentiment_score = weighted_sentiment / total_weight
            else:
                final_sentiment_score = 0.0

            # Classify overall sentiment
            if final_sentiment_score > 0.3:
                overall_sentiment = "bullish"
            elif final_sentiment_score < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"

            # Calculate confidence and impact metrics
            avg_confidence = sum(a.confidence for a in recent_analyses) / len(recent_analyses)
            max_magnitude = max(a.magnitude for a in recent_analyses)
            avg_magnitude = sum(a.magnitude for a in recent_analyses) / len(recent_analyses)

            # Estimate overall market impact
            market_impact_score = avg_magnitude * avg_confidence

            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': final_sentiment_score,
                'confidence': avg_confidence,
                'market_impact_score': market_impact_score,
                'max_individual_magnitude': max_magnitude,
                'average_magnitude': avg_magnitude,
                'total_news_items': len(recent_analyses),
                'bullish_count': sum(1 for a in recent_analyses if a.sentiment == 'bullish'),
                'bearish_count': sum(1 for a in recent_analyses if a.sentiment == 'bearish'),
                'neutral_count': sum(1 for a in recent_analyses if a.sentiment == 'neutral'),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Aggregate sentiment calculation failed: {e}")
            return self._get_default_sentiment()

    def extract_impact_signals(self, analyses: List[NewsImpactAnalysis]) -> Dict[str, Any]:
        """Extract actionable trading signals from news impact analyses"""

        try:
            if not analyses:
                return {'signals': [], 'signal_strength': 'NONE'}

            signals = []

            for analysis in analyses:
                # Only consider high-confidence, high-magnitude news
                if analysis.confidence >= 0.7 and analysis.magnitude >= 0.4:

                    signal = {
                        'type': 'news_impact',
                        'direction': analysis.sentiment,
                        'strength': analysis.magnitude * analysis.confidence,
                        'expected_duration_hours': analysis.half_life_hours * 2,  # Double half-life for full impact
                        'affected_tokens': analysis.affected_tokens,
                        'key_factors': analysis.key_factors,
                        'timeline': analysis.impact_timeline
                    }

                    # Add specific signal recommendations
                    if analysis.sentiment == 'bullish' and analysis.magnitude > 0.6:
                        signal['recommendation'] = 'STRONG_BUY_SIGNAL'
                    elif analysis.sentiment == 'bearish' and analysis.magnitude > 0.6:
                        signal['recommendation'] = 'STRONG_SELL_SIGNAL'
                    elif analysis.sentiment in ['bullish', 'bearish'] and analysis.magnitude > 0.4:
                        signal['recommendation'] = f'MODERATE_{analysis.sentiment.upper()}_SIGNAL'
                    else:
                        signal['recommendation'] = 'WATCH_SIGNAL'

                    signals.append(signal)

            # Calculate overall signal strength
            if not signals:
                signal_strength = 'NONE'
            else:
                avg_strength = sum(s['strength'] for s in signals) / len(signals)
                if avg_strength >= 0.7:
                    signal_strength = 'VERY_STRONG'
                elif avg_strength >= 0.5:
                    signal_strength = 'STRONG'
                elif avg_strength >= 0.3:
                    signal_strength = 'MODERATE'
                else:
                    signal_strength = 'WEAK'

            return {
                'signals': signals,
                'signal_strength': signal_strength,
                'total_signals': len(signals),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Signal extraction failed: {e}")
            return {'signals': [], 'signal_strength': 'NONE'}

    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Default sentiment when analysis fails"""
        return {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'market_impact_score': 0.0,
            'max_individual_magnitude': 0.0,
            'average_magnitude': 0.0,
            'total_news_items': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

# Example usage and testing
async def test_news_impact_scorer():
    """Test the news impact scoring system"""

    scorer = LLMNewsImpactScorer()

    # Sample news items
    test_news = [
        {
            'title': 'Bitcoin ETF Approval by SEC Expected This Week',
            'content': 'Multiple sources indicate that the SEC is preparing to approve the first Bitcoin ETF, which could bring institutional investment.',
            'source': 'CoinDesk',
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'title': 'Ethereum Merge Successfully Completed',
            'content': 'The Ethereum network has successfully transitioned to Proof of Stake, reducing energy consumption by 99%.',
            'source': 'Ethereum Foundation',
            'timestamp': datetime.utcnow().isoformat()
        }
    ]

    # Analyze news impact
    analyses = await scorer.analyze_news_impact(test_news)

    # Calculate aggregate sentiment
    aggregate = scorer.calculate_aggregate_sentiment(analyses)

    # Extract signals
    signals = scorer.extract_impact_signals(analyses)

    return {
        'individual_analyses': analyses,
        'aggregate_sentiment': aggregate,
        'trading_signals': signals
    }

if __name__ == "__main__":
    # Run test
    result = asyncio.run(test_news_impact_scorer())
    print(json.dumps(result, indent=2, default=str))
