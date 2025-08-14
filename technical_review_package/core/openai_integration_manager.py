#!/usr/bin/env python3
"""
OpenAI Integration Manager - Centralized LLM operations
Manages all OpenAI interactions across the trading system with consistent interfaces
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from core.robust_openai_adapter import (
    get_openai_adapter,
    NewsImpactSchema,
    SentimentSchema,
    LLMTaskType,
    LLMConfig,
)
from core.structured_logger import get_structured_logger


class OpenAIIntegrationManager:
    """Centralized manager for all OpenAI operations in the trading system"""

    def __init__(self):
        self.logger = get_structured_logger("OpenAIIntegrationManager")
        self.adapter = get_openai_adapter()

        # Feature flags
        self.features_enabled = {
            "news_analysis": True,
            "sentiment_analysis": True,
            "market_commentary": True,
            "anomaly_detection": True,
            "feature_explanation": True,
        }

        # Performance tracking
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "fallback_requests": 0,
            "cache_hits": 0,
            "total_cost": 0.0,
        }

    async def process_news_batch(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of news items for trading signals"""

        self.logger.info(f"Processing {len(news_items)} news items")

        if not self.features_enabled["news_analysis"]:
            self.logger.warning("News analysis disabled - returning empty results")
            return []

        try:
            processed_items = []

            # Process in smaller batches to respect rate limits
            batch_size = 5
            for i in range(0, len(news_items), batch_size):
                batch = news_items[i : i + batch_size]

                # Process batch concurrently
                tasks = []
                for item in batch:
                    content = item.get("content", item.get("title", ""))
                    if content.strip():
                        tasks.append(self._process_single_news_item(item, content))

                if tasks:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in batch_results:
                        if isinstance(result, Exception):
                            self.logger.error(f"News processing failed: {result}")
                            continue

                        if result:
                            processed_items.append(result)

                # Small delay between batches
                if i + batch_size < len(news_items):
                    await asyncio.sleep(0.5)

            self.logger.info(f"Successfully processed {len(processed_items)} news items")
            return processed_items

        except Exception as e:
            self.logger.error(f"News batch processing failed: {e}")
            return []

    async def _process_single_news_item(
        self, news_item: Dict[str, Any], content: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single news item"""

        try:
            # Track request
            self.usage_stats["total_requests"] += 1

            # Analyze news impact
            impact_analysis = await self.adapter.analyze_news_impact(content)

            # Track success
            if impact_analysis.confidence > 0.5:  # Threshold for successful analysis
                self.usage_stats["successful_requests"] += 1
            else:
                self.usage_stats["fallback_requests"] += 1

            # Create enhanced news item
            enhanced_item = {
                "original_item": news_item,
                "ai_analysis": {
                    "sentiment": impact_analysis.sentiment,
                    "impact_magnitude": impact_analysis.impact_magnitude,
                    "confidence": impact_analysis.confidence,
                    "half_life_hours": impact_analysis.half_life_hours,
                    "affected_symbols": impact_analysis.affected_symbols,
                    "key_factors": impact_analysis.key_factors,
                    "impact_timeline": impact_analysis.impact_timeline,
                    "reasoning": impact_analysis.reasoning,
                },
                "trading_signal": self._generate_trading_signal(impact_analysis),
                "processed_timestamp": datetime.utcnow().isoformat(),
            }

            return enhanced_item

        except Exception as e:
            self.logger.error(f"Single news processing failed: {e}")
            return None

    def _generate_trading_signal(self, impact_analysis: NewsImpactSchema) -> Dict[str, Any]:
        """Generate actionable trading signal from news analysis"""

        # Signal strength based on impact and confidence
        signal_strength = impact_analysis.impact_magnitude * impact_analysis.confidence

        # Direction mapping
        direction_map = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}

        # Timeline to signal urgency
        urgency_map = {
            "immediate": "HIGH",
            "short_term": "MEDIUM",
            "medium_term": "LOW",
            "long_term": "VERY_LOW",
        }

        # Generate signal
        signal = {
            "direction": direction_map.get(impact_analysis.sentiment, "HOLD"),
            "strength": signal_strength,
            "urgency": urgency_map.get(impact_analysis.impact_timeline, "LOW"),
            "confidence": impact_analysis.confidence,
            "duration_estimate_hours": impact_analysis.half_life_hours * 2,
            "affected_symbols": impact_analysis.affected_symbols,
            "actionable": signal_strength > 0.3 and impact_analysis.confidence > 0.6,
        }

        return signal

    async def analyze_market_sentiment(self, market_texts: List[str]) -> Dict[str, Any]:
        """Analyze overall market sentiment from multiple text sources"""

        if not self.features_enabled["sentiment_analysis"]:
            return self._get_default_sentiment()

        try:
            self.logger.info(f"Analyzing sentiment from {len(market_texts)} sources")

            # Process all texts
            sentiment_tasks = [
                self.adapter.await get_sentiment_analyzer().analyze_text(text) for text in market_texts if text.strip()
            ]

            if not sentiment_tasks:
                return self._get_default_sentiment()

            sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)

            # Aggregate successful results
            valid_results = [
                result for result in sentiment_results if isinstance(result, SentimentSchema)
            ]

            if not valid_results:
                return self._get_default_sentiment()

            # Calculate aggregate metrics
            avg_sentiment = sum(r.sentiment_score for r in valid_results) / len(valid_results)
            avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results)

            # Collect all emotions and key phrases
            all_emotions = []
            all_phrases = []
            for result in valid_results:
                all_emotions.extend(result.emotions)
                all_phrases.extend(result.key_phrases)

            # Get most common emotions
            emotion_counts = {}
            for emotion in all_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            # Overall sentiment classification
            if avg_sentiment > 0.3:
                overall_sentiment = "bullish"
            elif avg_sentiment < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"

            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": avg_sentiment,
                "confidence": avg_confidence,
                "sources_analyzed": len(valid_results),
                "top_emotions": [emotion for emotion, count in top_emotions],
                "key_phrases": list(set(all_phrases))[:10],  # Top 10 unique phrases
                "market_relevance": sum(r.market_relevance for r in valid_results)
                / len(valid_results),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Market sentiment analysis failed: {e}")
            return self._get_default_sentiment()

    async def generate_market_commentary(self, market_data: Dict[str, Any]) -> str:
        """Generate AI-powered market commentary"""

        if not self.features_enabled["market_commentary"]:
            return "Market commentary disabled."

        try:
            # Create market summary prompt
            prompt = self._create_market_commentary_prompt(market_data)

            # Use generic OpenAI call for commentary (not structured data)
            if self.adapter.client:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = self.adapter.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional cryptocurrency market analyst. Provide concise, objective market commentary in Dutch.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )

                commentary = response.choices[0].message.content
                self.logger.info("Generated AI market commentary")
                return commentary
            else:
                return self._generate_fallback_commentary(market_data)

        except Exception as e:
            self.logger.error(f"Market commentary generation failed: {e}")
            return self._generate_fallback_commentary(market_data)

    def _create_market_commentary_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create prompt for market commentary"""

        btc_price = market_data.get("btc_price", 45000)
        total_volume = market_data.get("total_volume", 50000000000)
        bullish_coins = market_data.get("bullish_count", 0)
        bearish_coins = market_data.get("bearish_count", 0)

        prompt = f"""
Geef een korte marktanalyse van de huidige crypto markt situatie:

- Bitcoin prijs: ${btc_price:,.0f}
- Totaal volume 24h: ${total_volume / 1e9:.1f}B
- Bullish coins: {bullish_coins}
- Bearish coins: {bearish_coins}

Schrijf een professionele, korte analyse van maximaal 150 woorden in het Nederlands.
Focus op: trend, volume, marktsentiment en outlook voor de komende dagen.
"""

        return prompt

    def _generate_fallback_commentary(self, market_data: Dict[str, Any]) -> str:
        """Generate basic commentary when AI fails"""

        bullish_count = market_data.get("bullish_count", 0)
        bearish_count = market_data.get("bearish_count", 0)

        if bullish_count > bearish_count:
            sentiment = "overwegend positief"
        elif bearish_count > bullish_count:
            sentiment = "overwegend negatief"
        else:
            sentiment = "gemengd"

        return f"""
De crypto markt toont momenteel een {sentiment} sentiment met {bullish_count} stijgende 
en {bearish_count} dalende coins. Het volume blijft binnen normale ranges. 
Traders wordt geadviseerd voorzichtig te zijn en risicomanagement toe te passen.
"""

    async def detect_market_anomalies(self, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unusual market patterns using AI"""

        if not self.features_enabled["anomaly_detection"]:
            return {"anomalies_detected": False, "anomalies": []}

        try:
            # Create anomaly detection prompt
            prompt = f"""
Analyze the following market data for unusual patterns or anomalies:

{json.dumps(market_features, indent=2)}

Look for:
- Unusual volume spikes
- Price movements that don't match historical patterns
- Cross-asset correlation anomalies
- Liquidity issues

Respond with JSON:
{{
    "anomalies_detected": true/false,
    "anomalies": [
        {{
            "type": "volume_spike/price_anomaly/correlation_break/liquidity_issue",
            "severity": 0.0-1.0,
            "description": "explanation",
            "affected_assets": ["BTC", "ETH", ...],
            "confidence": 0.0-1.0
        }}
    ],
    "overall_risk_level": "low/medium/high"
}}
"""

            if self.adapter.client:
                response = self.adapter.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert market anomaly detector. Analyze data and provide structured JSON output.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

                return json.loads(response.choices[0].message.content)
            else:
                return self._detect_simple_anomalies(market_features)

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {"anomalies_detected": False, "anomalies": [], "error": str(e)}

    def _detect_simple_anomalies(self, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Simple anomaly detection fallback"""

        anomalies = []

        # Check for volume spikes
        volume = market_features.get("total_volume", 0)
        if volume > 100e9:  # $100B threshold
            anomalies.append(
                {
                    "type": "volume_spike",
                    "severity": 0.6,
                    "description": "Unusually high trading volume detected",
                    "affected_assets": ["MARKET"],
                    "confidence": 0.7,
                }
            )

        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "overall_risk_level": "medium" if anomalies else "low",
        }

    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Default sentiment when analysis fails"""
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "sources_analyzed": 0,
            "top_emotions": ["neutral"],
            "key_phrases": [],
            "market_relevance": 0.5,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of OpenAI integration"""

        adapter_status = self.adapter.get_status()

        return {
            "adapter_status": adapter_status,
            "features_enabled": self.features_enabled,
            "usage_statistics": self.usage_stats,
            "health_score": self._calculate_health_score(adapter_status),
            "recommendations": self._get_health_recommendations(adapter_status),
        }

    def _calculate_health_score(self, adapter_status: Dict[str, Any]) -> float:
        """Calculate overall health score (0-1)"""

        score = 1.0

        # API availability
        if not adapter_status["api_available"]:
            score *= 0.3  # Major penalty for no API

        # Circuit breaker state
        if adapter_status["circuit_breaker_state"] == "OPEN":
            score *= 0.2
        elif adapter_status["circuit_breaker_state"] == "HALF_OPEN":
            score *= 0.7

        # Cost limits
        if not adapter_status["cost_limit_ok"]:
            score *= 0.5

        # Success rate
        total_requests = self.usage_stats["total_requests"]
        if total_requests > 0:
            success_rate = self.usage_stats["successful_requests"] / total_requests
            score *= success_rate

        return score

    def _get_health_recommendations(self, adapter_status: Dict[str, Any]) -> List[str]:
        """Get health improvement recommendations"""

        recommendations = []

        if not adapter_status["api_available"]:
            recommendations.append("Configure OpenAI API key")

        if adapter_status["circuit_breaker_state"] == "OPEN":
            recommendations.append("Circuit breaker is open - check API connectivity")

        if not adapter_status["cost_limit_ok"]:
            recommendations.append("Hourly cost limit exceeded - review usage")

        fallback_rate = self.usage_stats["fallback_requests"] / max(
            self.usage_stats["total_requests"], 1
        )
        if fallback_rate > 0.3:
            recommendations.append("High fallback usage - check API reliability")

        return recommendations


# Global instance
_global_integration_manager: Optional[OpenAIIntegrationManager] = None


def get_openai_integration_manager() -> OpenAIIntegrationManager:
    """Get or create global OpenAI integration manager"""
    global _global_integration_manager

    if _global_integration_manager is None:
        _global_integration_manager = OpenAIIntegrationManager()

    return _global_integration_manager
