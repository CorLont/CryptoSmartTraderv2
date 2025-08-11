#!/usr/bin/env python3
"""
Sentiment Analysis Agent
Advanced sentiment analysis with uncertainty quantification and multiple sources
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from core.logging_manager import get_logger
from agents.scraping_core import AsyncScrapeClient, get_async_client

@dataclass
class SentimentResult:
    """Sentiment analysis result with uncertainty"""
    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float       # 0 to 1
    volume_mentions: int
    source_breakdown: Dict[str, float]
    uncertainty_range: tuple  # (lower_bound, upper_bound)

class SentimentAnalysisAgent:
    """Advanced sentiment analysis with multi-source aggregation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()
        self.config = config or {}
        self.scraper = AsyncScraper()
        self.rate_limiter = RateLimiter(max_requests=100, time_window=3600)
        
    async def analyze_sentiment(self, symbols: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple symbols with uncertainty quantification"""
        
        self.logger.info(f"Starting sentiment analysis for {len(symbols)} symbols")
        
        results = []
        
        for symbol in symbols:
            try:
                # Multi-source sentiment collection
                sentiment_data = await self._collect_multi_source_sentiment(symbol)
                
                # Aggregate with uncertainty quantification
                sentiment_result = self._aggregate_sentiment_with_uncertainty(symbol, sentiment_data)
                
                results.append(sentiment_result)
                
            except Exception as e:
                self.logger.error(f"Sentiment analysis failed for {symbol}: {e}")
                
                # Return neutral sentiment with low confidence for failed analysis
                results.append(SentimentResult(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    sentiment_score=0.0,
                    confidence=0.0,
                    volume_mentions=0,
                    source_breakdown={},
                    uncertainty_range=(-1.0, 1.0)
                ))
        
        self.logger.info(f"Sentiment analysis completed for {len(results)} symbols")
        return results
    
    async def _collect_multi_source_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Collect sentiment from multiple sources"""
        
        # In production, would implement actual API calls
        # For now, return realistic mock data
        sentiment_data = {
            "twitter": {
                "sentiment": np.random.normal(0.1, 0.3),  # Slightly positive bias
                "confidence": np.random.beta(6, 2),       # Higher confidence distribution
                "mentions": np.random.poisson(50)
            },
            "reddit": {
                "sentiment": np.random.normal(0.05, 0.25),
                "confidence": np.random.beta(5, 3),
                "mentions": np.random.poisson(25)
            },
            "news": {
                "sentiment": np.random.normal(0.0, 0.2),  # More neutral
                "confidence": np.random.beta(7, 2),       # Higher confidence for news
                "mentions": np.random.poisson(15)
            }
        }
        
        # Simulate API rate limiting
        await asyncio.sleep(0.1)
        
        return sentiment_data
    
    def _aggregate_sentiment_with_uncertainty(
        self, 
        symbol: str, 
        sentiment_data: Dict[str, Any]
    ) -> SentimentResult:
        """Aggregate multi-source sentiment with Bayesian uncertainty quantification"""
        
        # Extract individual sentiments and confidences
        sentiments = []
        confidences = []
        mentions = []
        source_breakdown = {}
        
        for source, data in sentiment_data.items():
            sentiment = np.clip(data["sentiment"], -1, 1)
            confidence = np.clip(data["confidence"], 0, 1)
            mention_count = max(0, data["mentions"])
            
            sentiments.append(sentiment)
            confidences.append(confidence)
            mentions.append(mention_count)
            source_breakdown[source] = sentiment
        
        if not sentiments:
            # No data available
            return SentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=0.0,
                confidence=0.0,
                volume_mentions=0,
                source_breakdown={},
                uncertainty_range=(-1.0, 1.0)
            )
        
        # Weighted aggregation by confidence and mention volume
        weights = []
        for i, (conf, ment) in enumerate(zip(confidences, mentions)):
            # Weight by confidence and log(mentions + 1) to avoid dominance by high-volume sources
            weight = conf * np.log(ment + 1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(sentiments)] * len(sentiments)
        
        # Calculate weighted sentiment
        aggregated_sentiment = sum(s * w for s, w in zip(sentiments, weights))
        
        # Calculate overall confidence using weighted harmonic mean
        if confidences:
            # Weighted harmonic mean gives more conservative confidence
            confidence_sum = sum(w / max(c, 0.001) for w, c in zip(weights, confidences))
            overall_confidence = len(confidences) / confidence_sum if confidence_sum > 0 else 0.0
        else:
            overall_confidence = 0.0
        
        # Calculate uncertainty range using confidence intervals
        sentiment_variance = sum(w * (s - aggregated_sentiment) ** 2 for s, w in zip(sentiments, weights))
        uncertainty_std = np.sqrt(sentiment_variance)
        
        # 95% confidence interval
        margin = 1.96 * uncertainty_std / np.sqrt(len(sentiments))
        lower_bound = max(-1.0, aggregated_sentiment - margin)
        upper_bound = min(1.0, aggregated_sentiment + margin)
        
        return SentimentResult(
            symbol=symbol,
            timestamp=datetime.now(),
            sentiment_score=float(aggregated_sentiment),
            confidence=float(overall_confidence),
            volume_mentions=int(sum(mentions)),
            source_breakdown=source_breakdown,
            uncertainty_range=(float(lower_bound), float(upper_bound))
        )