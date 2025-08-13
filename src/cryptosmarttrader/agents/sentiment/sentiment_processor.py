#!/usr/bin/env python3
"""
Sentiment Processor - Core sentiment analysis processing
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.structured_logger import get_logger
from .model import SentimentModel

class SentimentProcessor:
    """Core sentiment processing with FinBERT and calibration"""

    def __init__(self):
        self.logger = get_logger("SentimentProcessor")
        self.model = SentimentModel()
        self.initialized = False

    async def initialize(self):
        """Initialize the sentiment processor"""
        try:
            self.logger.info("Initializing sentiment processor")
            await self.model.initialize()
            self.initialized = True
            self.logger.info("Sentiment processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Sentiment processor initialization failed: {e}")
            raise

    async def process_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts for sentiment analysis"""

        try:
            if not self.initialized:
                await self.initialize()

            # Process texts in batch
            batch_result = await self.model.predict_batch(texts)

            # Convert to standardized format
            results = []
            for result in batch_result.results:
                processed_result = {
                    "text": result.text,
                    "sentiment_score": result.score,
                    "confidence": result.confidence,
                    "positive_prob": result.prob_pos,
                    "negative_prob": result.prob_neg,
                    "neutral_prob": result.prob_neutral,
                    "sarcasm_detected": result.sarcasm > 0.5,
                    "processing_time": result.processing_time
                }
                results.append(processed_result)

            return results

        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return []

    async def analyze_market_sentiment(self, news_data: List[Dict], social_data: List[Dict] = None) -> Dict[str, Any]:
        """Analyze market sentiment from news and social media"""

        try:
            all_texts = []
            text_sources = []

            # Collect news texts
            for news_item in news_data:
                text = news_item.get('title', '') + ' ' + news_item.get('summary', '')
                if text.strip():
                    all_texts.append(text.strip())
                    text_sources.append({'type': 'news', 'data': news_item})

            # Collect social media texts
            if social_data:
                for social_item in social_data:
                    text = social_item.get('text', '')
                    if text.strip():
                        all_texts.append(text.strip())
                        text_sources.append({'type': 'social', 'data': social_item})

            if not all_texts:
                return {
                    "overall_sentiment": 0.0,
                    "confidence": 0.0,
                    "total_analyzed": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
                }

            # Process all texts
            sentiment_results = await self.process_texts(all_texts)

            # Calculate aggregated sentiment
            if sentiment_results:
                scores = [r["sentiment_score"] for r in sentiment_results]
                confidences = [r["confidence"] for r in sentiment_results]

                # Weighted average by confidence
                weighted_scores = [s * c for s, c in zip(scores, confidences)]
                total_weight = sum(confidences)

                overall_sentiment = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
                average_confidence = np.mean(confidences)

                # Sentiment distribution
                positive_count = len([s for s in scores if s > 0.1])
                negative_count = len([s for s in scores if s < -0.1])
                neutral_count = len(scores) - positive_count - negative_count

                sentiment_distribution = {
                    "positive": positive_count / len(scores),
                    "negative": negative_count / len(scores),
                    "neutral": neutral_count / len(scores)
                }

                return {
                    "overall_sentiment": overall_sentiment,
                    "confidence": average_confidence,
                    "total_analyzed": len(sentiment_results),
                    "sentiment_distribution": sentiment_distribution,
                    "detailed_results": sentiment_results
                }
            else:
                return {
                    "overall_sentiment": 0.0,
                    "confidence": 0.0,
                    "total_analyzed": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
                }

        except Exception as e:
            self.logger.error(f"Market sentiment analysis failed: {e}")
            return {
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "total_analyzed": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "error": str(e)
            }
