#!/usr/bin/env python3
"""
Backward compatibility aliases for Sentiment Analysis
DEPRECATED: Use src.cryptosmarttrader.analysis.enterprise_sentiment_analysis directly
"""

import asyncio
import warnings
from src.cryptosmarttrader.analysis.enterprise_sentiment_analysis import get_sentiment_analyzer

# Deprecated functions - use enterprise framework instead
async def analyze_sentiment(text, use_llm=False):
    warnings.warn("analyze_sentiment is deprecated. Use get_sentiment_analyzer().analyze_text(text)", DeprecationWarning)
    result = await get_sentiment_analyzer().analyze_text(text)
    return {
        "sentiment_score": result.overall_sentiment_score,
        "confidence": result.overall_confidence,
        "sentiment": result.sentiment_strength.value
    }

def analyze_sentiment_sync(text, use_llm=False):
    warnings.warn("Synchronous sentiment analysis is deprecated. Use async get_sentiment_analyzer().analyze_text(text)", DeprecationWarning)
    return asyncio.run(analyze_sentiment(text, use_llm))

class SentimentModel:
    def __init__(self, use_llm=False):
        warnings.warn("SentimentModel is deprecated. Use get_sentiment_analyzer() directly", DeprecationWarning)
        self.analyzer = get_sentiment_analyzer()
    
    async def predict_single(self, text):
        result = await self.analyzer.analyze_text(text)
        return {
            "sentiment_score": result.overall_sentiment_score,
            "confidence": result.overall_confidence
        }
