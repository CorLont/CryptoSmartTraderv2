#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Analysis Package
Enterprise-grade technical and sentiment analysis frameworks
"""

from .enterprise_technical_analysis import (
    EnterpriseTechnicalAnalyzer,
    get_technical_analyzer,
    IndicatorResult,
    IndicatorType,
    TAConfig,
    RSIIndicator,
    MACDIndicator,
    BollingerBandsIndicator
)

from .enterprise_sentiment_analysis import (
    EnterpriseSentimentAnalyzer,
    get_sentiment_analyzer,
    SentimentAnalysisResult,
    SentimentSignal,
    SentimentSource,
    SentimentStrength,
    SentimentConfig
)

__all__ = [
    # Technical Analysis
    'EnterpriseTechnicalAnalyzer',
    'get_technical_analyzer', 
    'IndicatorResult',
    'IndicatorType',
    'TAConfig',
    'RSIIndicator',
    'MACDIndicator',
    'BollingerBandsIndicator',
    
    # Sentiment Analysis
    'EnterpriseSentimentAnalyzer',
    'get_sentiment_analyzer',
    'SentimentAnalysisResult',
    'SentimentSignal',
    'SentimentSource',
    'SentimentStrength',
    'SentimentConfig'
]

# Version info
__version__ = "2.0.0"
__author__ = "CryptoSmartTrader V2 Team"
__description__ = "Enterprise-grade cryptocurrency analysis frameworks"