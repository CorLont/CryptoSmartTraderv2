"""
Sentiment Analysis Agent
Advanced sentiment analysis from multiple sources with uncertainty quantification
"""

from .sentiment_agent import SentimentAnalysisAgent
# from .sentiment_sources import TwitterScraper, RedditScraper, NewsScraper  # Legacy imports
from .model import SentimentModel, get_sentiment_model
# SarcasmDetector temporarily disabled for workstation compatibility
from .sentiment_processor import SentimentProcessor
from .sentiment_models import SentimentEnsemble

__all__ = [
    'SentimentAnalysisAgent',
    'SentimentModel',
    'get_sentiment_model',
    'SarcasmDetector',
    'SentimentProcessor',
    'SentimentEnsemble'
]