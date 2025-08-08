"""
Agents Package - Specialized Data Collection Agents
Multi-agent system for crypto data gathering and analysis
"""

from .sentiment import SentimentAnalysisAgent
from .ta import TechnicalAnalysisAgent  
from .onchain import OnChainAnalysisAgent
from .scraping_core import ScrapingOrchestrator

__all__ = [
    'SentimentAnalysisAgent',
    'TechnicalAnalysisAgent', 
    'OnChainAnalysisAgent',
    'ScrapingOrchestrator'
]