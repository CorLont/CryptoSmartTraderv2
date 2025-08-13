"""
CryptoSmartTrader V2 - Trading Agents Module

Intelligent trading agents for market analysis and decision making:
- Ensemble Voting Agent: Advanced ML prediction aggregation
- Early Mover System: Fast market opportunity detection
- Technical Analysis: Advanced charting and indicators
- Sentiment Analysis: Market sentiment and news analysis
- Portfolio Optimization: Risk-adjusted portfolio management
"""

# Import core agents for easy access
from .ensemble_voting_agent import EnsembleVotingAgent
from .early_mover_system import EarlyMoverSystem
from .listing_detection_agent import ListingDetectionAgent
from .enhanced_technical_agent import EnhancedTechnicalAgent
from .enhanced_sentiment_agent import EnhancedSentimentAgent
from .portfolio_optimizer_agent import PortfolioOptimizerAgent

__all__ = [
    "EnsembleVotingAgent",
    "EarlyMoverSystem",
    "ListingDetectionAgent",
    "EnhancedTechnicalAgent",
    "EnhancedSentimentAgent",
    "PortfolioOptimizerAgent",
]
