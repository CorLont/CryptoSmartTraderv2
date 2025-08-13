"""
Agents Module

Multi-agent cryptocurrency trading intelligence system with specialized agents
for different market analysis and trading tasks.
"""

# Import base classes and utilities
try:
    from .sentiment_agent import SentimentAgent, SentimentData, SentimentSummary

    HAS_SENTIMENT_AGENT = True
except ImportError:
    HAS_SENTIMENT_AGENT = False

try:
    from .whale_detector_agent import WhaleDetectorAgent, WhaleTransaction, WhaleMetrics

    HAS_WHALE_AGENT = True
except ImportError:
    HAS_WHALE_AGENT = False

# Make agents available
__all__ = []

if HAS_SENTIMENT_AGENT:
    __all__.extend(["SentimentAgent", "SentimentData", "SentimentSummary"])

if HAS_WHALE_AGENT:
    __all__.extend(["WhaleDetectorAgent", "WhaleTransaction", "WhaleMetrics"])

# Agent availability status
AVAILABLE_AGENTS = {"sentiment": HAS_SENTIMENT_AGENT, "whale_detector": HAS_WHALE_AGENT}
