"""
Event & Basis Signals Module

Implements advanced event-driven trading signals based on:
- Funding rate flips and anomalies
- Open interest divergences
- Perpetual-spot basis analysis
- Cross-market arbitrage opportunities
- Event-driven mean reversion and trend continuation setups
"""

from .funding_analyzer import FundingAnalyzer, FundingEvent, FundingFlip
from .open_interest_tracker import OpenInterestTracker, OIEvent, OIDivergence
from .basis_analyzer import BasisAnalyzer, BasisSignal, BasisZScore
from .event_detector import EventDetector, MarketEvent, EventType
from .signal_generator import EventSignalGenerator, EventSignal
from .event_analytics import EventAnalytics, EventMetrics

__all__ = [
    "FundingAnalyzer",
    "FundingEvent",
    "FundingFlip",
    "OpenInterestTracker",
    "OIEvent",
    "OIDivergence",
    "BasisAnalyzer",
    "BasisSignal",
    "BasisZScore",
    "EventDetector",
    "MarketEvent",
    "EventType",
    "EventSignalGenerator",
    "EventSignal",
    "EventAnalytics",
    "EventMetrics",
]
