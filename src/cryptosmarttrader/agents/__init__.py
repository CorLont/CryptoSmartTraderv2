"""
Agents module - Multi-agent system components

Contains all agent implementations for the distributed cryptocurrency
trading intelligence system.
"""

try:
    from .technical_agent import TechnicalAgent
except ImportError:
    TechnicalAgent = None

__all__ = [
    "TechnicalAgent"
]