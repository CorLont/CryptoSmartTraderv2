"""
CryptoSmartTrader Agents Module
Multi-agent cryptocurrency trading intelligence system.
"""

# Import main agent classes to avoid import issues
try:
    from .agents.ensemble_voting_agent import EnsembleVotingAgent
except ImportError:
    # Fallback if ensemble voting agent has issues
    EnsembleVotingAgent = None

# Expose main classes
__all__ = [
    'EnsembleVotingAgent',
]