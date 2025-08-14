"""
Alias for RiskGuard
Redirects to canonical implementation at src/cryptosmarttrader/risk/risk_guard.py
"""

# Import canonical implementation
from ..risk.risk_guard import RiskGuard

# Re-export for backward compatibility
__all__ = ['RiskGuard']
