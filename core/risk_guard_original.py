"""
Risk Guard - Alias to canonical implementation
This file redirects to the single source of truth in src/cryptosmarttrader/core
"""

# Import from canonical source
from src.cryptosmarttrader.risk.risk_guard import *

# Maintain backward compatibility
from src.cryptosmarttrader.risk.risk_guard import RiskGuard as RiskGuardSystem
