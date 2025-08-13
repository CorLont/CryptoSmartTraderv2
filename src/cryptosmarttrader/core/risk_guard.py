"""
Risk Guard - Alias to canonical implementation
This file redirects to the single source of truth in risk module
"""

# Import from canonical source
from ..risk.risk_guard import *

# Maintain backward compatibility
from ..risk.risk_guard import RiskGuard as RiskGuardSystem
