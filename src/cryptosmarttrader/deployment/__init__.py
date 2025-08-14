"""
Deployment Module
Advanced deployment systems for safe model rollouts

Components:
- ParityValidator: Backtest-live parity validation
- CanaryManager: Canary deployment with automatic rollback
"""

from .parity_validator import ParityValidator, ParityMetrics, DriftSeverity
from .canary_manager import CanaryManager, CanaryDeployment, CanaryPhase, CanaryStatus

__all__ = [
    'ParityValidator',
    'ParityMetrics', 
    'DriftSeverity',
    'CanaryManager',
    'CanaryDeployment',
    'CanaryPhase',
    'CanaryStatus'
]