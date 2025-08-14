"""
Alias for ExecutionPolicy
Redirects to canonical implementation at src/cryptosmarttrader/execution/execution_policy.py
"""

# Import canonical implementation
from ..execution.execution_policy import ExecutionPolicy

# Re-export for backward compatibility
__all__ = ['ExecutionPolicy']
