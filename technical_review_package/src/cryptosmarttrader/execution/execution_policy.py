# CONSOLIDATED ExecutionPolicy - Single source of truth
# All execution logic moved to hard_execution_policy.py for FASE C compliance

from .hard_execution_policy import (
    HardExecutionPolicy as ExecutionPolicy,
    OrderRequest, MarketConditions, ExecutionResult,
    OrderSide, TimeInForce, ExecutionDecision,
    get_execution_policy, reset_execution_policy
)

# Export main class and components
__all__ = [
    'ExecutionPolicy',
    'OrderRequest', 'MarketConditions', 'ExecutionResult', 
    'OrderSide', 'TimeInForce', 'ExecutionDecision',
    'get_execution_policy', 'reset_execution_policy'
]

# Execution policy implementation consolidated to hard_execution_policy.py
# This file now serves as the main import point with backward compatibility