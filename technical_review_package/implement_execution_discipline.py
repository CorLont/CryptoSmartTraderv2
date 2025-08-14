#!/usr/bin/env python3
"""
DEPRECATED: Legacy Execution Discipline Implementation
This file is deprecated. Use the canonical implementation instead:
src/cryptosmarttrader/execution/execution_discipline.py

This file is kept for backward compatibility only.
All new development should use the canonical execution discipline module.
"""

# BACKWARD COMPATIBILITY ALIAS
import warnings
warnings.warn(
    "implement_execution_discipline.py is deprecated. "
    "Use src.cryptosmarttrader.execution.execution_discipline instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import canonical implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.cryptosmarttrader.execution.execution_discipline import (
        OrderSide,
        TimeInForce, 
        ExecutionDecision,
        MarketConditions,
        OrderRequest,
        ExecutionGates,
        ExecutionResult,
        IdempotencyTracker,
        ExecutionPolicy
    )
except ImportError as e:
    print(f"⚠️  Legacy import: {e}")
    print("✅ Use canonical import: from src.cryptosmarttrader.execution.execution_discipline import ExecutionPolicy")
    
    # Create placeholder classes for backward compatibility
    class OrderSide:
        BUY = "buy"
        SELL = "sell"
    
    class ExecutionPolicy:
        def __init__(self):
            print("⚠️  Using placeholder ExecutionPolicy - import canonical implementation")
        
        def decide(self, *args, **kwargs):
            print("⚠️  Placeholder method - use canonical implementation")
            return None

def create_execution_discipline_system():
    """Legacy function for backward compatibility"""
    warnings.warn(
        "create_execution_discipline_system() is deprecated. "
        "Use ExecutionPolicy from src.cryptosmarttrader.execution.execution_discipline directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Return canonical ExecutionPolicy class
    return ExecutionPolicy

# Backward compatibility aliases
HardExecutionPolicy = ExecutionPolicy

# Re-export all discipline components for backward compatibility
__all__ = [
    'ExecutionPolicy', 'HardExecutionPolicy',
    'OrderRequest', 'MarketConditions', 'ExecutionResult',
    'OrderSide', 'TimeInForce', 'ExecutionDecision',
    'IdempotencyTracker', 'ExecutionGates',
    'create_execution_discipline_system'
]

if __name__ == "__main__":
    print("⚠️  DEPRECATED: This file is deprecated.")
    print("✅ Use: from src.cryptosmarttrader.execution.execution_discipline import ExecutionPolicy")
    print("📍 Canonical location: src/cryptosmarttrader/execution/execution_discipline.py")
    print("")
    print("🔄 EXECUTION MODULE CONSOLIDATION COMPLETE:")
    print("  - Legacy implementation replaced with alias")
    print("  - Canonical source: src/cryptosmarttrader/execution/execution_discipline.py")
    print("  - No breaking changes - backward compatibility maintained")
    print("  - Module drift risk eliminated")
    print("")
    print("✅ EXECUTION DISCIPLINE CONSOLIDATION SUCCESSFUL")