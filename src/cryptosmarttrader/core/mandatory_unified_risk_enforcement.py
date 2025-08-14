#!/usr/bin/env python3
"""
Mandatory Unified Risk Enforcement
Hard-wired integration dat ALLE order execution door UnifiedRiskGuard gaat
"""

import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# Import the unified risk guard
from src.cryptosmarttrader.risk.unified_risk_guard import (
    OrderSide,
    RiskDecision,
    StandardOrderRequest,
    UnifiedRiskGuard,
)


def mandatory_unified_risk_check(func):
    """
    Decorator dat ELKE order execution function forceert door UnifiedRiskGuard

    NO BYPASS MOGELIJK - elke functie die orders uitvoert MOET deze decorator hebben
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name and module for logging
        func_name = func.__name__
        module_name = func.__module__

        logger.info(f"🛡️ MANDATORY UNIFIED RISK CHECK: {module_name}.{func_name}")

        # Try to extract order from function arguments
        order_data = _extract_order_from_args(args, kwargs)

        if order_data:
            # Convert to StandardOrderRequest
            try:
                standard_order = _convert_to_standard_order(order_data)

                # Get market data if available
                market_data = _extract_market_data_from_args(args, kwargs)

                # MANDATORY: Get unified risk guard instance
                risk_guard = UnifiedRiskGuard()

                # MANDATORY: Evaluate through unified risk guard
                risk_result = risk_guard.evaluate_order(standard_order, market_data)

                # Log the risk decision
                logger.info(f"🛡️ Risk Decision: {risk_result.decision.value} - {risk_result.reason}")

                # Block execution if not approved
                if risk_result.decision == RiskDecision.REJECT:
                    raise RiskViolationError(f"Order REJECTED by UnifiedRiskGuard: {risk_result.reason}")

                elif risk_result.decision == RiskDecision.EMERGENCY_STOP:
                    raise EmergencyStopError(f"EMERGENCY STOP: {risk_result.reason}")

                elif risk_result.decision == RiskDecision.REDUCE_SIZE:
                    # Modify order size in kwargs
                    if risk_result.adjusted_size is not None:
                        logger.warning(f"🛡️ Order size reduced: {standard_order.size} → {risk_result.adjusted_size}")
                        _update_order_size_in_args(args, kwargs, risk_result.adjusted_size)

                # Risk check passed - execute original function
                logger.info(f"✅ Risk check PASSED for {func_name}")
                return func(*args, **kwargs)

            except (RiskViolationError, EmergencyStopError):
                # Re-raise risk-specific errors
                raise
            except ValueError as e:
                logger.error(f"❌ Value error in risk evaluation for {func_name}: {e}")
                raise RiskEvaluationError(f"Invalid order data: {e}") from e
            except TypeError as e:
                logger.error(f"❌ Type error in risk evaluation for {func_name}: {e}")
                raise RiskEvaluationError(f"Order format error: {e}") from e
            except Exception as e:
                logger.error(f"❌ Unexpected error in risk evaluation for {func_name}: {e}")
                raise RiskEvaluationError(f"Risk evaluation failed: {e}") from e

        else:
            # No order data found - might be non-trading function
            logger.debug(f"No order data found in {func_name} - proceeding without risk check")
            return func(*args, **kwargs)

    return wrapper


def _extract_order_from_args(args, kwargs) -> dict[str, Any] | None:
    """Extract order data from function arguments"""

    # Check for common order parameter names
    order_keys = ['order', 'order_request', 'trade_request', 'execution_request']

    for key in order_keys:
        if key in kwargs:
            order_data = kwargs[key]
            if hasattr(order_data, '__dict__'):
                return order_data.__dict__
            elif isinstance(order_data, dict):
                return order_data

    # Check positional arguments
    for arg in args:
        if hasattr(arg, '__dict__') and any(attr in arg.__dict__ for attr in ['symbol', 'side', 'size']):
            return arg.__dict__
        elif isinstance(arg, dict) and any(key in arg for key in ['symbol', 'side', 'size']):
            return arg

    # Check for individual parameters
    if 'symbol' in kwargs and 'side' in kwargs and 'size' in kwargs:
        return {
            'symbol': kwargs.get('symbol'),
            'side': kwargs.get('side'),
            'size': kwargs.get('size'),
            'price': kwargs.get('price'),
            'order_type': kwargs.get('order_type', 'market'),
            'client_order_id': kwargs.get('client_order_id')
        }

    return None


def _convert_to_standard_order(order_data: dict[str, Any]) -> StandardOrderRequest:
    """Convert any order format to StandardOrderRequest"""

    # Normalize side value
    side_value = order_data.get('side', '').lower()
    if side_value in ['buy', 'long', 'open_long']:
        side = OrderSide.BUY
    elif side_value in ['sell', 'short', 'open_short', 'close_long']:
        side = OrderSide.SELL
    else:
        raise ValueError(f"Invalid order side: {side_value}")

    return StandardOrderRequest(
        symbol=order_data.get('symbol', ''),
        side=side,
        size=float(order_data.get('size', 0)),
        price=float(order_data.get('price')) if order_data.get('price') else None,
        order_type=order_data.get('order_type', 'market'),
        client_order_id=order_data.get('client_order_id'),
        strategy_id=order_data.get('strategy_id', 'default')
    )


def _extract_market_data_from_args(args, kwargs) -> dict[str, Any] | None:
    """Extract market data from function arguments"""

    market_data_keys = ['market_data', 'market_conditions', 'ticker_data', 'price_data']

    for key in market_data_keys:
        if key in kwargs:
            data = kwargs[key]
            if hasattr(data, '__dict__'):
                return data.__dict__
            elif isinstance(data, dict):
                return data

    return None


def _update_order_size_in_args(args, kwargs, new_size: float):
    """Update order size in function arguments"""

    # Update in kwargs
    if 'size' in kwargs:
        kwargs['size'] = new_size

    if 'order' in kwargs and hasattr(kwargs['order'], 'size'):
        kwargs['order'].size = new_size

    if 'order_request' in kwargs and hasattr(kwargs['order_request'], 'size'):
        kwargs['order_request'].size = new_size


class RiskViolationError(Exception):
    """Error raised when order violates risk limits"""
    pass


class EmergencyStopError(Exception):
    """Error raised when kill switch is active"""
    pass


class RiskEvaluationError(Exception):
    """Error raised when risk evaluation fails"""
    pass


def register_unified_risk_enforcement():
    """
    Register unified risk enforcement globally
    Call this at startup to activate mandatory risk checking
    """

    logger.critical("🛡️ UNIFIED RISK ENFORCEMENT ACTIVATED")
    logger.critical("🛡️ ALL order execution now MANDATORY through UnifiedRiskGuard")

    # Initialize the unified risk guard
    risk_guard = UnifiedRiskGuard()

    if risk_guard.is_operational():
        logger.info("✅ UnifiedRiskGuard operational")
    else:
        logger.error("❌ UnifiedRiskGuard NOT operational - check configuration")

    return risk_guard


# Export key components
__all__ = [
    'mandatory_unified_risk_check',
    'register_unified_risk_enforcement',
    'RiskViolationError',
    'EmergencyStopError',
    'RiskEvaluationError'
]


# Auto-initialize on import
if __name__ != "__main__":
    try:
        register_unified_risk_enforcement()
    except Exception as e:
        logger.error(f"Failed to initialize unified risk enforcement: {e}")
