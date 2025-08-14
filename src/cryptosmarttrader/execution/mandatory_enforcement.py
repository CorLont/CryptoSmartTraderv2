"""
Mandatory Execution Discipline Enforcement
Decorators and guards to enforce ExecutionDiscipline.decide() on ALL order flows
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional
from .execution_discipline import ExecutionPolicy, OrderRequest, MarketConditions, ExecutionDecision

logger = logging.getLogger(__name__)

# Global execution policy instance
_global_execution_policy: Optional[ExecutionPolicy] = None


def get_global_execution_policy() -> ExecutionPolicy:
    """Get or create global execution policy instance"""
    global _global_execution_policy
    if _global_execution_policy is None:
        _global_execution_policy = ExecutionPolicy()
        logger.info("✅ Global ExecutionPolicy initialized")
    return _global_execution_policy


def reset_global_execution_policy():
    """Reset global execution policy (for testing)"""
    global _global_execution_policy
    _global_execution_policy = None


class ExecutionDisciplineViolation(Exception):
    """Raised when execution discipline is bypassed"""
    pass


def require_execution_discipline(
    extract_order_request: Callable = None,
    extract_market_conditions: Callable = None
):
    """
    Decorator to enforce ExecutionDiscipline.decide() on order functions
    
    Args:
        extract_order_request: Function to extract OrderRequest from args/kwargs
        extract_market_conditions: Function to extract MarketConditions from args/kwargs
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get execution policy
            policy = get_global_execution_policy()
            
            # Extract order request and market conditions
            order_request = None
            market_conditions = None
            
            try:
                if extract_order_request:
                    order_request = extract_order_request(*args, **kwargs)
                if extract_market_conditions:
                    market_conditions = extract_market_conditions(*args, **kwargs)
                    
                # If we have both, enforce discipline
                if order_request and market_conditions:
                    result = policy.decide(order_request, market_conditions)
                    
                    if result.decision == ExecutionDecision.REJECT:
                        logger.warning(f"🚫 ExecutionDiscipline REJECTED order: {result.reason}")
                        raise ExecutionDisciplineViolation(f"Order rejected: {result.reason}")
                    
                    elif result.decision == ExecutionDecision.DEFER:
                        logger.info(f"⏳ ExecutionDiscipline DEFERRED order: {result.reason}")
                        raise ExecutionDisciplineViolation(f"Order deferred: {result.reason}")
                    
                    logger.info(f"✅ ExecutionDiscipline APPROVED order: {order_request.client_order_id}")
                
                else:
                    logger.warning(f"⚠️  ExecutionDiscipline bypassed in {func.__name__} - missing order_request or market_conditions")
                    
            except ExecutionDisciplineViolation:
                raise
            except Exception as e:
                logger.error(f"❌ ExecutionDiscipline enforcement failed in {func.__name__}: {e}")
                # Don't fail the order due to discipline check errors, but log them
                
            # Execute original function
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def ensure_execution_discipline_in_exchange_calls():
    """
    Monkey patch common exchange methods to ensure ExecutionDiscipline
    """
    import ccxt
    
    # Store original methods
    original_create_order = getattr(ccxt.Exchange, 'create_order', None)
    original_create_limit_order = getattr(ccxt.Exchange, 'create_limit_order', None)
    original_create_market_order = getattr(ccxt.Exchange, 'create_market_order', None)
    
    def disciplined_create_order(self, symbol, type, side, amount, price=None, params={}):
        """Disciplined version of ccxt create_order"""
        logger.warning(f"🚨 UNDISCIPLINED ORDER DETECTED: {symbol} {side} {amount} - ExecutionDiscipline.decide() was BYPASSED!")
        logger.warning(f"🛡️  RECOMMENDATION: Use ExecutionPolicy.decide() before calling exchange methods")
        
        # Still allow the order but log the violation
        if original_create_order:
            return original_create_order(self, symbol, type, side, amount, price, params)
        else:
            raise NotImplementedError("Original create_order method not found")
    
    def disciplined_create_limit_order(self, symbol, side, amount, price, params={}):
        """Disciplined version of ccxt create_limit_order"""
        logger.warning(f"🚨 UNDISCIPLINED LIMIT ORDER: {symbol} {side} {amount} @ {price} - ExecutionDiscipline BYPASSED!")
        
        if original_create_limit_order:
            return original_create_limit_order(self, symbol, side, amount, price, params)
        else:
            raise NotImplementedError("Original create_limit_order method not found")
    
    def disciplined_create_market_order(self, symbol, side, amount, params={}):
        """Disciplined version of ccxt create_market_order"""
        logger.warning(f"🚨 UNDISCIPLINED MARKET ORDER: {symbol} {side} {amount} - ExecutionDiscipline BYPASSED!")
        
        if original_create_market_order:
            return original_create_market_order(self, symbol, side, amount, params)
        else:
            raise NotImplementedError("Original create_market_order method not found")
    
    # Apply monkey patches
    if original_create_order:
        ccxt.Exchange.create_order = disciplined_create_order
    if original_create_limit_order:
        ccxt.Exchange.create_limit_order = disciplined_create_limit_order
    if original_create_market_order:
        ccxt.Exchange.create_market_order = disciplined_create_market_order
    
    logger.info("🛡️  ExecutionDiscipline monitoring enabled on ccxt exchange methods")


class DisciplinedExchangeManager:
    """
    Exchange manager that enforces ExecutionDiscipline on all orders
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.execution_policy = get_global_execution_policy()
        self.logger = logging.getLogger(__name__)
    
    def execute_disciplined_order(
        self,
        order_request: OrderRequest,
        market_conditions: MarketConditions,
        exchange_name: str = "kraken"
    ) -> Dict[str, Any]:
        """
        Execute order through mandatory ExecutionDiscipline gates
        
        Returns:
            Dict with execution result
        """
        
        # Mandatory ExecutionDiscipline.decide() gate
        result = self.execution_policy.decide(order_request, market_conditions)
        
        if result.decision == ExecutionDecision.REJECT:
            return {
                "success": False,
                "error": f"ExecutionDiscipline rejected: {result.reason}",
                "gate_results": result.gate_results,
                "order_id": None
            }
        
        if result.decision == ExecutionDecision.DEFER:
            return {
                "success": False,
                "error": f"ExecutionDiscipline deferred: {result.reason}",
                "gate_results": result.gate_results,
                "order_id": None
            }
        
        # Order approved - execute through exchange
        try:
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if not exchange:
                return {
                    "success": False,
                    "error": f"Exchange {exchange_name} not available",
                    "gate_results": result.gate_results,
                    "order_id": None
                }
            
            # Prepare order parameters
            order_params = {
                "type": order_request.order_type,
                "timeInForce": order_request.time_in_force.value,
                "clientOrderId": order_request.client_order_id
            }
            
            # Execute order through exchange
            if order_request.order_type == "limit" and order_request.limit_price:
                exchange_result = exchange.create_limit_order(
                    symbol=order_request.symbol,
                    side=order_request.side.value,
                    amount=order_request.size,
                    price=order_request.limit_price,
                    params=order_params
                )
            else:
                exchange_result = exchange.create_market_order(
                    symbol=order_request.symbol,
                    side=order_request.side.value,
                    amount=order_request.size,
                    params=order_params
                )
            
            self.logger.info(f"✅ Order executed: {order_request.client_order_id}")
            
            return {
                "success": True,
                "exchange_result": exchange_result,
                "gate_results": result.gate_results,
                "order_id": exchange_result.get("id"),
                "client_order_id": order_request.client_order_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Order execution failed: {e}")
            return {
                "success": False,
                "error": f"Exchange execution failed: {str(e)}",
                "gate_results": result.gate_results,
                "order_id": None
            }


# Initialize discipline monitoring on module import
def initialize_execution_discipline_monitoring():
    """Initialize execution discipline monitoring across the system"""
    try:
        ensure_execution_discipline_in_exchange_calls()
        logger.info("✅ ExecutionDiscipline monitoring initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ExecutionDiscipline monitoring: {e}")


# Auto-initialize when module is imported
initialize_execution_discipline_monitoring()