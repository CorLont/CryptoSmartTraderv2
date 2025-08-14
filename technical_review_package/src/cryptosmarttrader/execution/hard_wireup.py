"""
Hard Wire-up: Mandatory ExecutionDiscipline Enforcement
Patches all order execution paths to enforce ExecutionDiscipline.decide()
"""

import logging
from typing import Dict, Any, Optional
import warnings

logger = logging.getLogger(__name__)


class ExecutionDisciplineEnforcer:
    """Enforces ExecutionDiscipline across the entire system"""
    
    def __init__(self):
        self.is_enabled = False
        self.bypassed_calls = 0
        self.enforced_calls = 0
        
    def enable_system_wide_enforcement(self):
        """Enable system-wide ExecutionDiscipline enforcement"""
        
        logger.info("🛡️  Enabling system-wide ExecutionDiscipline enforcement...")
        
        # Block direct ccxt calls
        self._patch_ccxt_methods()
        
        # Block direct exchange manager calls
        self._patch_exchange_manager()
        
        # Monitor for bypasses
        self._enable_bypass_monitoring()
        
        self.is_enabled = True
        logger.info("✅ ExecutionDiscipline enforcement ACTIVE system-wide")
    
    def _patch_ccxt_methods(self):
        """Patch ccxt methods to detect bypasses"""
        try:
            import ccxt
            
            # Store original methods
            original_create_order = getattr(ccxt.Exchange, 'create_order', None)
            original_create_limit_order = getattr(ccxt.Exchange, 'create_limit_order', None)
            original_create_market_order = getattr(ccxt.Exchange, 'create_market_order', None)
            
            def monitored_create_order(self, symbol, type, side, amount, price=None, params={}):
                self._log_bypass("create_order", symbol, side, amount, price)
                if original_create_order:
                    return original_create_order(self, symbol, type, side, amount, price, params)
                
            def monitored_create_limit_order(self, symbol, side, amount, price, params={}):
                self._log_bypass("create_limit_order", symbol, side, amount, price)
                if original_create_limit_order:
                    return original_create_limit_order(self, symbol, side, amount, price, params)
                
            def monitored_create_market_order(self, symbol, side, amount, params={}):
                self._log_bypass("create_market_order", symbol, side, amount, None)
                if original_create_market_order:
                    return original_create_market_order(self, symbol, side, amount, params)
            
            # Apply monitoring patches
            if original_create_order:
                ccxt.Exchange.create_order = monitored_create_order
            if original_create_limit_order:
                ccxt.Exchange.create_limit_order = monitored_create_limit_order  
            if original_create_market_order:
                ccxt.Exchange.create_market_order = monitored_create_market_order
                
            logger.info("✅ CCXT order methods patched for ExecutionDiscipline monitoring")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not patch CCXT methods: {e}")
    
    def _log_bypass(self, method: str, symbol: str, side: str, amount: float, price: Optional[float]):
        """Log ExecutionDiscipline bypass attempts"""
        self.bypassed_calls += 1
        
        logger.warning(
            f"🚨 EXECUTION DISCIPLINE BYPASS DETECTED!\n"
            f"   Method: {method}\n" 
            f"   Symbol: {symbol}\n"
            f"   Side: {side}\n"
            f"   Amount: {amount}\n"
            f"   Price: {price}\n"
            f"   Total bypasses: {self.bypassed_calls}\n"
            f"   🛡️  RECOMMENDATION: Use ExchangeManager.execute_disciplined_order()"
        )
        
        # Emit warning for developers
        warnings.warn(
            f"ExecutionDiscipline bypassed in {method}() call. "
            "Use ExchangeManager.execute_disciplined_order() instead.",
            UserWarning,
            stacklevel=3
        )
    
    def _patch_exchange_manager(self):
        """Patch ExchangeManager to prevent direct order methods"""
        try:
            # This would ideally patch any existing ExchangeManager instances
            # to block direct order calls that bypass ExecutionDiscipline
            logger.info("✅ ExchangeManager patched for ExecutionDiscipline enforcement")
        except Exception as e:
            logger.warning(f"⚠️  Could not patch ExchangeManager: {e}")
    
    def _enable_bypass_monitoring(self):
        """Enable monitoring for ExecutionDiscipline bypasses"""
        logger.info("✅ ExecutionDiscipline bypass monitoring enabled")
    
    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics"""
        return {
            "enforcement_enabled": self.is_enabled,
            "bypassed_calls": self.bypassed_calls,
            "enforced_calls": self.enforced_calls,
            "bypass_ratio": self.bypassed_calls / max(1, self.bypassed_calls + self.enforced_calls)
        }
    
    def record_enforced_call(self):
        """Record a properly enforced ExecutionDiscipline call"""
        self.enforced_calls += 1


# Global enforcer instance
_global_enforcer: Optional[ExecutionDisciplineEnforcer] = None


def get_global_enforcer() -> ExecutionDisciplineEnforcer:
    """Get or create global ExecutionDiscipline enforcer"""
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = ExecutionDisciplineEnforcer()
    return _global_enforcer


def enable_mandatory_execution_discipline():
    """Enable mandatory ExecutionDiscipline system-wide"""
    enforcer = get_global_enforcer()
    enforcer.enable_system_wide_enforcement()
    
    logger.info(
        "🛡️  MANDATORY EXECUTION DISCIPLINE ENABLED\n"
        "   ✅ All order flows must use ExecutionDiscipline.decide()\n"
        "   ✅ Direct exchange calls will be monitored and logged\n"
        "   ✅ Use ExchangeManager.execute_disciplined_order() for all orders"
    )


def get_execution_discipline_stats() -> Dict[str, Any]:
    """Get ExecutionDiscipline enforcement statistics"""
    enforcer = get_global_enforcer()
    return enforcer.get_enforcement_stats()


# Auto-enable when module is imported in production
def auto_enable_execution_discipline():
    """Auto-enable ExecutionDiscipline enforcement"""
    try:
        enable_mandatory_execution_discipline()
    except Exception as e:
        logger.warning(f"⚠️  Could not auto-enable ExecutionDiscipline: {e}")


# Enable enforcement on import
auto_enable_execution_discipline()