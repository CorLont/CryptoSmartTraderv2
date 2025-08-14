"""
Risk-Enforced Execution Manager
Combineert ExecutionDiscipline en CentralRiskGuard voor complete trading pipeline
"""

import logging
from typing import Dict, Any, Optional
from .execution_discipline import ExecutionPolicy, OrderRequest, MarketConditions, ExecutionDecision
from .mandatory_enforcement import get_global_execution_policy
from ..risk.central_risk_guard import (
    CentralRiskGuard, TradingOperation, RiskDecision, get_global_risk_guard
)

logger = logging.getLogger(__name__)


class RiskEnforcedExecutionManager:
    """
    Complete execution pipeline met RiskGuard + ExecutionDiscipline
    VERPLICHTE poortwachter voor alle trading operaties
    """
    
    def __init__(self, exchange_manager=None):
        self.exchange_manager = exchange_manager
        self.execution_policy = get_global_execution_policy()
        self.risk_guard = get_global_risk_guard()
        self.logger = logging.getLogger(__name__)
        
        # Execution statistics
        self.total_requests = 0
        self.risk_rejections = 0
        self.execution_rejections = 0
        self.successful_executions = 0
    
    def execute_trading_operation(
        self,
        operation_type: str,  # "entry", "resize", "hedge", "exit"
        symbol: str,
        side: str,
        size_usd: float,
        limit_price: Optional[float] = None,
        strategy_id: str = "default",
        exchange_name: str = "kraken"
    ) -> Dict[str, Any]:
        """
        🛡️  COMPLETE RISK-ENFORCED EXECUTION PIPELINE
        
        1. CentralRiskGuard evaluatie (poortwachter)
        2. ExecutionDiscipline gates (market conditions)  
        3. Exchange execution (indien goedgekeurd)
        
        Returns comprehensive execution result
        """
        
        self.total_requests += 1
        current_price = limit_price or 50000.0  # Placeholder - should get from market data
        
        self.logger.info(
            f"🛡️  Risk-Enforced Execution: {operation_type} {symbol} "
            f"{side} ${size_usd:,.0f} @ {current_price}"
        )
        
        # STEP 1: CentralRiskGuard Evaluation (CRITICAL GATES)
        trading_operation = TradingOperation(
            operation_type=operation_type,
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            current_price=current_price,
            strategy_id=strategy_id
        )
        
        risk_evaluation = self.risk_guard.evaluate_operation(trading_operation)
        
        # Check RiskGuard decision
        if risk_evaluation.decision == RiskDecision.REJECT:
            self.risk_rejections += 1
            return {
                "success": False,
                "stage": "risk_guard",
                "error": f"RiskGuard rejected: {'; '.join(risk_evaluation.reasons)}",
                "risk_evaluation": risk_evaluation,
                "execution_result": None,
                "total_rejections": {
                    "risk": self.risk_rejections,
                    "execution": self.execution_rejections
                }
            }
        
        if risk_evaluation.decision == RiskDecision.KILL_SWITCH_ACTIVATED:
            self.risk_rejections += 1
            return {
                "success": False,
                "stage": "risk_guard",
                "error": "KILL SWITCH ACTIVATED - All trading stopped",
                "risk_evaluation": risk_evaluation,
                "execution_result": None,
                "kill_switch": True
            }
        
        # Adjust size if RiskGuard recommends reduction
        approved_size_usd = risk_evaluation.approved_size_usd
        if approved_size_usd < size_usd:
            self.logger.info(
                f"🛡️  RiskGuard reduced size: ${size_usd:,.0f} → ${approved_size_usd:,.0f}"
            )
        
        # STEP 2: ExecutionDiscipline Gates (MARKET CONDITIONS)
        if not self.exchange_manager:
            return {
                "success": False,
                "stage": "execution_discipline", 
                "error": "ExchangeManager not available",
                "risk_evaluation": risk_evaluation,
                "execution_result": None
            }
        
        # Get market conditions
        market_conditions = self.exchange_manager.create_market_conditions(symbol, exchange_name)
        if not market_conditions:
            self.execution_rejections += 1
            return {
                "success": False,
                "stage": "execution_discipline",
                "error": f"Could not get market conditions for {symbol}",
                "risk_evaluation": risk_evaluation,
                "execution_result": None
            }
        
        # Calculate position size (convert USD to units)
        position_size = approved_size_usd / current_price
        
        # Create OrderRequest for ExecutionDiscipline
        order_request = OrderRequest(
            symbol=symbol,
            side=side.upper(),
            size=position_size,
            order_type="limit",
            limit_price=limit_price,
            strategy_id=strategy_id
        )
        
        # ExecutionDiscipline evaluation
        execution_result = self.execution_policy.decide(order_request, market_conditions)
        
        if execution_result.decision == ExecutionDecision.REJECT:
            self.execution_rejections += 1
            return {
                "success": False,
                "stage": "execution_discipline",
                "error": f"ExecutionDiscipline rejected: {execution_result.reason}",
                "risk_evaluation": risk_evaluation,
                "execution_result": execution_result,
                "gate_results": execution_result.gate_results
            }
        
        if execution_result.decision == ExecutionDecision.DEFER:
            return {
                "success": False,
                "stage": "execution_discipline",
                "error": f"ExecutionDiscipline deferred: {execution_result.reason}",
                "risk_evaluation": risk_evaluation,
                "execution_result": execution_result,
                "retry_recommended": True
            }
        
        # STEP 3: Exchange Execution (BOTH GUARDS PASSED)
        self.logger.info(f"✅ Both guards passed - executing order: {order_request.client_order_id}")
        
        try:
            # Execute through disciplined exchange manager
            exchange_result = self.exchange_manager.execute_disciplined_order(
                symbol=symbol,
                side=side,
                size=position_size,
                order_type="limit",
                limit_price=limit_price,
                strategy_id=strategy_id,
                exchange_name=exchange_name
            )
            
            if exchange_result.get("success"):
                self.successful_executions += 1
                self.logger.info(f"✅ Order executed successfully: {exchange_result.get('order_id')}")
            
            return {
                "success": exchange_result.get("success", False),
                "stage": "exchange_execution",
                "exchange_result": exchange_result,
                "risk_evaluation": risk_evaluation,
                "execution_result": execution_result,
                "order_id": exchange_result.get("order_id"),
                "client_order_id": exchange_result.get("client_order_id"),
                "approved_size_usd": approved_size_usd,
                "final_size": position_size,
                "execution_stats": self.get_execution_stats()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Exchange execution failed: {e}")
            return {
                "success": False,
                "stage": "exchange_execution",
                "error": f"Exchange execution failed: {str(e)}",
                "risk_evaluation": risk_evaluation,
                "execution_result": execution_result
            }
    
    def update_portfolio_state(
        self,
        total_equity: float,
        daily_pnl: float,
        open_positions: int,
        total_exposure_usd: float,
        position_sizes: Dict[str, float],
        correlations: Dict[str, float] = None
    ):
        """Update portfolio state in RiskGuard"""
        self.risk_guard.update_portfolio_state(
            total_equity=total_equity,
            daily_pnl=daily_pnl,
            open_positions=open_positions,
            total_exposure_usd=total_exposure_usd,
            position_sizes=position_sizes,
            correlations=correlations or {}
        )
        
        self.logger.debug(f"📊 Portfolio updated: ${total_equity:,.0f} equity, {open_positions} positions")
    
    def activate_emergency_stop(self, reason: str):
        """Activate emergency kill switch"""
        self.risk_guard.activate_kill_switch(reason)
        self.logger.critical(f"🚨 EMERGENCY STOP: {reason}")
    
    def deactivate_emergency_stop(self, reason: str):
        """Deactivate emergency kill switch"""
        self.risk_guard.deactivate_kill_switch(reason)
        self.logger.info(f"✅ Emergency stop deactivated: {reason}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk and execution status"""
        risk_status = self.risk_guard.get_risk_status()
        execution_stats = self.get_execution_stats()
        
        return {
            "risk_guard": risk_status,
            "execution_stats": execution_stats,
            "overall_health": self._calculate_overall_health(risk_status, execution_stats)
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_executions": self.successful_executions,
            "risk_rejections": self.risk_rejections,
            "execution_rejections": self.execution_rejections,
            "success_rate": self.successful_executions / max(1, self.total_requests) * 100,
            "risk_rejection_rate": self.risk_rejections / max(1, self.total_requests) * 100,
            "execution_rejection_rate": self.execution_rejections / max(1, self.total_requests) * 100
        }
    
    def _calculate_overall_health(self, risk_status: Dict, execution_stats: Dict) -> Dict[str, Any]:
        """Calculate overall system health score"""
        
        # Risk health (0-100)
        risk_health = 100
        portfolio = risk_status["portfolio_state"]
        
        if portfolio["daily_pnl_pct"] < -1.0:
            risk_health -= 20
        if portfolio["current_drawdown_pct"] > 5.0:
            risk_health -= 30
        if portfolio["total_exposure_pct"] > 40.0:
            risk_health -= 15
        if portfolio["data_age_minutes"] > 3.0:
            risk_health -= 25
        
        # Execution health (0-100)
        execution_health = min(100, execution_stats["success_rate"])
        
        # Overall health
        overall_health = (risk_health + execution_health) / 2
        
        status = "HEALTHY"
        if overall_health < 50:
            status = "CRITICAL"
        elif overall_health < 70:
            status = "WARNING"
        elif overall_health < 85:
            status = "CAUTION"
        
        return {
            "overall_score": overall_health,
            "risk_score": risk_health,
            "execution_score": execution_health,
            "status": status,
            "kill_switch_active": risk_status["risk_limits"]["kill_switch_active"]
        }


# Global instance for easy access
_global_risk_enforced_manager: Optional[RiskEnforcedExecutionManager] = None


def get_global_risk_enforced_manager(exchange_manager=None) -> RiskEnforcedExecutionManager:
    """Get or create global RiskEnforcedExecutionManager"""
    global _global_risk_enforced_manager
    if _global_risk_enforced_manager is None:
        _global_risk_enforced_manager = RiskEnforcedExecutionManager(exchange_manager)
        logger.info("✅ Global RiskEnforcedExecutionManager initialized")
    return _global_risk_enforced_manager


def reset_global_risk_enforced_manager():
    """Reset global manager (for testing)"""
    global _global_risk_enforced_manager
    _global_risk_enforced_manager = None