"""
Risk Guard Integration System
Integrates central risk guard with trading execution and portfolio management
"""

from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass

from .central_risk_guard import (
    get_risk_guard, CentralRiskGuard, RiskCheckResult, PositionInfo, DataGap
)
from ..execution.execution_discipline import ExecutionPolicy, OrderRequest, MarketConditions

logger = logging.getLogger(__name__)


class RiskIntegratedExecutionPolicy(ExecutionPolicy):
    """
    ExecutionPolicy with integrated central risk management
    ALL orders pass through both execution discipline AND risk guard
    """
    
    def __init__(self, risk_guard: Optional[CentralRiskGuard] = None):
        super().__init__()
        self.risk_guard = risk_guard or get_risk_guard()
    
    def decide(self, order_request: OrderRequest, market_conditions: MarketConditions):
        """
        Enhanced decision with integrated risk checking
        
        1. First: Standard execution discipline gates
        2. Then: Central risk guard validation
        3. Final: Combined decision
        """
        
        # Step 1: Standard execution discipline
        execution_result = super().decide(order_request, market_conditions)
        
        if execution_result.decision.value != "approve":
            return execution_result  # Already rejected by execution gates
        
        # Step 2: Central risk guard check
        trade_size_usd = order_request.size * (order_request.limit_price or market_conditions.last_price)
        
        risk_result = self.risk_guard.check_trade_risk(
            symbol=order_request.symbol,
            trade_size_usd=trade_size_usd,
            strategy_id=order_request.strategy_id
        )
        
        # Step 3: Combined decision
        if not risk_result.is_safe:
            # Override execution approval with risk rejection
            execution_result.decision = execution_result.decision.__class__("reject")  # Keep same enum type
            
            violation_reasons = [v.description for v in risk_result.violations]
            execution_result.reason = f"Risk guard rejection: {'; '.join(violation_reasons)}"
            
            # Add risk information to gate results
            execution_result.gate_results.update({
                'risk_guard': False,
                'risk_score': risk_result.risk_score,
                'violations_count': len(risk_result.violations)
            })
            
            logger.warning(f"Risk guard rejected order {order_request.client_order_id}: {execution_result.reason}")
        else:
            # Add risk information to approved order
            execution_result.gate_results.update({
                'risk_guard': True,
                'risk_score': risk_result.risk_score,
                'violations_count': 0
            })
            
            logger.info(f"Risk guard approved order {order_request.client_order_id} (risk: {risk_result.risk_score:.2f})")
        
        return execution_result


class RiskAwarePortfolioManager:
    """Portfolio manager with integrated risk monitoring"""
    
    def __init__(self, risk_guard: Optional[CentralRiskGuard] = None):
        self.risk_guard = risk_guard or get_risk_guard()
        self.positions: Dict[str, PositionInfo] = {}
    
    def add_position(self, position: PositionInfo):
        """Add position with risk tracking"""
        # Update local tracking
        key = f"{position.symbol}_{position.strategy_id}"
        self.positions[key] = position
        
        # Update central risk guard
        self.risk_guard.update_position(position)
        
        logger.info(f"Position added: {position.symbol} ${position.size_usd:,.0f}")
        
        # Check if this position triggers any warnings
        self._check_position_risk(position)
    
    def update_position(self, symbol: str, current_price: float, strategy_id: str = "default"):
        """Update position with current market price"""
        key = f"{symbol}_{strategy_id}"
        
        if key in self.positions:
            position = self.positions[key]
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.size_usd / position.entry_price
            
            # Update risk guard
            self.risk_guard.update_position(position)
            
            # Check for risk violations
            self._check_position_risk(position)
    
    def close_position(self, symbol: str, strategy_id: str = "default"):
        """Close position and update risk tracking"""
        key = f"{symbol}_{strategy_id}"
        
        if key in self.positions:
            position = self.positions[key]
            
            # Update daily P&L
            realized_pnl = position.unrealized_pnl
            current_daily_pnl = getattr(self.risk_guard, 'daily_pnl', 0.0)
            self.risk_guard.update_daily_pnl(current_daily_pnl + realized_pnl)
            
            # Remove from tracking
            del self.positions[key]
            self.risk_guard.remove_position(symbol, strategy_id)
            
            logger.info(f"Position closed: {symbol} PnL: ${realized_pnl:,.2f}")
    
    def _check_position_risk(self, position: PositionInfo):
        """Check individual position for risk issues"""
        # Check for large unrealized losses
        if position.unrealized_pnl < -1000:  # $1k loss threshold
            logger.warning(f"Large unrealized loss: {position.symbol} ${position.unrealized_pnl:,.2f}")
        
        # Check position size vs limits
        total_equity = getattr(self.risk_guard, 'current_equity', 100000.0)
        position_percent = abs(position.size_usd) / total_equity * 100
        
        if position_percent > self.risk_guard.limits.max_single_position_percent:
            logger.warning(f"Large position: {position.symbol} {position_percent:.1f}% of equity")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with risk metrics"""
        total_value = sum(pos.size_usd for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_value_usd': total_value,
            'total_unrealized_pnl': total_pnl,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'size_usd': pos.size_usd,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'strategy': pos.strategy_id
                }
                for pos in self.positions.values()
            ],
            'risk_summary': self.risk_guard.get_risk_summary()
        }


class DataQualityMonitor:
    """Monitor data quality and report gaps to risk guard"""
    
    def __init__(self, risk_guard: Optional[CentralRiskGuard] = None):
        self.risk_guard = risk_guard or get_risk_guard()
        self.last_data_times: Dict[str, float] = {}
    
    def report_data_point(self, symbol: str, timestamp: float, data_type: str = "price"):
        """Report new data point and check for gaps"""
        current_time = timestamp
        
        if symbol in self.last_data_times:
            gap_seconds = current_time - self.last_data_times[symbol]
            gap_minutes = gap_seconds / 60
            
            # Check for significant gaps
            if gap_minutes > self.risk_guard.limits.max_data_gap_minutes:
                from datetime import datetime
                
                gap = DataGap(
                    symbol=symbol,
                    gap_start=datetime.fromtimestamp(self.last_data_times[symbol]),
                    gap_end=datetime.fromtimestamp(current_time),
                    gap_minutes=gap_minutes,
                    data_type=data_type
                )
                
                self.risk_guard.report_data_gap(gap)
        
        self.last_data_times[symbol] = current_time
    
    def check_data_freshness(self) -> Dict[str, float]:
        """Check freshness of all tracked symbols"""
        import time
        current_time = time.time()
        
        freshness = {}
        for symbol, last_time in self.last_data_times.items():
            minutes_old = (current_time - last_time) / 60
            freshness[symbol] = minutes_old
        
        return freshness


# Global instances for easy integration
_integrated_execution_policy: Optional[RiskIntegratedExecutionPolicy] = None
_portfolio_manager: Optional[RiskAwarePortfolioManager] = None
_data_monitor: Optional[DataQualityMonitor] = None


def get_integrated_execution_policy() -> RiskIntegratedExecutionPolicy:
    """Get risk-integrated execution policy"""
    global _integrated_execution_policy
    if _integrated_execution_policy is None:
        _integrated_execution_policy = RiskIntegratedExecutionPolicy()
    return _integrated_execution_policy


def get_portfolio_manager() -> RiskAwarePortfolioManager:
    """Get risk-aware portfolio manager"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = RiskAwarePortfolioManager()
    return _portfolio_manager


def get_data_monitor() -> DataQualityMonitor:
    """Get data quality monitor"""
    global _data_monitor
    if _data_monitor is None:
        _data_monitor = DataQualityMonitor()
    return _data_monitor
