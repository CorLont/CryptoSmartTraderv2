#!/usr/bin/env python3
"""
Unified Risk Guard - Eenduidige, herbruikbare klasse voor ELKE order
Mandatory risk enforcement met zero-bypass architecture
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Risk decision outcomes"""
    APPROVE = "approve"
    REJECT = "reject"  
    REDUCE_SIZE = "reduce_size"
    EMERGENCY_STOP = "emergency_stop"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class StandardOrderRequest:
    """Standardized order request structure - eenduidige interface"""
    symbol: str
    side: OrderSide
    size: float
    price: Optional[float] = None
    order_type: str = "market"
    client_order_id: Optional[str] = None
    strategy_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary voor logging"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'order_type': self.order_type,
            'client_order_id': self.client_order_id,
            'strategy_id': self.strategy_id,
            'timestamp': self.timestamp
        }


@dataclass
class RiskEvaluationResult:
    """Eenduidige risk evaluation result"""
    decision: RiskDecision
    reason: str
    adjusted_size: Optional[float] = None
    risk_score: float = 0.0
    evaluation_time_ms: float = 0.0
    breach_details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary voor audit trail"""
        return {
            'decision': self.decision.value,
            'reason': self.reason,
            'adjusted_size': self.adjusted_size,
            'risk_score': self.risk_score,
            'evaluation_time_ms': self.evaluation_time_ms,
            'breach_details': self.breach_details,
            'timestamp': self.timestamp
        }


@dataclass
class RiskLimits:
    """Comprehensive risk limits configuration"""
    # Kill-switch
    kill_switch_active: bool = False
    
    # Daily limits
    max_daily_loss_usd: float = 5000.0
    max_daily_loss_percent: float = 5.0
    
    # Drawdown limits
    max_drawdown_percent: float = 10.0
    
    # Position limits
    max_position_count: int = 10
    max_single_position_usd: float = 50000.0
    max_total_exposure_usd: float = 100000.0
    
    # Correlation limits
    max_correlation_exposure: float = 0.7
    
    # Data quality gates
    min_data_completeness: float = 0.95
    max_data_age_minutes: int = 5


@dataclass
class PortfolioState:
    """Current portfolio state voor risk calculations"""
    total_value_usd: float = 100000.0
    daily_pnl_usd: float = 0.0
    max_drawdown_from_peak: float = 0.0
    position_count: int = 0
    total_exposure_usd: float = 0.0
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


class UnifiedRiskGuard:
    """
    Unified Risk Guard - EENDUIDIGE, HERBRUIKBARE klasse voor ELKE order
    
    Design Principles:
    1. Single Responsibility: Risk evaluation voor orders
    2. Consistent Interface: Elke order gebruikt dezelfde methode
    3. Zero-bypass: Mandatory approval voor ALL orders
    4. Thread-safe: Concurrent access protection
    5. Audit Trail: Complete logging van alle decisions
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern ensures one central risk authority"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Core configuration
        self.limits = RiskLimits()
        self.portfolio_state = PortfolioState()
        
        # Audit trail
        self.decision_history: List[RiskEvaluationResult] = []
        self.audit_log_path = Path("logs/unified_risk_audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.rejections_count = 0
        self.approvals_count = 0
        
        # Kill switch state
        self.kill_switch_triggered_at: Optional[datetime] = None
        self.kill_switch_reason: Optional[str] = None
        
        # Emergency state persistence
        self.emergency_state_file = Path("emergency_unified_risk_state.json")
        
        logger.info("UnifiedRiskGuard initialized - eenduidige risk enforcement active")
    
    def evaluate_order(
        self, 
        order: StandardOrderRequest, 
        market_data: Optional[Dict[str, Any]] = None
    ) -> RiskEvaluationResult:
        """
        MANDATORY: Evaluate order through unified risk gates
        
        This is THE single method that ALL orders MUST use.
        No bypass possible, no alternative paths.
        
        Args:
            order: Standardized order request
            market_data: Optional market conditions
            
        Returns:
            RiskEvaluationResult: Eenduidige decision result
        """
        start_time = time.time()
        
        with self._lock:
            self.evaluation_count += 1
            
            try:
                # Execute mandatory unified risk evaluation
                result = self._execute_unified_risk_gates(order, market_data)
                
                # Record evaluation time
                evaluation_time_ms = (time.time() - start_time) * 1000
                result.evaluation_time_ms = evaluation_time_ms
                self.total_evaluation_time += evaluation_time_ms / 1000
                
                # Update counters
                if result.decision == RiskDecision.APPROVE:
                    self.approvals_count += 1
                else:
                    self.rejections_count += 1
                
                # Record decision in audit trail
                self._record_decision(order, result)
                
                return result
                
            except Exception as e:
                # Emergency fallback - REJECT on ANY error
                error_result = RiskEvaluationResult(
                    decision=RiskDecision.REJECT,
                    reason=f"EVALUATION_ERROR: {str(e)}",
                    evaluation_time_ms=(time.time() - start_time) * 1000
                )
                
                self.rejections_count += 1
                self._record_decision(order, error_result)
                
                logger.error(f"Risk evaluation failed for order {order.client_order_id}: {e}")
                return error_result
    
    def _execute_unified_risk_gates(
        self, 
        order: StandardOrderRequest, 
        market_data: Optional[Dict[str, Any]]
    ) -> RiskEvaluationResult:
        """Execute ALL mandatory risk gates in sequence"""
        
        # Gate 1: Kill Switch Check (HIGHEST PRIORITY)
        if self.limits.kill_switch_active:
            return RiskEvaluationResult(
                decision=RiskDecision.EMERGENCY_STOP,
                reason="KILL_SWITCH_ACTIVE: All trading halted by emergency stop",
                risk_score=1.0
            )
        
        # Gate 2: Data Quality Validation
        data_quality_result = self._validate_data_quality(market_data)
        if data_quality_result.decision != RiskDecision.APPROVE:
            return data_quality_result
        
        # Gate 3: Daily Loss Limits
        daily_loss_result = self._check_daily_loss_limits(order)
        if daily_loss_result.decision != RiskDecision.APPROVE:
            return daily_loss_result
        
        # Gate 4: Drawdown Limits
        drawdown_result = self._check_drawdown_limits(order)
        if drawdown_result.decision != RiskDecision.APPROVE:
            return drawdown_result
        
        # Gate 5: Position Count Limits
        position_count_result = self._check_position_count_limits(order)
        if position_count_result.decision != RiskDecision.APPROVE:
            return position_count_result
        
        # Gate 6: Total Exposure Limits (with size reduction)
        exposure_result = self._check_exposure_limits(order)
        if exposure_result.decision not in [RiskDecision.APPROVE, RiskDecision.REDUCE_SIZE]:
            return exposure_result
        
        # Gate 7: Single Position Size Limits
        position_size_result = self._check_position_size_limits(order)
        if position_size_result.decision != RiskDecision.APPROVE:
            return position_size_result
        
        # Gate 8: Correlation Limits
        correlation_result = self._check_correlation_limits(order)
        if correlation_result.decision != RiskDecision.APPROVE:
            return correlation_result
        
        # All gates passed - calculate final risk score
        risk_score = self._calculate_risk_score(order, market_data)
        
        # Return final decision (could be REDUCE_SIZE from exposure gate)
        final_decision = exposure_result.decision
        adjusted_size = exposure_result.adjusted_size
        
        return RiskEvaluationResult(
            decision=final_decision,
            reason="APPROVED: All unified risk gates passed",
            adjusted_size=adjusted_size,
            risk_score=risk_score
        )
    
    def _validate_data_quality(self, market_data: Optional[Dict[str, Any]]) -> RiskEvaluationResult:
        """Validate market data quality"""
        if market_data is None:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason="DATA_GAP: No market data available",
                risk_score=0.8
            )
        
        # Check data age
        data_timestamp = market_data.get('timestamp', 0)
        data_age_minutes = (time.time() - data_timestamp) / 60
        
        if data_age_minutes > self.limits.max_data_age_minutes:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"STALE_DATA: Market data {data_age_minutes:.1f}m old > {self.limits.max_data_age_minutes}m limit",
                risk_score=0.7
            )
        
        # Check data completeness
        required_fields = ['price', 'volume', 'timestamp']
        present_fields = sum(1 for field in required_fields if field in market_data and market_data[field] is not None)
        completeness = present_fields / len(required_fields)
        
        if completeness < self.limits.min_data_completeness:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"INCOMPLETE_DATA: Completeness {completeness:.1%} < {self.limits.min_data_completeness:.1%}",
                risk_score=0.6
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="DATA_QUALITY_OK",
            risk_score=0.1
        )
    
    def _check_daily_loss_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check daily loss limits"""
        current_daily_pnl = self.portfolio_state.daily_pnl_usd
        
        # Check absolute loss limit
        if current_daily_pnl <= -self.limits.max_daily_loss_usd:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"DAILY_LOSS_LIMIT: Current loss ${abs(current_daily_pnl):,.0f} >= ${self.limits.max_daily_loss_usd:,.0f} limit",
                risk_score=0.9,
                breach_details={'daily_pnl_usd': current_daily_pnl, 'limit': self.limits.max_daily_loss_usd}
            )
        
        # Check percentage loss limit
        loss_percent = abs(current_daily_pnl) / self.portfolio_state.total_value_usd * 100
        if loss_percent >= self.limits.max_daily_loss_percent:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"DAILY_LOSS_PERCENT: Current loss {loss_percent:.1f}% >= {self.limits.max_daily_loss_percent:.1f}% limit",
                risk_score=0.85,
                breach_details={'loss_percent': loss_percent, 'limit_percent': self.limits.max_daily_loss_percent}
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="DAILY_LOSS_OK",
            risk_score=0.2
        )
    
    def _check_drawdown_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check maximum drawdown limits"""
        current_drawdown = self.portfolio_state.max_drawdown_from_peak
        
        if current_drawdown >= self.limits.max_drawdown_percent:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"MAX_DRAWDOWN: Current drawdown {current_drawdown:.1f}% >= {self.limits.max_drawdown_percent:.1f}% limit",
                risk_score=0.95,
                breach_details={'drawdown_percent': current_drawdown, 'limit': self.limits.max_drawdown_percent}
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="DRAWDOWN_OK",
            risk_score=0.15
        )
    
    def _check_position_count_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check position count limits"""
        current_positions = self.portfolio_state.position_count
        
        # Only check for new positions (buy orders or sell orders opening new positions)
        if order.side == OrderSide.BUY or order.symbol not in self.portfolio_state.positions:
            if current_positions >= self.limits.max_position_count:
                return RiskEvaluationResult(
                    decision=RiskDecision.REJECT,
                    reason=f"POSITION_COUNT_LIMIT: Current {current_positions} >= {self.limits.max_position_count} limit",
                    risk_score=0.8,
                    breach_details={'position_count': current_positions, 'limit': self.limits.max_position_count}
                )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="POSITION_COUNT_OK",
            risk_score=0.1
        )
    
    def _check_exposure_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check total exposure limits with size reduction capability"""
        order_value = order.size * (order.price or 1000.0)  # Estimate if no price
        current_exposure = self.portfolio_state.total_exposure_usd
        new_total_exposure = current_exposure + order_value
        
        if new_total_exposure > self.limits.max_total_exposure_usd:
            # Calculate maximum allowed order size
            available_exposure = self.limits.max_total_exposure_usd - current_exposure
            
            if available_exposure <= 0:
                return RiskEvaluationResult(
                    decision=RiskDecision.REJECT,
                    reason=f"TOTAL_EXPOSURE_LIMIT: No available exposure (current: ${current_exposure:,.0f})",
                    risk_score=0.9,
                    breach_details={'exposure_usd': current_exposure, 'limit': self.limits.max_total_exposure_usd}
                )
            
            # Reduce order size to fit within limits
            max_allowed_size = available_exposure / (order.price or 1000.0)
            adjusted_size = min(order.size, max_allowed_size)
            
            if adjusted_size < order.size * 0.1:  # Less than 10% of original size
                return RiskEvaluationResult(
                    decision=RiskDecision.REJECT,
                    reason=f"EXPOSURE_REDUCTION_TOO_SMALL: Adjusted size {adjusted_size:.4f} < 10% of requested {order.size}",
                    risk_score=0.8
                )
            
            return RiskEvaluationResult(
                decision=RiskDecision.REDUCE_SIZE,
                reason=f"EXPOSURE_SIZE_REDUCED: From {order.size} to {adjusted_size:.4f}",
                adjusted_size=adjusted_size,
                risk_score=0.6
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="TOTAL_EXPOSURE_OK",
            risk_score=0.2
        )
    
    def _check_position_size_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check single position size limits"""
        order_value = order.size * (order.price or 1000.0)
        
        if order_value > self.limits.max_single_position_usd:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"POSITION_SIZE_LIMIT: Order value ${order_value:,.0f} > ${self.limits.max_single_position_usd:,.0f} limit",
                risk_score=0.7,
                breach_details={'order_value_usd': order_value, 'limit': self.limits.max_single_position_usd}
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="POSITION_SIZE_OK",
            risk_score=0.1
        )
    
    def _check_correlation_limits(self, order: StandardOrderRequest) -> RiskEvaluationResult:
        """Check correlation exposure limits"""
        # Simplified correlation check - in real implementation would calculate actual correlations
        symbol_group = self._get_symbol_group(order.symbol)
        group_exposure = sum(
            pos['value_usd'] for pos in self.portfolio_state.positions.values()
            if self._get_symbol_group(pos.get('symbol', '')) == symbol_group
        )
        
        total_portfolio_value = self.portfolio_state.total_value_usd
        group_exposure_ratio = group_exposure / total_portfolio_value
        
        if group_exposure_ratio > self.limits.max_correlation_exposure:
            return RiskEvaluationResult(
                decision=RiskDecision.REJECT,
                reason=f"CORRELATION_LIMIT: {symbol_group} exposure {group_exposure_ratio:.1%} > {self.limits.max_correlation_exposure:.1%}",
                risk_score=0.8,
                breach_details={'group': symbol_group, 'exposure_ratio': group_exposure_ratio}
            )
        
        return RiskEvaluationResult(
            decision=RiskDecision.APPROVE,
            reason="CORRELATION_OK",
            risk_score=0.15
        )
    
    def _get_symbol_group(self, symbol: str) -> str:
        """Get symbol correlation group"""
        # Simplified grouping logic
        if any(crypto in symbol.upper() for crypto in ['BTC', 'BITCOIN']):
            return 'BTC_GROUP'
        elif any(crypto in symbol.upper() for crypto in ['ETH', 'ETHEREUM']):
            return 'ETH_GROUP'
        elif any(crypto in symbol.upper() for crypto in ['SOL', 'SOLANA']):
            return 'SOL_GROUP'
        else:
            return 'OTHER_GROUP'
    
    def _calculate_risk_score(self, order: StandardOrderRequest, market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)"""
        risk_factors = []
        
        # Portfolio concentration risk
        portfolio_value = self.portfolio_state.total_value_usd
        order_value = order.size * (order.price or 1000.0)
        concentration_risk = order_value / portfolio_value
        risk_factors.append(min(concentration_risk * 2, 1.0))  # Cap at 1.0
        
        # Position count risk
        position_count_risk = self.portfolio_state.position_count / self.limits.max_position_count
        risk_factors.append(position_count_risk)
        
        # Daily PnL risk
        if self.portfolio_state.daily_pnl_usd < 0:
            daily_pnl_risk = abs(self.portfolio_state.daily_pnl_usd) / self.limits.max_daily_loss_usd
            risk_factors.append(min(daily_pnl_risk, 1.0))
        else:
            risk_factors.append(0.0)
        
        # Market volatility risk (if market data available)
        if market_data and 'volatility' in market_data:
            volatility_risk = min(market_data['volatility'] / 0.5, 1.0)  # Normalize to 50% volatility
            risk_factors.append(volatility_risk)
        
        # Return average risk score
        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
    
    def _record_decision(self, order: StandardOrderRequest, result: RiskEvaluationResult):
        """Record decision in audit trail"""
        # Add to in-memory history
        self.decision_history.append(result)
        
        # Keep history manageable
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
        
        # Log to audit file
        audit_entry = {
            'timestamp': result.timestamp,
            'order': order.to_dict(),
            'result': result.to_dict()
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch"""
        with self._lock:
            self.limits.kill_switch_active = True
            self.kill_switch_triggered_at = datetime.utcnow()
            self.kill_switch_reason = reason
            
            # Persist emergency state
            self._save_emergency_state()
            
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate_kill_switch(self, reason: str = "Manual deactivation"):
        """Deactivate kill switch (requires explicit action)"""
        with self._lock:
            self.limits.kill_switch_active = False
            self.kill_switch_triggered_at = None
            self.kill_switch_reason = None
            
            logger.warning(f"Kill switch deactivated: {reason}")
    
    def _save_emergency_state(self):
        """Save emergency state to disk"""
        emergency_state = {
            'kill_switch_active': self.limits.kill_switch_active,
            'triggered_at': self.kill_switch_triggered_at.isoformat() if self.kill_switch_triggered_at else None,
            'reason': self.kill_switch_reason,
            'portfolio_state': {
                'total_value_usd': self.portfolio_state.total_value_usd,
                'daily_pnl_usd': self.portfolio_state.daily_pnl_usd,
                'position_count': self.portfolio_state.position_count,
                'total_exposure_usd': self.portfolio_state.total_exposure_usd
            }
        }
        
        try:
            with open(self.emergency_state_file, 'w') as f:
                json.dump(emergency_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")
    
    def update_portfolio_state(self, new_state: PortfolioState):
        """Update portfolio state (called by portfolio manager)"""
        with self._lock:
            self.portfolio_state = new_state
            self.portfolio_state.last_update = time.time()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get risk guard performance metrics"""
        with self._lock:
            avg_evaluation_time = self.total_evaluation_time / max(self.evaluation_count, 1)
            approval_rate = self.approvals_count / max(self.evaluation_count, 1)
            rejection_rate = self.rejections_count / max(self.evaluation_count, 1)
            
            return {
                'total_evaluations': self.evaluation_count,
                'approvals': self.approvals_count,
                'rejections': self.rejections_count,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'avg_evaluation_time_ms': avg_evaluation_time * 1000,
                'kill_switch_active': self.limits.kill_switch_active,
                'decision_history_size': len(self.decision_history)
            }
    
    def get_recent_decisions(self, limit: int = 100) -> List[RiskEvaluationResult]:
        """Get recent risk decisions"""
        with self._lock:
            return self.decision_history[-limit:]
    
    def is_operational(self) -> bool:
        """Check if risk guard is operational"""
        return (
            hasattr(self, '_initialized') and 
            self._initialized and 
            not self.limits.kill_switch_active
        )


# Export main class
__all__ = ['UnifiedRiskGuard', 'StandardOrderRequest', 'RiskEvaluationResult', 'RiskDecision', 'OrderSide']