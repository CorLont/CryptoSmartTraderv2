"""
Central RiskGuard - Mandatory risk gates voor ALL order execution
Zero-bypass architecture met comprehensive risk validation
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, Union
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


@dataclass
class OrderRequest:
    """Standardized order request structure"""
    symbol: str
    side: str  # buy/sell
    size: float
    price: Optional[float] = None
    order_type: str = "market"
    client_order_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskLimits:
    """Configurable risk limits"""
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
    """Current portfolio state for risk calculations"""
    total_value_usd: float = 100000.0
    daily_pnl_usd: float = 0.0
    max_drawdown_from_peak: float = 0.0
    position_count: int = 0
    total_exposure_usd: float = 0.0
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


class CentralRiskGuard:
    """
    Mandatory central risk guard - ALL orders must pass through here
    Zero-bypass architecture with comprehensive risk validation
    """
    
    def __init__(self, config_path: str = "config/risk_limits.json"):
        self.config_path = Path(config_path)
        self.limits = self._load_risk_limits()
        self.portfolio_state = PortfolioState()
        self.audit_log_path = Path("logs/risk_audit.log")
        self.audit_log_path.parent.mkdir(exist_ok=True)
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.rejections_count = 0
        
        logger.info("CentralRiskGuard initialized with zero-bypass architecture")
    
    def _load_risk_limits(self) -> RiskLimits:
        """Load risk limits from config file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return RiskLimits(**config_data)
            else:
                # Create default config
                default_limits = RiskLimits()
                self._save_risk_limits(default_limits)
                return default_limits
        except Exception as e:
            logger.error(f"Failed to load risk limits: {e}")
            return RiskLimits()
    
    def _save_risk_limits(self, limits: RiskLimits) -> None:
        """Save risk limits to config file"""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                # Convert dataclass to dict
                config_data = {
                    'kill_switch_active': limits.kill_switch_active,
                    'max_daily_loss_usd': limits.max_daily_loss_usd,
                    'max_daily_loss_percent': limits.max_daily_loss_percent,
                    'max_drawdown_percent': limits.max_drawdown_percent,
                    'max_position_count': limits.max_position_count,
                    'max_single_position_usd': limits.max_single_position_usd,
                    'max_total_exposure_usd': limits.max_total_exposure_usd,
                    'max_correlation_exposure': limits.max_correlation_exposure,
                    'min_data_completeness': limits.min_data_completeness,
                    'max_data_age_minutes': limits.max_data_age_minutes
                }
                json.dump(config_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk limits: {e}")
    
    def evaluate_order(self, order: OrderRequest, market_data: Optional[Dict] = None) -> Tuple[RiskDecision, str, Optional[float]]:
        """
        MANDATORY risk evaluation - ALL orders must pass through here
        Returns: (decision, reason, adjusted_size)
        """
        start_time = time.time()
        self.evaluation_count += 1
        
        try:
            # Gate 1: Kill-switch check
            if self.limits.kill_switch_active:
                decision = RiskDecision.EMERGENCY_STOP
                reason = "KILL_SWITCH_ACTIVE: Trading halted by emergency stop"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # Gate 2: Data gap check
            data_quality_result = self._check_data_quality(market_data)
            if not data_quality_result[0]:
                decision = RiskDecision.REJECT
                reason = f"DATA_QUALITY_FAIL: {data_quality_result[1]}"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # Gate 3: Daily loss limits
            daily_loss_result = self._check_daily_loss_limits(order)
            if not daily_loss_result[0]:
                decision = RiskDecision.REJECT
                reason = f"DAILY_LOSS_LIMIT: {daily_loss_result[1]}"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # Gate 4: Drawdown limits
            drawdown_result = self._check_drawdown_limits(order)
            if not drawdown_result[0]:
                decision = RiskDecision.REJECT
                reason = f"DRAWDOWN_LIMIT: {drawdown_result[1]}"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # Gate 5: Position count limits
            position_count_result = self._check_position_limits(order)
            if not position_count_result[0]:
                decision = RiskDecision.REJECT
                reason = f"POSITION_LIMIT: {position_count_result[1]}"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # Gate 6: Total exposure limits
            exposure_result = self._check_exposure_limits(order)
            if not exposure_result[0]:
                if exposure_result[2]:  # Has suggested size reduction
                    decision = RiskDecision.REDUCE_SIZE
                    reason = f"EXPOSURE_LIMIT: {exposure_result[1]}"
                    adjusted_size = exposure_result[2]
                    self._log_risk_decision(order, decision, reason, adjusted_size)
                    return decision, reason, adjusted_size
                else:
                    decision = RiskDecision.REJECT
                    reason = f"EXPOSURE_LIMIT: {exposure_result[1]}"
                    self._log_risk_decision(order, decision, reason)
                    return decision, reason, None
            
            # Gate 7: Single position size limits
            position_size_result = self._check_single_position_size(order)
            if not position_size_result[0]:
                if position_size_result[2]:  # Has suggested size reduction
                    decision = RiskDecision.REDUCE_SIZE
                    reason = f"POSITION_SIZE_LIMIT: {position_size_result[1]}"
                    adjusted_size = position_size_result[2]
                    self._log_risk_decision(order, decision, reason, adjusted_size)
                    return decision, reason, adjusted_size
                else:
                    decision = RiskDecision.REJECT
                    reason = f"POSITION_SIZE_LIMIT: {position_size_result[1]}"
                    self._log_risk_decision(order, decision, reason)
                    return decision, reason, None
            
            # Gate 8: Correlation limits
            correlation_result = self._check_correlation_limits(order)
            if not correlation_result[0]:
                decision = RiskDecision.REJECT
                reason = f"CORRELATION_LIMIT: {correlation_result[1]}"
                self._log_risk_decision(order, decision, reason)
                return decision, reason, None
            
            # All gates passed - APPROVE
            decision = RiskDecision.APPROVE
            reason = "ALL_RISK_GATES_PASSED"
            self._log_risk_decision(order, decision, reason)
            
            return decision, reason, None
            
        except Exception as e:
            # Fail-safe: reject on any error
            decision = RiskDecision.REJECT
            reason = f"RISK_EVALUATION_ERROR: {str(e)}"
            logger.error(f"Risk evaluation error: {e}")
            self._log_risk_decision(order, decision, reason)
            return decision, reason, None
            
        finally:
            # Performance tracking
            evaluation_time = time.time() - start_time
            self.total_evaluation_time += evaluation_time
            
            if evaluation_time > 0.010:  # > 10ms
                logger.warning(f"Slow risk evaluation: {evaluation_time:.3f}s for {order.symbol}")
    
    def _check_data_quality(self, market_data: Optional[Dict]) -> Tuple[bool, str]:
        """Check data quality and freshness"""
        if not market_data:
            return False, "No market data provided"
        
        # Check data age
        data_timestamp = market_data.get('timestamp', 0)
        if data_timestamp:
            data_age_minutes = (time.time() - data_timestamp) / 60
            if data_age_minutes > self.limits.max_data_age_minutes:
                return False, f"Data too old: {data_age_minutes:.1f}min > {self.limits.max_data_age_minutes}min"
        
        # Check data completeness
        required_fields = ['price', 'volume', 'spread']
        missing_fields = [field for field in required_fields if field not in market_data or market_data[field] is None]
        
        completeness = (len(required_fields) - len(missing_fields)) / len(required_fields)
        if completeness < self.limits.min_data_completeness:
            return False, f"Data incomplete: {completeness:.2%} < {self.limits.min_data_completeness:.2%}"
        
        return True, "Data quality OK"
    
    def _check_daily_loss_limits(self, order: OrderRequest) -> Tuple[bool, str]:
        """Check daily loss limits"""
        # Check absolute daily loss
        if abs(self.portfolio_state.daily_pnl_usd) > self.limits.max_daily_loss_usd:
            return False, f"Daily loss ${abs(self.portfolio_state.daily_pnl_usd):,.0f} > ${self.limits.max_daily_loss_usd:,.0f}"
        
        # Check percentage daily loss
        daily_loss_percent = abs(self.portfolio_state.daily_pnl_usd) / self.portfolio_state.total_value_usd * 100
        if daily_loss_percent > self.limits.max_daily_loss_percent:
            return False, f"Daily loss {daily_loss_percent:.1f}% > {self.limits.max_daily_loss_percent:.1f}%"
        
        return True, "Daily loss within limits"
    
    def _check_drawdown_limits(self, order: OrderRequest) -> Tuple[bool, str]:
        """Check maximum drawdown limits"""
        if self.portfolio_state.max_drawdown_from_peak > self.limits.max_drawdown_percent:
            return False, f"Max drawdown {self.portfolio_state.max_drawdown_from_peak:.1f}% > {self.limits.max_drawdown_percent:.1f}%"
        
        return True, "Drawdown within limits"
    
    def _check_position_limits(self, order: OrderRequest) -> Tuple[bool, str]:
        """Check position count limits"""
        current_count = self.portfolio_state.position_count
        
        # If opening new position
        if order.symbol not in self.portfolio_state.positions:
            if current_count >= self.limits.max_position_count:
                return False, f"Position count {current_count} >= {self.limits.max_position_count}"
        
        return True, "Position count within limits"
    
    def _check_exposure_limits(self, order: OrderRequest) -> Tuple[bool, str, Optional[float]]:
        """Check total exposure limits with size reduction option"""
        estimated_order_value = self._estimate_order_value(order)
        new_total_exposure = self.portfolio_state.total_exposure_usd + estimated_order_value
        
        if new_total_exposure > self.limits.max_total_exposure_usd:
            # Calculate maximum allowed size
            available_exposure = self.limits.max_total_exposure_usd - self.portfolio_state.total_exposure_usd
            
            if available_exposure <= 0:
                return False, f"No exposure capacity remaining", None
            
            # Suggest reduced size
            size_reduction_factor = available_exposure / estimated_order_value
            if size_reduction_factor > 0.1:  # Only suggest if reduction is reasonable
                adjusted_size = order.size * size_reduction_factor
                return False, f"Exposure ${new_total_exposure:,.0f} > ${self.limits.max_total_exposure_usd:,.0f}", adjusted_size
            else:
                return False, f"Insufficient exposure capacity", None
        
        return True, "Exposure within limits", None
    
    def _check_single_position_size(self, order: OrderRequest) -> Tuple[bool, str, Optional[float]]:
        """Check single position size limits with size reduction option"""
        estimated_order_value = self._estimate_order_value(order)
        
        if estimated_order_value > self.limits.max_single_position_usd:
            # Calculate maximum allowed size
            size_reduction_factor = self.limits.max_single_position_usd / estimated_order_value
            
            if size_reduction_factor > 0.1:  # Only suggest if reduction is reasonable
                adjusted_size = order.size * size_reduction_factor
                return False, f"Position size ${estimated_order_value:,.0f} > ${self.limits.max_single_position_usd:,.0f}", adjusted_size
            else:
                return False, f"Position size too large", None
        
        return True, "Position size within limits", None
    
    def _check_correlation_limits(self, order: OrderRequest) -> Tuple[bool, str]:
        """Check correlation exposure limits"""
        # Simplified correlation check - in production would use actual correlation matrix
        symbol_base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol[:3]
        
        # Count exposure to same base currency
        base_exposure = 0.0
        for symbol, position in self.portfolio_state.positions.items():
            if symbol.startswith(symbol_base):
                base_exposure += position.get('value_usd', 0)
        
        total_base_exposure_percent = base_exposure / self.portfolio_state.total_value_usd
        
        if total_base_exposure_percent > self.limits.max_correlation_exposure:
            return False, f"Base currency exposure {total_base_exposure_percent:.1%} > {self.limits.max_correlation_exposure:.1%}"
        
        return True, "Correlation within limits"
    
    def _estimate_order_value(self, order: OrderRequest) -> float:
        """Estimate order value in USD"""
        # Simplified estimation - in production would use real market data
        if order.price:
            return order.size * order.price
        else:
            # Use last known price or reasonable estimate
            return order.size * 50000  # Conservative BTC estimate
    
    def _log_risk_decision(self, order: OrderRequest, decision: RiskDecision, reason: str, adjusted_size: Optional[float] = None) -> None:
        """Log risk decision for audit trail"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'size': order.size,
                'adjusted_size': adjusted_size,
                'decision': decision.value,
                'reason': reason,
                'portfolio_value': self.portfolio_state.total_value_usd,
                'daily_pnl': self.portfolio_state.daily_pnl_usd,
                'position_count': self.portfolio_state.position_count,
                'total_exposure': self.portfolio_state.total_exposure_usd
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Also log to main logger
            log_level = logging.WARNING if decision == RiskDecision.REJECT else logging.INFO
            logger.log(log_level, f"Risk decision {decision.value}: {reason} for {order.symbol}")
            
            # Track rejections
            if decision in [RiskDecision.REJECT, RiskDecision.EMERGENCY_STOP]:
                self.rejections_count += 1
                
        except Exception as e:
            logger.error(f"Failed to log risk decision: {e}")
    
    def update_portfolio_state(self, new_state: PortfolioState) -> None:
        """Update portfolio state for risk calculations"""
        self.portfolio_state = new_state
        self.portfolio_state.last_update = time.time()
        logger.debug(f"Portfolio state updated: value=${new_state.total_value_usd:,.0f}, positions={new_state.position_count}")
    
    def activate_kill_switch(self, reason: str) -> None:
        """Activate emergency kill switch"""
        self.limits.kill_switch_active = True
        self._save_risk_limits(self.limits)
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Log emergency stop
        emergency_log = {
            'timestamp': datetime.now().isoformat(),
            'event': 'KILL_SWITCH_ACTIVATED',
            'reason': reason,
            'portfolio_value': self.portfolio_state.total_value_usd,
            'daily_pnl': self.portfolio_state.daily_pnl_usd
        }
        
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(emergency_log) + '\n')
    
    def deactivate_kill_switch(self, reason: str) -> None:
        """Deactivate kill switch (requires manual intervention)"""
        self.limits.kill_switch_active = False
        self._save_risk_limits(self.limits)
        
        logger.warning(f"KILL SWITCH DEACTIVATED: {reason}")
        
        # Log reactivation
        reactivation_log = {
            'timestamp': datetime.now().isoformat(),
            'event': 'KILL_SWITCH_DEACTIVATED',
            'reason': reason
        }
        
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(reactivation_log) + '\n')
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk guard performance metrics"""
        avg_evaluation_time = self.total_evaluation_time / max(self.evaluation_count, 1)
        rejection_rate = self.rejections_count / max(self.evaluation_count, 1)
        
        return {
            'evaluation_count': self.evaluation_count,
            'rejection_count': self.rejections_count,
            'rejection_rate': rejection_rate,
            'avg_evaluation_time_ms': avg_evaluation_time * 1000,
            'kill_switch_active': self.limits.kill_switch_active,
            'portfolio_value': self.portfolio_state.total_value_usd,
            'daily_pnl': self.portfolio_state.daily_pnl_usd,
            'position_count': self.portfolio_state.position_count,
            'total_exposure': self.portfolio_state.total_exposure_usd,
            'last_update': self.portfolio_state.last_update
        }


# Global risk guard instance
central_risk_guard = CentralRiskGuard()


def get_central_risk_guard() -> CentralRiskGuard:
    """Get global central risk guard instance"""
    return central_risk_guard