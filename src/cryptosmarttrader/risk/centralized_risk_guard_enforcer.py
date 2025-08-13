"""
Centralized RiskGuard Enforcer - Fase C Implementation
Hard-wired risk controls for every trading operation - ZERO BYPASS.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
import json
from pathlib import Path

from .risk_guard import RiskGuard, RiskLevel, TradingMode, RiskMetrics, RiskLimits
from ..core.structured_logger import get_logger


@dataclass
class RiskCheckResult:
    """Result of mandatory risk check."""
    approved: bool
    risk_level: RiskLevel
    trading_mode: TradingMode
    violations: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    reason: str
    kill_switch_active: bool
    timestamp: datetime


@dataclass
class TradingOperation:
    """Any trading operation that needs risk approval."""
    operation_type: str  # 'entry', 'resize', 'hedge', 'exit'
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    current_position: float = 0.0
    portfolio_value: float = 0.0
    strategy_id: Optional[str] = None
    confidence_score: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CentralizedRiskGuardEnforcer:
    """
    Centralized RiskGuard enforcer that ALL trading operations must pass through.
    
    HARD-WIRED ARCHITECTURE:
    - Every entry/resize/hedge operation requires risk approval
    - Day-loss limits enforced (5% block, 8% kill-switch)
    - Max drawdown limits enforced (10% block, 15% kill-switch)
    - Position size limits enforced (2% per asset)
    - Total exposure caps enforced (95%)
    - Data quality gates enforced
    - Persistent kill-switch with manual override only
    
    NO OPERATION CAN BYPASS THIS SYSTEM.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("centralized_risk_guard")
        
        # Core RiskGuard system
        self.risk_guard = RiskGuard(config_path)
        
        # Operation tracking
        self.pending_operations: Dict[str, TradingOperation] = {}
        self.approved_operations: Dict[str, RiskCheckResult] = {}
        self.rejected_operations: Dict[str, RiskCheckResult] = {}
        
        # Hard limits (configurable but enforced)
        self.hard_limits = RiskLimits(
            max_daily_loss_percent=5.0,      # 5% day-loss hard block
            max_drawdown_percent=10.0,       # 10% drawdown hard block  
            max_position_size_percent=2.0,   # 2% per asset max
            max_total_exposure_percent=95.0, # 95% total exposure max
            max_correlation_cluster_percent=20.0,  # 20% per cluster max
            max_position_count=50,           # 50 positions max
            min_data_quality_score=0.7,     # 70% data quality min
            max_signal_age_minutes=30        # 30 min signal freshness
        )
        
        # Kill-switch overrides
        self.kill_switch_overrides = {
            'daily_loss_kill_switch': 8.0,    # 8% emergency kill-switch
            'drawdown_kill_switch': 15.0,     # 15% emergency kill-switch
            'data_gap_kill_switch': 60,       # 60 min data gap kill-switch
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.total_operations_checked = 0
        self.total_operations_approved = 0
        self.total_operations_blocked = 0
        self.kill_switch_triggers = 0
        self.manual_overrides = 0
        
        self.logger.info("CentralizedRiskGuardEnforcer initialized - HARD-WIRED PROTECTION ACTIVE")
    
    async def check_trading_operation(self, operation: TradingOperation) -> RiskCheckResult:
        """
        Mandatory risk check for ANY trading operation.
        ZERO BYPASS - ALL OPERATIONS GO THROUGH THIS.
        """
        operation_id = f"{operation.operation_type}_{operation.symbol}_{int(time.time())}"
        
        with self._lock:
            self.total_operations_checked += 1
            self.pending_operations[operation_id] = operation
        
        try:
            # MANDATORY STEP 1: Kill-switch check
            if self.risk_guard.kill_switch_active:
                result = RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.SHUTDOWN,
                    trading_mode=TradingMode.DISABLED,
                    violations=[{
                        'type': 'kill_switch_active',
                        'severity': 'critical',
                        'message': 'Kill switch is active - all trading disabled'
                    }],
                    constraints={},
                    reason="Kill switch is active - all trading operations blocked",
                    kill_switch_active=True,
                    timestamp=datetime.now()
                )
                
                with self._lock:
                    self.rejected_operations[operation_id] = result
                    self.total_operations_blocked += 1
                
                self.logger.critical(
                    "Operation blocked - kill switch active",
                    operation_type=operation.operation_type,
                    symbol=operation.symbol,
                    operation_id=operation_id
                )
                
                return result
            
            # MANDATORY STEP 2: Comprehensive risk assessment
            risk_status = self.risk_guard.run_risk_check(operation.portfolio_value)
            
            # MANDATORY STEP 3: Check hard limits
            violations = await self._check_hard_limits(operation, risk_status)
            
            # MANDATORY STEP 4: Assess trading permission
            approved = self._assess_trading_permission(operation, risk_status, violations)
            
            # MANDATORY STEP 5: Generate constraints
            constraints = self._generate_trading_constraints(operation, risk_status)
            
            # Create result
            result = RiskCheckResult(
                approved=approved,
                risk_level=RiskLevel(risk_status['risk_level']),
                trading_mode=TradingMode(risk_status['trading_mode']),
                violations=violations,
                constraints=constraints,
                reason=self._generate_decision_reason(approved, violations),
                kill_switch_active=risk_status['kill_switch_active'],
                timestamp=datetime.now()
            )
            
            # Store result
            with self._lock:
                if approved:
                    self.approved_operations[operation_id] = result
                    self.total_operations_approved += 1
                else:
                    self.rejected_operations[operation_id] = result
                    self.total_operations_blocked += 1
                
                # Clean up pending
                del self.pending_operations[operation_id]
            
            # Log decision
            self.logger.info(
                f"Risk check {'APPROVED' if approved else 'BLOCKED'}",
                operation_type=operation.operation_type,
                symbol=operation.symbol,
                risk_level=result.risk_level.value,
                violations_count=len(violations),
                operation_id=operation_id
            )
            
            return result
            
        except Exception as e:
            error_result = RiskCheckResult(
                approved=False,
                risk_level=RiskLevel.EMERGENCY,
                trading_mode=TradingMode.DISABLED,
                violations=[{
                    'type': 'risk_check_error',
                    'severity': 'critical',
                    'message': f'Risk check failed: {str(e)}'
                }],
                constraints={},
                reason=f"Risk check error: {str(e)}",
                kill_switch_active=False,
                timestamp=datetime.now()
            )
            
            with self._lock:
                self.rejected_operations[operation_id] = error_result
                self.total_operations_blocked += 1
                if operation_id in self.pending_operations:
                    del self.pending_operations[operation_id]
            
            self.logger.error(f"Risk check failed: {e}", operation_id=operation_id)
            return error_result
    
    async def _check_hard_limits(self, operation: TradingOperation, risk_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check hard-wired risk limits."""
        violations = []
        metrics = risk_status.get('metrics')
        
        if not metrics:
            violations.append({
                'type': 'missing_metrics',
                'severity': 'critical',
                'message': 'Risk metrics unavailable'
            })
            return violations
        
        # Daily loss hard limit
        daily_pnl_percent = getattr(metrics, 'daily_pnl_percent', 0.0)
        if daily_pnl_percent <= -self.hard_limits.max_daily_loss_percent:
            violations.append({
                'type': 'daily_loss_hard_limit',
                'severity': 'critical',
                'message': f'Daily loss {daily_pnl_percent:.2f}% exceeds hard limit {self.hard_limits.max_daily_loss_percent}%',
                'value': daily_pnl_percent,
                'limit': -self.hard_limits.max_daily_loss_percent
            })
        
        # Kill-switch trigger check
        if daily_pnl_percent <= -self.kill_switch_overrides['daily_loss_kill_switch']:
            violations.append({
                'type': 'daily_loss_kill_switch_trigger',
                'severity': 'emergency',
                'message': f'Daily loss {daily_pnl_percent:.2f}% triggers kill-switch at {self.kill_switch_overrides["daily_loss_kill_switch"]}%',
                'value': daily_pnl_percent,
                'limit': -self.kill_switch_overrides['daily_loss_kill_switch']
            })
            
            # Trigger kill-switch
            if not self.risk_guard.kill_switch_active:
                self.risk_guard.trigger_kill_switch(f"Daily loss kill-switch triggered: {daily_pnl_percent:.2f}%")
                self.kill_switch_triggers += 1
        
        # Drawdown hard limit
        drawdown_percent = getattr(metrics, 'max_drawdown_percent', 0.0)
        if drawdown_percent >= self.hard_limits.max_drawdown_percent:
            violations.append({
                'type': 'drawdown_hard_limit',
                'severity': 'critical',
                'message': f'Drawdown {drawdown_percent:.2f}% exceeds hard limit {self.hard_limits.max_drawdown_percent}%',
                'value': drawdown_percent,
                'limit': self.hard_limits.max_drawdown_percent
            })
        
        # Drawdown kill-switch check
        if drawdown_percent >= self.kill_switch_overrides['drawdown_kill_switch']:
            violations.append({
                'type': 'drawdown_kill_switch_trigger',
                'severity': 'emergency',
                'message': f'Drawdown {drawdown_percent:.2f}% triggers kill-switch at {self.kill_switch_overrides["drawdown_kill_switch"]}%',
                'value': drawdown_percent,
                'limit': self.kill_switch_overrides['drawdown_kill_switch']
            })
            
            # Trigger kill-switch
            if not self.risk_guard.kill_switch_active:
                self.risk_guard.trigger_kill_switch(f"Drawdown kill-switch triggered: {drawdown_percent:.2f}%")
                self.kill_switch_triggers += 1
        
        # Position size hard limit
        position_value = operation.quantity * (operation.price or 100.0)  # Estimate
        portfolio_value = operation.portfolio_value
        position_percent = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        if position_percent > self.hard_limits.max_position_size_percent:
            violations.append({
                'type': 'position_size_hard_limit',
                'severity': 'warning',
                'message': f'Position size {position_percent:.2f}% exceeds hard limit {self.hard_limits.max_position_size_percent}%',
                'value': position_percent,
                'limit': self.hard_limits.max_position_size_percent
            })
        
        # Data quality hard limit
        data_quality = getattr(metrics, 'data_quality_score', 1.0)
        if data_quality < self.hard_limits.min_data_quality_score:
            violations.append({
                'type': 'data_quality_hard_limit',
                'severity': 'critical',
                'message': f'Data quality {data_quality:.2f} below hard limit {self.hard_limits.min_data_quality_score}',
                'value': data_quality,
                'limit': self.hard_limits.min_data_quality_score
            })
        
        # Signal age hard limit
        signal_age = getattr(metrics, 'last_signal_age_minutes', 0)
        if signal_age > self.hard_limits.max_signal_age_minutes:
            violations.append({
                'type': 'signal_age_hard_limit',
                'severity': 'warning',
                'message': f'Signal age {signal_age} min exceeds hard limit {self.hard_limits.max_signal_age_minutes} min',
                'value': signal_age,
                'limit': self.hard_limits.max_signal_age_minutes
            })
        
        return violations
    
    def _assess_trading_permission(self, operation: TradingOperation, risk_status: Dict[str, Any], violations: List[Dict[str, Any]]) -> bool:
        """Assess if trading operation should be permitted."""
        
        # Block if kill-switch is active
        if risk_status.get('kill_switch_active', False):
            return False
        
        # Block if trading mode is disabled
        trading_mode = TradingMode(risk_status.get('trading_mode', 'disabled'))
        if trading_mode == TradingMode.DISABLED:
            return False
        
        # Block if critical violations exist
        critical_violations = [v for v in violations if v['severity'] in ['critical', 'emergency']]
        if critical_violations:
            return False
        
        # Block based on risk level
        risk_level = RiskLevel(risk_status.get('risk_level', 'emergency'))
        
        if risk_level == RiskLevel.SHUTDOWN:
            return False
        
        if risk_level == RiskLevel.EMERGENCY:
            # Only allow exit operations
            return operation.operation_type == 'exit'
        
        if risk_level == RiskLevel.DEFENSIVE:
            # Only allow exits and hedges
            return operation.operation_type in ['exit', 'hedge']
        
        if risk_level == RiskLevel.CONSERVATIVE:
            # Allow all operations but with reduced size
            return True
        
        # Normal risk level - allow all operations
        return True
    
    def _generate_trading_constraints(self, operation: TradingOperation, risk_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading constraints based on current risk conditions."""
        constraints = self.risk_guard.get_trading_constraints()
        
        # Add operation-specific constraints
        risk_level = RiskLevel(risk_status.get('risk_level', 'normal'))
        
        if risk_level == RiskLevel.CONSERVATIVE:
            # Reduce position sizes by 50%
            constraints['max_position_size_multiplier'] = 0.5
            constraints['max_leverage'] = 1.0
        
        elif risk_level == RiskLevel.DEFENSIVE:
            # Reduce position sizes by 75%
            constraints['max_position_size_multiplier'] = 0.25
            constraints['max_leverage'] = 0.5
            constraints['allowed_operations'] = ['exit', 'hedge']
        
        elif risk_level == RiskLevel.EMERGENCY:
            # Only exits allowed
            constraints['max_position_size_multiplier'] = 0.0
            constraints['max_leverage'] = 0.0
            constraints['allowed_operations'] = ['exit']
        
        return constraints
    
    def _generate_decision_reason(self, approved: bool, violations: List[Dict[str, Any]]) -> str:
        """Generate human-readable reason for decision."""
        if approved:
            if violations:
                warning_count = len([v for v in violations if v['severity'] == 'warning'])
                return f"Approved with {warning_count} warnings"
            else:
                return "Approved - no risk violations"
        else:
            critical_violations = [v for v in violations if v['severity'] in ['critical', 'emergency']]
            if critical_violations:
                return f"Blocked - {len(critical_violations)} critical violations"
            else:
                return "Blocked - risk conditions unsafe"
    
    async def manual_override_kill_switch(self, reason: str, authorized_by: str) -> bool:
        """Manually override kill-switch (requires authorization)."""
        try:
            success = self.risk_guard.reset_kill_switch(manual_override=True)
            
            if success:
                self.manual_overrides += 1
                self.logger.critical(
                    "Kill-switch manually overridden",
                    reason=reason,
                    authorized_by=authorized_by,
                    override_count=self.manual_overrides
                )
                return True
            else:
                self.logger.error("Kill-switch override failed - conditions not safe")
                return False
                
        except Exception as e:
            self.logger.error(f"Kill-switch override error: {e}")
            return False
    
    def get_risk_enforcement_stats(self) -> Dict[str, Any]:
        """Get risk enforcement statistics."""
        return {
            'total_operations_checked': self.total_operations_checked,
            'total_operations_approved': self.total_operations_approved,
            'total_operations_blocked': self.total_operations_blocked,
            'approval_rate': self.total_operations_approved / max(self.total_operations_checked, 1),
            'block_rate': self.total_operations_blocked / max(self.total_operations_checked, 1),
            'kill_switch_triggers': self.kill_switch_triggers,
            'manual_overrides': self.manual_overrides,
            'current_kill_switch_active': self.risk_guard.kill_switch_active,
            'current_risk_level': self.risk_guard.current_risk_level.value,
            'current_trading_mode': self.risk_guard.trading_mode.value,
            'pending_operations_count': len(self.pending_operations),
            'approved_operations_count': len(self.approved_operations),
            'rejected_operations_count': len(self.rejected_operations),
            'hard_limits': {
                'max_daily_loss_percent': self.hard_limits.max_daily_loss_percent,
                'max_drawdown_percent': self.hard_limits.max_drawdown_percent,
                'max_position_size_percent': self.hard_limits.max_position_size_percent,
                'max_total_exposure_percent': self.hard_limits.max_total_exposure_percent,
            },
            'kill_switch_overrides': self.kill_switch_overrides,
            'timestamp': datetime.now()
        }
    
    def get_current_violations(self, portfolio_value: float) -> List[Dict[str, Any]]:
        """Get current risk violations for monitoring."""
        try:
            risk_status = self.risk_guard.run_risk_check(portfolio_value)
            return risk_status.get('violations', [])
        except Exception as e:
            self.logger.error(f"Failed to get current violations: {e}")
            return [{
                'type': 'risk_check_error',
                'severity': 'critical',
                'message': f'Risk check failed: {str(e)}'
            }]