#!/usr/bin/env python3
"""
Centralized Risk Guard
Enterprise risk management met kill-switch, daily loss limits, max drawdown, exposure limits
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class KillSwitchState(Enum):
    ACTIVE = "active"           # Normal trading
    SOFT_STOP = "soft_stop"     # Reduce position only
    HARD_STOP = "hard_stop"     # Stop all trading
    EMERGENCY = "emergency"     # Emergency shutdown


@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    # Daily loss limits
    daily_loss_limit_usd: float = 10000.0
    daily_loss_warning_pct: float = 0.7  # Warning at 70%
    
    # Drawdown limits
    max_drawdown_pct: float = 0.15       # 15% max drawdown
    drawdown_warning_pct: float = 0.10   # Warning at 10%
    
    # Exposure limits
    max_total_exposure_usd: float = 100000.0
    max_single_position_pct: float = 0.20  # 20% of portfolio
    max_correlation_exposure_pct: float = 0.50  # 50% in correlated assets
    
    # Position limits
    max_open_positions: int = 10
    max_position_size_usd: float = 20000.0
    
    # Data quality limits
    max_data_gap_minutes: int = 5
    min_data_quality_score: float = 0.8
    
    # Operational limits
    max_order_retries: int = 3
    max_slippage_bps: int = 50  # 50 basis points
    max_latency_ms: float = 500.0


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # PnL metrics
    daily_pnl_usd: float = 0.0
    total_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Exposure metrics
    total_exposure_usd: float = 0.0
    long_exposure_usd: float = 0.0
    short_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0
    
    # Position metrics
    open_positions: int = 0
    largest_position_pct: float = 0.0
    correlation_exposure_pct: float = 0.0
    
    # Data quality metrics
    last_data_update: Optional[datetime] = None
    data_gap_minutes: float = 0.0
    data_quality_score: float = 1.0
    
    # Operational metrics
    avg_latency_ms: float = 0.0
    recent_slippage_bps: float = 0.0
    failed_orders_count: int = 0


@dataclass
class RiskViolation:
    """Risk violation record"""
    timestamp: datetime
    violation_type: str
    current_value: float
    limit_value: float
    severity: RiskLevel
    description: str
    auto_action_taken: Optional[str] = None


class CentralizedRiskGuard:
    """Centralized risk management system with kill-switch"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.kill_switch_state = KillSwitchState.ACTIVE
        self.risk_level = RiskLevel.NORMAL
        self.is_monitoring = False
        
        # Metrics tracking
        self.current_metrics = RiskMetrics()
        self.risk_violations: List[RiskViolation] = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        
    def add_alert_callback(self, callback: callable):
        """Add alert callback for risk violations"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start risk monitoring"""
        self.is_monitoring = True
        self.logger.info("🚦 Risk Guard monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.is_monitoring = False
        self.logger.info("🛑 Risk Guard monitoring stopped")
    
    def update_metrics(self, metrics: RiskMetrics):
        """Update current risk metrics"""
        with self._lock:
            self.current_metrics = metrics
            self.current_metrics.timestamp = datetime.now()
            
            # Check for daily reset
            if datetime.now().date() > self.daily_reset_time.date():
                self._reset_daily_metrics()
            
            # Evaluate risk levels
            self._evaluate_risk_levels()
    
    def _reset_daily_metrics(self):
        """Reset daily metrics at midnight"""
        self.logger.info("🌅 Resetting daily risk metrics")
        self.current_metrics.daily_pnl_usd = 0.0
        self.current_metrics.failed_orders_count = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _evaluate_risk_levels(self):
        """Evaluate current risk levels and trigger alerts"""
        violations = []
        
        # Check daily loss limit
        if abs(self.current_metrics.daily_pnl_usd) > self.limits.daily_loss_limit_usd:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="daily_loss_limit",
                current_value=abs(self.current_metrics.daily_pnl_usd),
                limit_value=self.limits.daily_loss_limit_usd,
                severity=RiskLevel.EMERGENCY,
                description=f"Daily loss limit exceeded: ${abs(self.current_metrics.daily_pnl_usd):,.2f}",
                auto_action_taken="EMERGENCY_STOP"
            )
            violations.append(violation)
            self._trigger_kill_switch(KillSwitchState.EMERGENCY, "Daily loss limit exceeded")
        
        # Check daily loss warning
        elif abs(self.current_metrics.daily_pnl_usd) > self.limits.daily_loss_limit_usd * self.limits.daily_loss_warning_pct:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="daily_loss_warning",
                current_value=abs(self.current_metrics.daily_pnl_usd),
                limit_value=self.limits.daily_loss_limit_usd * self.limits.daily_loss_warning_pct,
                severity=RiskLevel.WARNING,
                description=f"Daily loss warning: ${abs(self.current_metrics.daily_pnl_usd):,.2f}",
            )
            violations.append(violation)
        
        # Check drawdown limit
        if self.current_metrics.current_drawdown_pct > self.limits.max_drawdown_pct:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="max_drawdown",
                current_value=self.current_metrics.current_drawdown_pct,
                limit_value=self.limits.max_drawdown_pct,
                severity=RiskLevel.CRITICAL,
                description=f"Max drawdown exceeded: {self.current_metrics.current_drawdown_pct:.1%}",
                auto_action_taken="HARD_STOP"
            )
            violations.append(violation)
            self._trigger_kill_switch(KillSwitchState.HARD_STOP, "Max drawdown exceeded")
        
        # Check exposure limits
        if self.current_metrics.total_exposure_usd > self.limits.max_total_exposure_usd:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="max_exposure",
                current_value=self.current_metrics.total_exposure_usd,
                limit_value=self.limits.max_total_exposure_usd,
                severity=RiskLevel.CRITICAL,
                description=f"Max exposure exceeded: ${self.current_metrics.total_exposure_usd:,.2f}",
                auto_action_taken="SOFT_STOP"
            )
            violations.append(violation)
            self._trigger_kill_switch(KillSwitchState.SOFT_STOP, "Max exposure exceeded")
        
        # Check position count
        if self.current_metrics.open_positions > self.limits.max_open_positions:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="max_positions",
                current_value=self.current_metrics.open_positions,
                limit_value=self.limits.max_open_positions,
                severity=RiskLevel.WARNING,
                description=f"Too many positions: {self.current_metrics.open_positions}"
            )
            violations.append(violation)
        
        # Check data gaps
        if self.current_metrics.data_gap_minutes > self.limits.max_data_gap_minutes:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="data_gap",
                current_value=self.current_metrics.data_gap_minutes,
                limit_value=self.limits.max_data_gap_minutes,
                severity=RiskLevel.CRITICAL,
                description=f"Data gap detected: {self.current_metrics.data_gap_minutes:.1f} minutes",
                auto_action_taken="HARD_STOP"
            )
            violations.append(violation)
            self._trigger_kill_switch(KillSwitchState.HARD_STOP, "Critical data gap")
        
        # Check data quality
        if self.current_metrics.data_quality_score < self.limits.min_data_quality_score:
            violation = RiskViolation(
                timestamp=datetime.now(),
                violation_type="data_quality",
                current_value=self.current_metrics.data_quality_score,
                limit_value=self.limits.min_data_quality_score,
                severity=RiskLevel.WARNING,
                description=f"Poor data quality: {self.current_metrics.data_quality_score:.1%}"
            )
            violations.append(violation)
        
        # Process violations
        for violation in violations:
            self._process_violation(violation)
        
        # Update overall risk level
        if violations:
            # Compare enum values using their ordering
            severity_levels = [RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]
            max_severity = violations[0].severity
            for violation in violations[1:]:
                if severity_levels.index(violation.severity) > severity_levels.index(max_severity):
                    max_severity = violation.severity
            self.risk_level = max_severity
        else:
            self.risk_level = RiskLevel.NORMAL
    
    def _trigger_kill_switch(self, state: KillSwitchState, reason: str):
        """Trigger kill switch"""
        if self.kill_switch_state != state:
            self.kill_switch_state = state
            self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {state.value} - {reason}")
            
            # Send alerts
            for callback in self.alert_callbacks:
                try:
                    callback({
                        'type': 'kill_switch',
                        'state': state.value,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def _process_violation(self, violation: RiskViolation):
        """Process risk violation"""
        self.risk_violations.append(violation)
        
        # Keep only last 1000 violations
        if len(self.risk_violations) > 1000:
            self.risk_violations = self.risk_violations[-1000:]
        
        # Log violation
        self.logger.warning(
            f"🚨 Risk violation: {violation.violation_type} - {violation.description}"
        )
        
        # Send alerts
        for callback in self.alert_callbacks:
            try:
                callback({
                    'type': 'risk_violation',
                    'violation': {
                        'type': violation.violation_type,
                        'description': violation.description,
                        'severity': violation.severity.value,
                        'current_value': violation.current_value,
                        'limit_value': violation.limit_value,
                        'auto_action': violation.auto_action_taken
                    },
                    'timestamp': violation.timestamp.isoformat()
                })
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def check_order_allowed(self, order: Dict[str, Any]) -> tuple[bool, str]:
        """Check if order is allowed given current risk state"""
        with self._lock:
            # Check kill switch state
            if self.kill_switch_state == KillSwitchState.EMERGENCY:
                return False, "Emergency stop - all trading halted"
            
            if self.kill_switch_state == KillSwitchState.HARD_STOP:
                return False, "Hard stop - trading halted"
            
            if self.kill_switch_state == KillSwitchState.SOFT_STOP:
                # Only allow position-reducing orders
                side = order.get('side', '').lower()
                symbol = order.get('symbol', '')
                current_position = self.positions.get(symbol, {}).get('size', 0)
                
                if side == 'buy' and current_position >= 0:
                    return False, "Soft stop - only position-reducing orders allowed"
                if side == 'sell' and current_position <= 0:
                    return False, "Soft stop - only position-reducing orders allowed"
            
            # Check position size limits
            order_size_usd = abs(order.get('size', 0)) * order.get('price', 0)
            if order_size_usd > self.limits.max_position_size_usd:
                return False, f"Order size exceeds limit: ${order_size_usd:,.2f} > ${self.limits.max_position_size_usd:,.2f}"
            
            # Check if would exceed position count
            if order.get('side', '').lower() in ['buy', 'sell']:
                symbol = order.get('symbol', '')
                if symbol not in self.positions and self.current_metrics.open_positions >= self.limits.max_open_positions:
                    return False, f"Would exceed max positions: {self.limits.max_open_positions}"
            
            # Check data quality
            if self.current_metrics.data_quality_score < self.limits.min_data_quality_score:
                return False, f"Data quality too low: {self.current_metrics.data_quality_score:.1%}"
            
            # Check data freshness
            if self.current_metrics.data_gap_minutes > self.limits.max_data_gap_minutes:
                return False, f"Data too stale: {self.current_metrics.data_gap_minutes:.1f} minutes"
            
            return True, "Order allowed"
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position data"""
        with self._lock:
            self.positions[symbol] = position_data
            self._recalculate_exposure_metrics()
    
    def remove_position(self, symbol: str):
        """Remove position"""
        with self._lock:
            if symbol in self.positions:
                del self.positions[symbol]
                self._recalculate_exposure_metrics()
    
    def _recalculate_exposure_metrics(self):
        """Recalculate exposure metrics from positions"""
        total_long = 0.0
        total_short = 0.0
        position_count = 0
        largest_position_pct = 0.0
        
        for symbol, position in self.positions.items():
            size = position.get('size', 0)
            price = position.get('mark_price', position.get('entry_price', 0))
            notional = abs(size * price)
            
            if size > 0:
                total_long += notional
            elif size < 0:
                total_short += notional
            
            if notional > 0:
                position_count += 1
                
            # Calculate position percentage (needs portfolio value)
            if hasattr(self, 'portfolio_value') and self.portfolio_value > 0:
                position_pct = notional / self.portfolio_value
                largest_position_pct = max(largest_position_pct, position_pct)
        
        self.current_metrics.long_exposure_usd = total_long
        self.current_metrics.short_exposure_usd = total_short
        self.current_metrics.total_exposure_usd = total_long + total_short
        self.current_metrics.net_exposure_usd = total_long - total_short
        self.current_metrics.open_positions = position_count
        self.current_metrics.largest_position_pct = largest_position_pct
    
    def manual_kill_switch(self, state: KillSwitchState, reason: str):
        """Manually trigger kill switch"""
        self._trigger_kill_switch(state, f"Manual trigger: {reason}")
    
    def reset_kill_switch(self, reason: str):
        """Reset kill switch to active"""
        if self.kill_switch_state != KillSwitchState.ACTIVE:
            self.kill_switch_state = KillSwitchState.ACTIVE
            self.logger.info(f"✅ Kill switch reset to ACTIVE: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        with self._lock:
            recent_violations = [
                {
                    'type': v.violation_type,
                    'description': v.description,
                    'severity': v.severity.value,
                    'timestamp': v.timestamp.isoformat(),
                    'auto_action': v.auto_action_taken
                }
                for v in self.risk_violations[-10:]  # Last 10 violations
            ]
            
            return {
                'kill_switch_state': self.kill_switch_state.value,
                'risk_level': self.risk_level.value,
                'is_monitoring': self.is_monitoring,
                'metrics': {
                    'daily_pnl_usd': self.current_metrics.daily_pnl_usd,
                    'current_drawdown_pct': self.current_metrics.current_drawdown_pct,
                    'total_exposure_usd': self.current_metrics.total_exposure_usd,
                    'open_positions': self.current_metrics.open_positions,
                    'data_gap_minutes': self.current_metrics.data_gap_minutes,
                    'data_quality_score': self.current_metrics.data_quality_score
                },
                'limits': {
                    'daily_loss_limit_usd': self.limits.daily_loss_limit_usd,
                    'max_drawdown_pct': self.limits.max_drawdown_pct,
                    'max_total_exposure_usd': self.limits.max_total_exposure_usd,
                    'max_open_positions': self.limits.max_open_positions
                },
                'recent_violations': recent_violations,
                'position_count': len(self.positions),
                'last_update': self.current_metrics.timestamp.isoformat()
            }


# Singleton instance for global access
_risk_guard_instance: Optional[CentralizedRiskGuard] = None

def get_risk_guard() -> CentralizedRiskGuard:
    """Get singleton risk guard instance"""
    global _risk_guard_instance
    if _risk_guard_instance is None:
        limits = RiskLimits()
        _risk_guard_instance = CentralizedRiskGuard(limits)
    return _risk_guard_instance

def initialize_risk_guard(limits: RiskLimits) -> CentralizedRiskGuard:
    """Initialize risk guard with custom limits"""
    global _risk_guard_instance
    _risk_guard_instance = CentralizedRiskGuard(limits)
    return _risk_guard_instance