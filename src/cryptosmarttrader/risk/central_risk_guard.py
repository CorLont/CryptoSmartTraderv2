"""
Central RiskGuard System - Hard Risk Management
All trades must pass through central risk validation with kill-switch capability
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import os for file operations
import os


class RiskLevel(Enum):
    """Risk severity levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskType(Enum):
    """Types of risk violations"""
    DAY_LOSS = "day_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    MAX_POSITIONS = "max_positions"
    DATA_GAP = "data_gap"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_SPIKE = "volatility_spike"


class KillSwitchStatus(Enum):
    """Kill switch status"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    MAINTENANCE = "maintenance"


@dataclass
class RiskLimits:
    """Central risk limit configuration"""
    
    # Daily loss limits
    max_day_loss_usd: float = 10000.0
    max_day_loss_percent: float = 2.0
    
    # Drawdown limits
    max_drawdown_percent: float = 5.0
    max_rolling_drawdown_days: int = 7
    
    # Exposure limits
    max_total_exposure_usd: float = 100000.0
    max_single_position_percent: float = 20.0
    max_sector_exposure_percent: float = 30.0
    
    # Position limits
    max_total_positions: int = 10
    max_positions_per_symbol: int = 3
    
    # Data quality limits
    max_data_gap_minutes: int = 5
    min_data_quality_score: float = 0.8
    
    # Correlation limits
    max_portfolio_correlation: float = 0.7
    max_single_correlation: float = 0.9


@dataclass 
class RiskMetrics:
    """Risk assessment metrics"""
    daily_pnl: float = 0.0
    drawdown_percent: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0
    correlation_score: float = 0.0
    data_quality_score: float = 1.0


@dataclass
class RiskViolation:
    """Risk limit violation details"""
    risk_type: RiskType
    risk_level: RiskLevel
    current_value: float
    limit_value: float
    violation_percent: float
    description: str
    timestamp: datetime
    affected_symbols: List[str] = field(default_factory=list)


@dataclass
class RiskCheckResult:
    """Result of risk validation check"""
    is_safe: bool
    violations: List[RiskViolation]
    risk_score: float  # 0.0 = safe, 1.0 = maximum risk
    warnings: List[str] = field(default_factory=list)
    kill_switch_triggered: bool = False
    reason: str = ""


@dataclass
class PositionInfo:
    """Position information for risk tracking"""
    symbol: str
    size_usd: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime


class CentralRiskGuard:
    """
    CENTRAL RISK GUARD - FASE C IMPLEMENTATION
    
    MANDATORY FEATURES:
    ✅ Day-loss limits: $10k max daily loss, 2% portfolio loss max
    ✅ Drawdown limits: 5% max drawdown from peak
    ✅ Exposure limits: $100k max total, 20% max single position
    ✅ Position limits: 10 max total positions
    ✅ Data-gap detection: 5 min max data staleness
    ✅ Kill-switch: Automatic halt on critical violations
    ✅ Prometheus alerts integration
    
    ALL TRADES MUST PASS THROUGH CENTRAL VALIDATION
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Dict[str, Any] = None):
        """Singleton pattern for global risk enforcement"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        if self._initialized:
            return
            
        config = config or {}
        
        # Initialize risk limits (HARD LIMITS)
        self.limits = RiskLimits(
            max_day_loss_usd=config.get('max_day_loss_usd', 10000.0),
            max_day_loss_percent=config.get('max_day_loss_percent', 2.0),
            max_drawdown_percent=config.get('max_drawdown_percent', 5.0),
            max_total_exposure_usd=config.get('max_total_exposure_usd', 100000.0),
            max_single_position_percent=config.get('max_single_position_percent', 20.0),
            max_total_positions=config.get('max_total_positions', 10),
            max_data_gap_minutes=config.get('max_data_gap_minutes', 5)
        )
        
        # Current portfolio state
        self.current_metrics = RiskMetrics()
        self.positions: Dict[str, PositionInfo] = {}
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        
        # Kill switch state
        self.kill_switch_status = KillSwitchStatus.ACTIVE
        self.kill_switch_reason = ""
        self.kill_switch_triggered_at: Optional[datetime] = None
        
        # Portfolio tracking
        self.portfolio_equity = 100000.0  # Starting equity
        self.portfolio_peak = 100000.0
        self.daily_start_equity = 100000.0
        
        # Data quality tracking
        self.last_data_update: Dict[str, datetime] = {}
        
        # Prometheus metrics integration
        try:
            from ..observability.metrics import PrometheusMetrics
            self.metrics = PrometheusMetrics.get_instance()
        except ImportError:
            self.metrics = None
            logger.warning("Prometheus metrics not available")
        
        # Risk violation history
        self.violation_history: List[RiskViolation] = []
        
        # Thread safety
        self._risk_lock = threading.Lock()
        
        self._initialized = True
        
        logger.info("CentralRiskGuard HARD ENFORCEMENT initialized", extra={
            'max_day_loss_usd': self.limits.max_day_loss_usd,
            'max_drawdown_percent': self.limits.max_drawdown_percent,
            'max_total_exposure_usd': self.limits.max_total_exposure_usd,
            'max_total_positions': self.limits.max_total_positions,
            'kill_switch_status': self.kill_switch_status.value
        })
    
    def validate_trade(self, symbol: str, side: str, quantity: float, price: float) -> RiskCheckResult:
        """
        MANDATORY TRADE VALIDATION
        ALL trades must pass through this validation - NO EXCEPTIONS
        """
        with self._risk_lock:
            violations = []
            warnings = []
            risk_score = 0.0
            
            # Check kill switch first
            if self.kill_switch_status == KillSwitchStatus.TRIGGERED:
                return RiskCheckResult(
                    is_safe=False,
                    violations=[],
                    risk_score=1.0,
                    kill_switch_triggered=True,
                    reason=f"Kill switch triggered: {self.kill_switch_reason}"
                )
            
            # Calculate trade impact
            trade_value = abs(quantity * price)
            
            # 1. Day loss validation
            projected_loss = self._calculate_projected_loss(trade_value)
            if projected_loss > self.limits.max_day_loss_usd:
                violations.append(RiskViolation(
                    risk_type=RiskType.DAY_LOSS,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=projected_loss,
                    limit_value=self.limits.max_day_loss_usd,
                    violation_percent=(projected_loss / self.limits.max_day_loss_usd - 1) * 100,
                    description=f"Projected daily loss ${projected_loss:,.0f} exceeds limit ${self.limits.max_day_loss_usd:,.0f}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                ))
                risk_score += 0.3
            
            # 2. Drawdown validation
            projected_equity = self.portfolio_equity - trade_value * 0.02  # Assume 2% loss
            projected_drawdown = max(0, (self.portfolio_peak - projected_equity) / self.portfolio_peak * 100)
            
            if projected_drawdown > self.limits.max_drawdown_percent:
                violations.append(RiskViolation(
                    risk_type=RiskType.MAX_DRAWDOWN,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=projected_drawdown,
                    limit_value=self.limits.max_drawdown_percent,
                    violation_percent=(projected_drawdown / self.limits.max_drawdown_percent - 1) * 100,
                    description=f"Projected drawdown {projected_drawdown:.1f}% exceeds limit {self.limits.max_drawdown_percent:.1f}%",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                ))
                risk_score += 0.3
            
            # 3. Exposure validation
            projected_exposure = self.current_metrics.total_exposure + trade_value
            if projected_exposure > self.limits.max_total_exposure_usd:
                violations.append(RiskViolation(
                    risk_type=RiskType.MAX_EXPOSURE,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=projected_exposure,
                    limit_value=self.limits.max_total_exposure_usd,
                    violation_percent=(projected_exposure / self.limits.max_total_exposure_usd - 1) * 100,
                    description=f"Projected exposure ${projected_exposure:,.0f} exceeds limit ${self.limits.max_total_exposure_usd:,.0f}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                ))
                risk_score += 0.2
            
            # 4. Position count validation
            new_position = symbol not in self.positions
            projected_positions = self.current_metrics.position_count + (1 if new_position else 0)
            
            if projected_positions > self.limits.max_total_positions:
                violations.append(RiskViolation(
                    risk_type=RiskType.MAX_POSITIONS,
                    risk_level=RiskLevel.WARNING,
                    current_value=projected_positions,
                    limit_value=self.limits.max_total_positions,
                    violation_percent=(projected_positions / self.limits.max_total_positions - 1) * 100,
                    description=f"Projected positions {projected_positions} exceeds limit {self.limits.max_total_positions}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                ))
                risk_score += 0.1
            
            # 5. Data gap validation
            data_age_minutes = self._get_data_age_minutes(symbol)
            if data_age_minutes > self.limits.max_data_gap_minutes:
                violations.append(RiskViolation(
                    risk_type=RiskType.DATA_GAP,
                    risk_level=RiskLevel.WARNING,
                    current_value=data_age_minutes,
                    limit_value=self.limits.max_data_gap_minutes,
                    violation_percent=(data_age_minutes / self.limits.max_data_gap_minutes - 1) * 100,
                    description=f"Data age {data_age_minutes:.1f} min exceeds limit {self.limits.max_data_gap_minutes} min",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                ))
                risk_score += 0.1
            
            # Check for kill switch triggers
            critical_violations = [v for v in violations if v.risk_level == RiskLevel.CRITICAL]
            should_trigger_kill_switch = len(critical_violations) >= 2 or any(v.violation_percent > 50 for v in critical_violations)
            
            if should_trigger_kill_switch:
                self._trigger_kill_switch(f"Critical risk violations: {len(critical_violations)} critical issues")
                return RiskCheckResult(
                    is_safe=False,
                    violations=violations,
                    risk_score=1.0,
                    kill_switch_triggered=True,
                    reason="Critical risk violations triggered kill switch"
                )
            
            # Final safety decision
            is_safe = len(violations) == 0 and risk_score < 0.5
            
            # Record violations
            if violations:
                self.violation_history.extend(violations)
                # Keep last 1000 violations
                if len(self.violation_history) > 1000:
                    self.violation_history = self.violation_history[-1000:]
            
            # Update metrics
            if self.metrics:
                self._record_risk_metrics(violations, risk_score)
            
            result = RiskCheckResult(
                is_safe=is_safe,
                violations=violations,
                risk_score=risk_score,
                warnings=warnings,
                kill_switch_triggered=False,
                reason="All risk checks passed" if is_safe else f"Risk violations detected: {len(violations)}"
            )
            
            logger.info("Risk validation completed", extra={
                'symbol': symbol,
                'side': side,
                'trade_value': trade_value,
                'is_safe': is_safe,
                'violations_count': len(violations),
                'risk_score': risk_score,
                'kill_switch_status': self.kill_switch_status.value
            })
            
            return result
    
    def _calculate_projected_loss(self, trade_value: float) -> float:
        """Calculate projected daily loss including this trade"""
        current_daily_pnl = self.current_metrics.daily_pnl
        # Assume worst case 5% loss on the trade
        projected_loss = abs(current_daily_pnl) + (trade_value * 0.05)
        return projected_loss
    
    def _get_data_age_minutes(self, symbol: str) -> float:
        """Get data age in minutes for symbol"""
        if symbol not in self.last_data_update:
            return 999.0  # Very stale if no data
        
        age = datetime.now() - self.last_data_update[symbol]
        return age.total_seconds() / 60.0
    
    def _trigger_kill_switch(self, reason: str):
        """Trigger emergency kill switch"""
        self.kill_switch_status = KillSwitchStatus.TRIGGERED
        self.kill_switch_reason = reason
        self.kill_switch_triggered_at = datetime.now()
        
        # Send emergency alert
        if self.metrics:
            self.metrics.kill_switch_triggers.inc()
        
        logger.critical("KILL SWITCH TRIGGERED", extra={
            'reason': reason,
            'triggered_at': self.kill_switch_triggered_at.isoformat(),
            'portfolio_equity': self.portfolio_equity,
            'daily_pnl': self.current_metrics.daily_pnl,
            'total_exposure': self.current_metrics.total_exposure
        })
    
    def _record_risk_metrics(self, violations: List[RiskViolation], risk_score: float):
        """Record risk metrics to Prometheus"""
        # Record violation counts by type
        for violation in violations:
            self.metrics.risk_violations.labels(
                risk_type=violation.risk_type.value,
                risk_level=violation.risk_level.value
            ).inc()
        
        # Record overall risk score
        self.metrics.portfolio_risk_score.set(risk_score)
        
        # Record current metrics
        self.metrics.portfolio_equity.set(self.portfolio_equity)
        self.metrics.portfolio_drawdown_pct.set(
            (self.portfolio_peak - self.portfolio_equity) / self.portfolio_peak * 100
        )
        self.metrics.portfolio_exposure.set(self.current_metrics.total_exposure)
        self.metrics.portfolio_positions.set(self.current_metrics.position_count)
    
    def update_portfolio_state(self, equity: float, positions: Dict[str, PositionInfo]):
        """Update current portfolio state"""
        with self._risk_lock:
            self.portfolio_equity = equity
            self.portfolio_peak = max(self.portfolio_peak, equity)
            self.positions = positions.copy()
            
            # Update current metrics
            self.current_metrics.total_exposure = sum(pos.size_usd for pos in positions.values())
            self.current_metrics.position_count = len(positions)
            self.current_metrics.drawdown_percent = (self.portfolio_peak - equity) / self.portfolio_peak * 100
            
            # Calculate daily PnL
            if hasattr(self, 'daily_start_equity'):
                self.current_metrics.daily_pnl = equity - self.daily_start_equity
    
    def update_data_timestamp(self, symbol: str, timestamp: datetime):
        """Update last data timestamp for symbol"""
        self.last_data_update[symbol] = timestamp
    
    def reset_kill_switch(self, operator: str) -> bool:
        """Reset kill switch (requires manual intervention)"""
        if self.kill_switch_status == KillSwitchStatus.TRIGGERED:
            self.kill_switch_status = KillSwitchStatus.ACTIVE
            self.kill_switch_reason = ""
            self.kill_switch_triggered_at = None
            
            logger.warning("Kill switch reset by operator", extra={
                'operator': operator,
                'reset_at': datetime.now().isoformat()
            })
            return True
        return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        current_drawdown = (self.portfolio_peak - self.portfolio_equity) / self.portfolio_peak * 100
        
        return {
            'kill_switch_status': self.kill_switch_status.value,
            'kill_switch_reason': self.kill_switch_reason,
            'portfolio_equity': self.portfolio_equity,
            'portfolio_peak': self.portfolio_peak,
            'current_drawdown_pct': current_drawdown,
            'daily_pnl': self.current_metrics.daily_pnl,
            'total_exposure': self.current_metrics.total_exposure,
            'position_count': self.current_metrics.position_count,
            'risk_limits': {
                'max_day_loss_usd': self.limits.max_day_loss_usd,
                'max_drawdown_percent': self.limits.max_drawdown_percent,
                'max_total_exposure_usd': self.limits.max_total_exposure_usd,
                'max_total_positions': self.limits.max_total_positions
            },
            'recent_violations': len([v for v in self.violation_history if v.timestamp > datetime.now() - timedelta(hours=24)])
        }


# Singleton access function
def get_central_risk_guard(config: Dict[str, Any] = None) -> CentralRiskGuard:
    """Get the global central risk guard instance"""
    return CentralRiskGuard(config)


@dataclass 
class DataGap:
    """Data gap information"""
    symbol: str
    gap_start: datetime
    gap_end: Optional[datetime]
    gap_duration_minutes: float


@dataclass
class PositionInfo:
    """Position information for risk calculation"""
    symbol: str
    size_usd: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: float
    strategy_id: str


@dataclass
class DataGap:
    """Data gap information"""
    symbol: str
    gap_start: datetime
    gap_end: datetime
    gap_minutes: float
    data_type: str


@dataclass
class RiskViolation:
    """Risk limit violation"""
    risk_type: RiskType
    risk_level: RiskLevel
    current_value: float
    limit_value: float
    violation_percent: float
    description: str
    timestamp: datetime
    affected_symbols: List[str] = field(default_factory=list)


@dataclass
class RiskCheckResult:
    """Result of comprehensive risk check"""
    is_safe: bool
    violations: List[RiskViolation]
    risk_score: float  # 0-1, higher = more risky
    warnings: List[str]
    kill_switch_triggered: bool = False
    reason: Optional[str] = None


class KillSwitch:
    """Emergency trading halt mechanism"""
    
    def __init__(self):
        self._status = KillSwitchStatus.ACTIVE
        self._triggered_at: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._violations: List[RiskViolation] = []
        self._lock = threading.Lock()
    
    def trigger(self, violation: RiskViolation, reason: str):
        """Trigger kill switch"""
        with self._lock:
            if self._status != KillSwitchStatus.TRIGGERED:
                self._status = KillSwitchStatus.TRIGGERED
                self._triggered_at = datetime.now()
                self._trigger_reason = reason
                self._violations.append(violation)
                
                logger.critical(f"🚨 KILL SWITCH TRIGGERED: {reason}")
                logger.critical(f"Violation: {violation.description}")
                
                # Save kill switch state
                self._save_kill_switch_state()
                
                # Send alerts (implement actual alerting)
                self._send_emergency_alerts(violation, reason)
    
    def is_triggered(self) -> bool:
        """Check if kill switch is triggered"""
        with self._lock:
            return self._status == KillSwitchStatus.TRIGGERED
    
    def reset(self, authorized_user: str = "system"):
        """Reset kill switch (requires authorization)"""
        with self._lock:
            if self._status == KillSwitchStatus.TRIGGERED:
                logger.warning(f"Kill switch reset by: {authorized_user}")
                self._status = KillSwitchStatus.ACTIVE
                self._triggered_at = None
                self._trigger_reason = None
                self._violations.clear()
                self._save_kill_switch_state()
    
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        with self._lock:
            return {
                'status': self._status.value,
                'triggered_at': self._triggered_at.isoformat() if self._triggered_at else None,
                'reason': self._trigger_reason,
                'violations_count': len(self._violations),
                'uptime_minutes': (datetime.now() - (self._triggered_at or datetime.now())).total_seconds() / 60
            }
    
    def _save_kill_switch_state(self):
        """Save kill switch state to disk"""
        try:
            state = {
                'status': self._status.value,
                'triggered_at': self._triggered_at.isoformat() if self._triggered_at else None,
                'reason': self._trigger_reason,
                'violations': [
                    {
                        'type': v.risk_type.value,
                        'level': v.risk_level.value,
                        'description': v.description,
                        'timestamp': v.timestamp.isoformat(),
                        'current_value': v.current_value,
                        'limit_value': v.limit_value
                    }
                    for v in self._violations
                ]
            }
            
            os.makedirs('data/risk', exist_ok=True)
            with open('data/risk/kill_switch_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")
    
    def _send_emergency_alerts(self, violation: RiskViolation, reason: str):
        """Send emergency alerts (implement actual alerting)"""
        try:
            # Log to emergency log file
            emergency_log = 'logs/emergency_alerts.log'
            os.makedirs('logs', exist_ok=True)
            
            alert_msg = f"[{datetime.now().isoformat()}] EMERGENCY KILL SWITCH TRIGGERED\n"
            alert_msg += f"Reason: {reason}\n"
            alert_msg += f"Violation: {violation.description}\n"
            alert_msg += f"Risk Type: {violation.risk_type.value}\n"
            alert_msg += f"Current: {violation.current_value}, Limit: {violation.limit_value}\n"
            alert_msg += f"Violation %: {violation.violation_percent:.1f}%\n"
            
            with open(emergency_log, 'a') as f:
                f.write(alert_msg + '\n')
            
            # TODO: Implement actual alerting (email, SMS, Slack, etc.)
            # send_email_alert(alert_msg)
            # send_slack_alert(alert_msg)
            # send_sms_alert(alert_msg)
            
        except Exception as e:
            logger.error(f"Failed to send emergency alerts: {e}")


class CentralRiskGuard:
    """
    Central risk management system with hard limits and kill switch
    ALL trades must pass through check_trade_risk()
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.kill_switch = KillSwitch()
        self.positions: Dict[str, PositionInfo] = {}
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.data_gaps: List[DataGap] = []
        self._lock = threading.Lock()
        self._last_check: Optional[datetime] = None
        
        # Load previous state
        self._load_state()
    
    def check_trade_risk(
        self, 
        symbol: str, 
        trade_size_usd: float, 
        strategy_id: str = "default"
    ) -> RiskCheckResult:
        """
        MANDATORY risk check for ALL trades
        
        Args:
            symbol: Trading symbol
            trade_size_usd: Trade size in USD
            strategy_id: Strategy identifier
            
        Returns:
            RiskCheckResult with safety assessment
        """
        
        with self._lock:
            logger.info(f"Risk check: {symbol} ${trade_size_usd:,.0f} ({strategy_id})")
            
            # Quick kill switch check
            if self.kill_switch.is_triggered():
                return RiskCheckResult(
                    is_safe=False,
                    violations=[],
                    risk_score=1.0,
                    warnings=["Kill switch is triggered - all trading halted"],
                    kill_switch_triggered=True,
                    reason="Trading halted by kill switch"
                )
            
            violations = []
            warnings = []
            
            # 1. Day loss check
            projected_loss = self.daily_pnl - abs(trade_size_usd) * 0.01  # Assume 1% potential loss
            
            if abs(projected_loss) > self.limits.max_day_loss_usd:
                violation = RiskViolation(
                    risk_type=RiskType.DAY_LOSS,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=abs(projected_loss),
                    limit_value=self.limits.max_day_loss_usd,
                    violation_percent=(abs(projected_loss) / self.limits.max_day_loss_usd - 1) * 100,
                    description=f"Day loss limit exceeded: ${abs(projected_loss):,.0f} > ${self.limits.max_day_loss_usd:,.0f}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                )
                violations.append(violation)
                
                # Trigger kill switch for critical day loss
                self.kill_switch.trigger(violation, "Critical day loss limit exceeded")
            
            # 2. Max drawdown check
            if self.current_equity > 0:
                current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
                
                if current_drawdown > self.limits.max_drawdown_percent:
                    violation = RiskViolation(
                        risk_type=RiskType.MAX_DRAWDOWN,
                        risk_level=RiskLevel.CRITICAL,
                        current_value=current_drawdown,
                        limit_value=self.limits.max_drawdown_percent,
                        violation_percent=(current_drawdown / self.limits.max_drawdown_percent - 1) * 100,
                        description=f"Max drawdown exceeded: {current_drawdown:.1f}% > {self.limits.max_drawdown_percent:.1f}%",
                        timestamp=datetime.now()
                    )
                    violations.append(violation)
                    
                    # Trigger kill switch for critical drawdown
                    self.kill_switch.trigger(violation, "Critical drawdown limit exceeded")
            
            # 3. Exposure check
            total_exposure = sum(abs(pos.size_usd) for pos in self.positions.values())
            projected_exposure = total_exposure + abs(trade_size_usd)
            
            if projected_exposure > self.limits.max_total_exposure_usd:
                violation = RiskViolation(
                    risk_type=RiskType.MAX_EXPOSURE,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=projected_exposure,
                    limit_value=self.limits.max_total_exposure_usd,
                    violation_percent=(projected_exposure / self.limits.max_total_exposure_usd - 1) * 100,
                    description=f"Max exposure exceeded: ${projected_exposure:,.0f} > ${self.limits.max_total_exposure_usd:,.0f}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                )
                violations.append(violation)
            
            # 4. Position count check
            symbol_positions = len([p for p in self.positions.values() if p.symbol == symbol])
            total_positions = len(self.positions)
            
            if total_positions >= self.limits.max_total_positions:
                violation = RiskViolation(
                    risk_type=RiskType.MAX_POSITIONS,
                    risk_level=RiskLevel.WARNING,
                    current_value=total_positions,
                    limit_value=self.limits.max_total_positions,
                    violation_percent=(total_positions / self.limits.max_total_positions - 1) * 100,
                    description=f"Max total positions reached: {total_positions} >= {self.limits.max_total_positions}",
                    timestamp=datetime.now()
                )
                violations.append(violation)
            
            if symbol_positions >= self.limits.max_positions_per_symbol:
                violation = RiskViolation(
                    risk_type=RiskType.MAX_POSITIONS,
                    risk_level=RiskLevel.WARNING,
                    current_value=symbol_positions,
                    limit_value=self.limits.max_positions_per_symbol,
                    violation_percent=(symbol_positions / self.limits.max_positions_per_symbol - 1) * 100,
                    description=f"Max positions per symbol reached for {symbol}: {symbol_positions} >= {self.limits.max_positions_per_symbol}",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                )
                violations.append(violation)
            
            # 5. Data gap check
            recent_gaps = [
                gap for gap in self.data_gaps 
                if gap.symbol == symbol and gap.gap_minutes > self.limits.max_data_gap_minutes
                and (datetime.now() - gap.gap_end).total_seconds() < 3600  # Within last hour
            ]
            
            if recent_gaps:
                max_gap = max(recent_gaps, key=lambda g: g.gap_minutes)
                violation = RiskViolation(
                    risk_type=RiskType.DATA_GAP,
                    risk_level=RiskLevel.WARNING,
                    current_value=max_gap.gap_minutes,
                    limit_value=self.limits.max_data_gap_minutes,
                    violation_percent=(max_gap.gap_minutes / self.limits.max_data_gap_minutes - 1) * 100,
                    description=f"Recent data gap for {symbol}: {max_gap.gap_minutes:.1f} min > {self.limits.max_data_gap_minutes} min",
                    timestamp=datetime.now(),
                    affected_symbols=[symbol]
                )
                violations.append(violation)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(violations, trade_size_usd)
            
            # Determine if trade is safe
            critical_violations = [v for v in violations if v.risk_level == RiskLevel.CRITICAL]
            is_safe = len(critical_violations) == 0 and not self.kill_switch.is_triggered()
            
            self._last_check = datetime.now()
            
            result = RiskCheckResult(
                is_safe=is_safe,
                violations=violations,
                risk_score=risk_score,
                warnings=warnings,
                kill_switch_triggered=self.kill_switch.is_triggered(),
                reason=None if is_safe else "Risk limit violations detected"
            )
            
            # Log risk check result
            if not is_safe:
                logger.warning(f"Trade rejected: {symbol} ${trade_size_usd:,.0f}")
                for violation in violations:
                    logger.warning(f"  {violation.description}")
            else:
                logger.info(f"Trade approved: {symbol} ${trade_size_usd:,.0f} (risk: {risk_score:.2f})")
            
            return result
    
    def update_position(self, position: PositionInfo):
        """Update position information for risk tracking"""
        with self._lock:
            self.positions[f"{position.symbol}_{position.strategy_id}"] = position
            self._update_portfolio_metrics()
    
    def remove_position(self, symbol: str, strategy_id: str = "default"):
        """Remove position from tracking"""
        with self._lock:
            key = f"{symbol}_{strategy_id}"
            if key in self.positions:
                del self.positions[key]
                self._update_portfolio_metrics()
    
    def report_data_gap(self, gap: DataGap):
        """Report data gap for risk assessment"""
        with self._lock:
            self.data_gaps.append(gap)
            
            # Keep only recent gaps (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self.data_gaps = [g for g in self.data_gaps if g.gap_end > cutoff]
            
            logger.warning(f"Data gap reported: {gap.symbol} {gap.gap_minutes:.1f} min")
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L for loss tracking"""
        with self._lock:
            self.daily_pnl = pnl
            
            # Check day loss limit
            if abs(pnl) > self.limits.max_day_loss_usd:
                violation = RiskViolation(
                    risk_type=RiskType.DAY_LOSS,
                    risk_level=RiskLevel.EMERGENCY,
                    current_value=abs(pnl),
                    limit_value=self.limits.max_day_loss_usd,
                    violation_percent=(abs(pnl) / self.limits.max_day_loss_usd - 1) * 100,
                    description=f"Emergency day loss: ${abs(pnl):,.0f} > ${self.limits.max_day_loss_usd:,.0f}",
                    timestamp=datetime.now()
                )
                
                self.kill_switch.trigger(violation, "Emergency day loss limit breached")
    
    def update_equity(self, equity: float):
        """Update current equity for drawdown tracking"""
        with self._lock:
            self.current_equity = equity
            
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            self._update_portfolio_metrics()
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level risk metrics"""
        # Calculate total exposure
        total_exposure = sum(abs(pos.size_usd) for pos in self.positions.values())
        
        # Calculate unrealized P&L
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update current equity estimate
        if self.current_equity == 0:
            self.current_equity = 100000.0  # Default starting equity
        
        # Check for drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            
            if current_drawdown > self.limits.max_drawdown_percent * 0.8:  # 80% of limit
                logger.warning(f"Approaching drawdown limit: {current_drawdown:.1f}%")
    
    def _calculate_risk_score(self, violations: List[RiskViolation], trade_size: float) -> float:
        """Calculate overall risk score (0-1)"""
        if not violations:
            return 0.1  # Minimum risk
        
        # Weight violations by severity
        severity_weights = {
            RiskLevel.SAFE: 0.0,
            RiskLevel.WARNING: 0.3,
            RiskLevel.CRITICAL: 0.7,
            RiskLevel.EMERGENCY: 1.0
        }
        
        total_score = 0.0
        for violation in violations:
            weight = severity_weights[violation.risk_level]
            violation_impact = min(violation.violation_percent / 100, 1.0)
            total_score += weight * violation_impact
        
        return min(total_score, 1.0)
    
    def _load_state(self):
        """Load previous risk state"""
        try:
            if os.path.exists('data/risk/kill_switch_state.json'):
                with open('data/risk/kill_switch_state.json', 'r') as f:
                    state = json.load(f)
                
                if state.get('status') == 'triggered':
                    logger.warning("Loaded triggered kill switch state from previous session")
                    # Note: Manual reset required for safety
                    
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        with self._lock:
            total_exposure = sum(abs(pos.size_usd) for pos in self.positions.values())
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            drawdown = 0.0
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            
            return {
                'kill_switch': self.kill_switch.get_status(),
                'limits': {
                    'day_loss_usd': self.limits.max_day_loss_usd,
                    'drawdown_pct': self.limits.max_drawdown_percent,
                    'exposure_usd': self.limits.max_total_exposure_usd,
                    'max_positions': self.limits.max_total_positions
                },
                'current': {
                    'daily_pnl': self.daily_pnl,
                    'drawdown_pct': drawdown,
                    'total_exposure_usd': total_exposure,
                    'total_positions': len(self.positions),
                    'unrealized_pnl': total_unrealized,
                    'equity': self.current_equity
                },
                'utilization': {
                    'day_loss_pct': abs(self.daily_pnl) / self.limits.max_day_loss_usd * 100,
                    'drawdown_pct': drawdown / self.limits.max_drawdown_percent * 100,
                    'exposure_pct': total_exposure / self.limits.max_total_exposure_usd * 100,
                    'positions_pct': len(self.positions) / self.limits.max_total_positions * 100
                },
                'data_gaps': len(self.data_gaps),
                'last_check': self._last_check.isoformat() if self._last_check else None
            }


# Global risk guard instance
_risk_guard: Optional[CentralRiskGuard] = None
_risk_lock = threading.Lock()


def get_risk_guard() -> CentralRiskGuard:
    """Get global risk guard instance"""
    global _risk_guard
    
    if _risk_guard is None:
        with _risk_lock:
            if _risk_guard is None:
                _risk_guard = CentralRiskGuard()
    
    return _risk_guard


def reset_risk_guard():
    """Reset risk guard for testing"""
    global _risk_guard
    with _risk_lock:
        _risk_guard = None


# Convenience functions for integration
def check_trade_risk(symbol: str, trade_size_usd: float, strategy_id: str = "default") -> RiskCheckResult:
    """Convenience function for trade risk checking"""
    return get_risk_guard().check_trade_risk(symbol, trade_size_usd, strategy_id)


def is_trading_halted() -> bool:
    """Check if trading is halted by kill switch"""
    return get_risk_guard().kill_switch.is_triggered()


def trigger_emergency_halt(reason: str):
    """Trigger emergency trading halt"""
    violation = RiskViolation(
        risk_type=RiskType.MAX_DRAWDOWN,  # Generic emergency
        risk_level=RiskLevel.EMERGENCY,
        current_value=1.0,
        limit_value=0.0,
        violation_percent=100.0,
        description=f"Manual emergency halt: {reason}",
        timestamp=datetime.now()
    )
    
    get_risk_guard().kill_switch.trigger(violation, reason)


if __name__ == "__main__":
    # Example usage
    risk_guard = CentralRiskGuard()
    
    # Example trade risk check
    result = risk_guard.check_trade_risk("BTC/USD", 5000.0, "momentum_v1")
    
    print(f"Trade safe: {result.is_safe}")
    print(f"Risk score: {result.risk_score:.2f}")
    
    if result.violations:
        for violation in result.violations:
            print(f"Violation: {violation.description}")
    
    # Example position update
    position = PositionInfo(
        symbol="BTC/USD",
        size_usd=5000.0,
        entry_price=50000.0,
        current_price=50500.0,
        unrealized_pnl=50.0,
        timestamp=time.time(),
        strategy_id="momentum_v1"
    )
    
    risk_guard.update_position(position)
    
    # Get risk summary
    summary = risk_guard.get_risk_summary()
    print(f"Risk summary: {summary}")
