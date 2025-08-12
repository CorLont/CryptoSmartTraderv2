"""
Risk Limits System

Comprehensive risk management with daily loss limits, max-drawdown guards,
exposure limits per asset/cluster, and real-time enforcement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskLimitType(Enum):
    """Types of risk limits"""
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_SIZE = "position_size"
    ASSET_EXPOSURE = "asset_exposure"
    CLUSTER_EXPOSURE = "cluster_exposure"
    TOTAL_EXPOSURE = "total_exposure"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"

class RiskStatus(Enum):
    """Risk status levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"

class ActionType(Enum):
    """Risk management actions"""
    MONITOR = "monitor"
    REDUCE_SIZE = "reduce_size"
    HALT_NEW_POSITIONS = "halt_new_positions"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskLimit:
    """Individual risk limit configuration"""
    name: str
    limit_type: RiskLimitType
    threshold_value: float
    warning_threshold: float
    critical_threshold: float
    
    # Scope
    symbol: Optional[str] = None
    cluster: Optional[str] = None
    
    # Actions
    warning_action: ActionType = ActionType.MONITOR
    critical_action: ActionType = ActionType.REDUCE_SIZE
    breach_action: ActionType = ActionType.HALT_NEW_POSITIONS
    
    # Timing
    time_window_hours: float = 24.0
    cooldown_minutes: int = 15
    
    # Status tracking
    current_value: float = 0.0
    last_breach_time: Optional[datetime] = None
    breach_count: int = 0

@dataclass
class RiskViolation:
    """Risk limit violation record"""
    timestamp: datetime
    limit_name: str
    limit_type: RiskLimitType
    current_value: float
    threshold_value: float
    severity: RiskStatus
    action_taken: ActionType
    
    # Context
    symbol: Optional[str] = None
    cluster: Optional[str] = None
    portfolio_value: float = 0.0
    
    # Resolution
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_action: Optional[str] = None

class RiskLimitManager:
    """
    Comprehensive risk limit management and enforcement
    """
    
    def __init__(self, initial_portfolio_value: float = 100000.0):
        self.initial_portfolio_value = initial_portfolio_value
        self.current_portfolio_value = initial_portfolio_value
        
        # Risk limits registry
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.violations: List[RiskViolation] = []
        
        # Portfolio tracking
        self.daily_pnl_history: List[Tuple[date, float]] = []
        self.high_water_mark = initial_portfolio_value
        self.current_drawdown = 0.0
        
        # Position tracking
        self.current_positions: Dict[str, float] = {}  # symbol -> position value
        self.cluster_allocations: Dict[str, List[str]] = {}  # cluster -> symbols
        
        # Emergency state
        self.emergency_stop_active = False
        self.kill_switch_triggered = False
        self.last_check_time = datetime.now()
        
        # Setup default limits
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default risk limits"""
        
        # Daily loss limit (5% of portfolio)
        self.add_risk_limit(RiskLimit(
            name="daily_loss_limit",
            limit_type=RiskLimitType.DAILY_LOSS,
            threshold_value=0.05,  # 5%
            warning_threshold=0.03,  # 3%
            critical_threshold=0.045,  # 4.5%
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.HALT_NEW_POSITIONS,
            time_window_hours=24.0
        ))
        
        # Maximum drawdown limit (10% of high water mark)
        self.add_risk_limit(RiskLimit(
            name="max_drawdown_limit",
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold_value=0.10,  # 10%
            warning_threshold=0.07,  # 7%
            critical_threshold=0.09,  # 9%
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.CLOSE_POSITIONS,
            time_window_hours=24.0 * 30  # 30 days
        ))
        
        # Single position size limit (5% of portfolio)
        self.add_risk_limit(RiskLimit(
            name="single_position_limit",
            limit_type=RiskLimitType.POSITION_SIZE,
            threshold_value=0.05,  # 5%
            warning_threshold=0.04,  # 4%
            critical_threshold=0.045,  # 4.5%
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.HALT_NEW_POSITIONS
        ))
        
        # Total crypto exposure limit (25% of portfolio)
        self.add_risk_limit(RiskLimit(
            name="total_crypto_exposure",
            limit_type=RiskLimitType.TOTAL_EXPOSURE,
            threshold_value=0.25,  # 25%
            warning_threshold=0.20,  # 20%
            critical_threshold=0.23,  # 23%
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.HALT_NEW_POSITIONS
        ))
        
        # Large cap cluster limit (15% of portfolio)
        self.add_risk_limit(RiskLimit(
            name="large_cap_cluster_limit",
            limit_type=RiskLimitType.CLUSTER_EXPOSURE,
            threshold_value=0.15,  # 15%
            warning_threshold=0.12,  # 12%
            critical_threshold=0.14,  # 14%
            cluster="large_cap",
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.HALT_NEW_POSITIONS
        ))
        
        # Meme coin cluster limit (3% of portfolio)
        self.add_risk_limit(RiskLimit(
            name="meme_cluster_limit",
            limit_type=RiskLimitType.CLUSTER_EXPOSURE,
            threshold_value=0.03,  # 3%
            warning_threshold=0.02,  # 2%
            critical_threshold=0.025,  # 2.5%
            cluster="meme",
            warning_action=ActionType.MONITOR,
            critical_action=ActionType.REDUCE_SIZE,
            breach_action=ActionType.CLOSE_POSITIONS
        ))
    
    def add_risk_limit(self, risk_limit: RiskLimit):
        """Add a risk limit to the system"""
        self.risk_limits[risk_limit.name] = risk_limit
        logger.info(f"Added risk limit: {risk_limit.name} ({risk_limit.limit_type.value})")
    
    def update_portfolio_value(self, new_value: float):
        """Update current portfolio value and calculate PnL"""
        previous_value = self.current_portfolio_value
        self.current_portfolio_value = new_value
        
        # Update high water mark
        if new_value > self.high_water_mark:
            self.high_water_mark = new_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.high_water_mark - new_value) / self.high_water_mark
        
        # Record daily PnL
        today = datetime.now().date()
        daily_pnl = new_value - previous_value
        
        # Update or add today's PnL
        if self.daily_pnl_history and self.daily_pnl_history[-1][0] == today:
            # Update today's PnL
            total_daily_pnl = self.daily_pnl_history[-1][1] + daily_pnl
            self.daily_pnl_history[-1] = (today, total_daily_pnl)
        else:
            # New day
            self.daily_pnl_history.append((today, daily_pnl))
        
        # Keep only last 30 days
        cutoff_date = today - timedelta(days=30)
        self.daily_pnl_history = [
            (d, pnl) for d, pnl in self.daily_pnl_history if d >= cutoff_date
        ]
        
        logger.debug(f"Portfolio value updated: ${new_value:,.2f} (PnL: ${daily_pnl:+,.2f})")
    
    def update_position(self, symbol: str, position_value: float):
        """Update position value for a symbol"""
        self.current_positions[symbol] = position_value
        logger.debug(f"Position updated: {symbol} = ${position_value:,.2f}")
    
    def set_cluster_allocation(self, cluster: str, symbols: List[str]):
        """Set which symbols belong to a cluster"""
        self.cluster_allocations[cluster] = symbols
        logger.debug(f"Cluster allocation set: {cluster} = {symbols}")
    
    def check_daily_loss_limit(self) -> Tuple[RiskStatus, float]:
        """Check daily loss limit"""
        if not self.daily_pnl_history:
            return RiskStatus.NORMAL, 0.0
        
        # Get today's PnL
        today = datetime.now().date()
        today_pnl = 0.0
        
        for d, pnl in self.daily_pnl_history:
            if d == today:
                today_pnl = pnl
                break
        
        # Calculate loss percentage
        if today_pnl >= 0:
            return RiskStatus.NORMAL, 0.0
        
        loss_pct = abs(today_pnl) / self.current_portfolio_value
        
        # Update limit current value
        daily_loss_limit = self.risk_limits.get("daily_loss_limit")
        if daily_loss_limit:
            daily_loss_limit.current_value = loss_pct
        
        # Check thresholds
        if daily_loss_limit:
            if loss_pct >= daily_loss_limit.threshold_value:
                return RiskStatus.BREACH, loss_pct
            elif loss_pct >= daily_loss_limit.critical_threshold:
                return RiskStatus.CRITICAL, loss_pct
            elif loss_pct >= daily_loss_limit.warning_threshold:
                return RiskStatus.WARNING, loss_pct
        
        return RiskStatus.NORMAL, loss_pct
    
    def check_drawdown_limit(self) -> Tuple[RiskStatus, float]:
        """Check maximum drawdown limit"""
        # Update limit current value
        drawdown_limit = self.risk_limits.get("max_drawdown_limit")
        if drawdown_limit:
            drawdown_limit.current_value = self.current_drawdown
        
        # Check thresholds
        if drawdown_limit:
            if self.current_drawdown >= drawdown_limit.threshold_value:
                return RiskStatus.BREACH, self.current_drawdown
            elif self.current_drawdown >= drawdown_limit.critical_threshold:
                return RiskStatus.CRITICAL, self.current_drawdown
            elif self.current_drawdown >= drawdown_limit.warning_threshold:
                return RiskStatus.WARNING, self.current_drawdown
        
        return RiskStatus.NORMAL, self.current_drawdown
    
    def check_position_size_limits(self) -> List[Tuple[str, RiskStatus, float]]:
        """Check individual position size limits"""
        violations = []
        
        position_limit = self.risk_limits.get("single_position_limit")
        if not position_limit or self.current_portfolio_value <= 0:
            return violations
        
        for symbol, position_value in self.current_positions.items():
            if position_value <= 0:
                continue
            
            position_pct = position_value / self.current_portfolio_value
            
            # Check thresholds
            if position_pct >= position_limit.threshold_value:
                status = RiskStatus.BREACH
            elif position_pct >= position_limit.critical_threshold:
                status = RiskStatus.CRITICAL
            elif position_pct >= position_limit.warning_threshold:
                status = RiskStatus.WARNING
            else:
                status = RiskStatus.NORMAL
            
            if status != RiskStatus.NORMAL:
                violations.append((symbol, status, position_pct))
        
        return violations
    
    def check_cluster_exposure_limits(self) -> List[Tuple[str, RiskStatus, float]]:
        """Check cluster exposure limits"""
        violations = []
        
        if self.current_portfolio_value <= 0:
            return violations
        
        # Calculate cluster exposures
        cluster_exposures = {}
        for cluster, symbols in self.cluster_allocations.items():
            total_exposure = sum(
                self.current_positions.get(symbol, 0) for symbol in symbols
            )
            cluster_exposures[cluster] = total_exposure / self.current_portfolio_value
        
        # Check each cluster limit
        for limit_name, risk_limit in self.risk_limits.items():
            if risk_limit.limit_type != RiskLimitType.CLUSTER_EXPOSURE:
                continue
            
            cluster = risk_limit.cluster
            if not cluster or cluster not in cluster_exposures:
                continue
            
            exposure_pct = cluster_exposures[cluster]
            risk_limit.current_value = exposure_pct
            
            # Check thresholds
            if exposure_pct >= risk_limit.threshold_value:
                status = RiskStatus.BREACH
            elif exposure_pct >= risk_limit.critical_threshold:
                status = RiskStatus.CRITICAL
            elif exposure_pct >= risk_limit.warning_threshold:
                status = RiskStatus.WARNING
            else:
                status = RiskStatus.NORMAL
            
            if status != RiskStatus.NORMAL:
                violations.append((cluster, status, exposure_pct))
        
        return violations
    
    def check_total_exposure_limit(self) -> Tuple[RiskStatus, float]:
        """Check total crypto exposure limit"""
        if self.current_portfolio_value <= 0:
            return RiskStatus.NORMAL, 0.0
        
        total_crypto_value = sum(self.current_positions.values())
        total_exposure_pct = total_crypto_value / self.current_portfolio_value
        
        # Update limit current value
        exposure_limit = self.risk_limits.get("total_crypto_exposure")
        if exposure_limit:
            exposure_limit.current_value = total_exposure_pct
        
        # Check thresholds
        if exposure_limit:
            if total_exposure_pct >= exposure_limit.threshold_value:
                return RiskStatus.BREACH, total_exposure_pct
            elif total_exposure_pct >= exposure_limit.critical_threshold:
                return RiskStatus.CRITICAL, total_exposure_pct
            elif total_exposure_pct >= exposure_limit.warning_threshold:
                return RiskStatus.WARNING, total_exposure_pct
        
        return RiskStatus.NORMAL, total_exposure_pct
    
    def check_all_limits(self) -> List[RiskViolation]:
        """Check all risk limits and return violations"""
        violations = []
        current_time = datetime.now()
        
        # Daily loss limit
        daily_loss_status, daily_loss_value = self.check_daily_loss_limit()
        if daily_loss_status != RiskStatus.NORMAL:
            limit = self.risk_limits["daily_loss_limit"]
            violation = RiskViolation(
                timestamp=current_time,
                limit_name="daily_loss_limit",
                limit_type=RiskLimitType.DAILY_LOSS,
                current_value=daily_loss_value,
                threshold_value=limit.threshold_value,
                severity=daily_loss_status,
                action_taken=self._get_action_for_status(limit, daily_loss_status),
                portfolio_value=self.current_portfolio_value
            )
            violations.append(violation)
        
        # Drawdown limit
        drawdown_status, drawdown_value = self.check_drawdown_limit()
        if drawdown_status != RiskStatus.NORMAL:
            limit = self.risk_limits["max_drawdown_limit"]
            violation = RiskViolation(
                timestamp=current_time,
                limit_name="max_drawdown_limit",
                limit_type=RiskLimitType.MAX_DRAWDOWN,
                current_value=drawdown_value,
                threshold_value=limit.threshold_value,
                severity=drawdown_status,
                action_taken=self._get_action_for_status(limit, drawdown_status),
                portfolio_value=self.current_portfolio_value
            )
            violations.append(violation)
        
        # Position size limits
        position_violations = self.check_position_size_limits()
        for symbol, status, value in position_violations:
            limit = self.risk_limits["single_position_limit"]
            violation = RiskViolation(
                timestamp=current_time,
                limit_name="single_position_limit",
                limit_type=RiskLimitType.POSITION_SIZE,
                current_value=value,
                threshold_value=limit.threshold_value,
                severity=status,
                action_taken=self._get_action_for_status(limit, status),
                symbol=symbol,
                portfolio_value=self.current_portfolio_value
            )
            violations.append(violation)
        
        # Cluster exposure limits
        cluster_violations = self.check_cluster_exposure_limits()
        for cluster, status, value in cluster_violations:
            # Find the specific cluster limit
            cluster_limit = None
            for limit_name, limit in self.risk_limits.items():
                if (limit.limit_type == RiskLimitType.CLUSTER_EXPOSURE and 
                    limit.cluster == cluster):
                    cluster_limit = limit
                    break
            
            if cluster_limit:
                violation = RiskViolation(
                    timestamp=current_time,
                    limit_name=cluster_limit.name,
                    limit_type=RiskLimitType.CLUSTER_EXPOSURE,
                    current_value=value,
                    threshold_value=cluster_limit.threshold_value,
                    severity=status,
                    action_taken=self._get_action_for_status(cluster_limit, status),
                    cluster=cluster,
                    portfolio_value=self.current_portfolio_value
                )
                violations.append(violation)
        
        # Total exposure limit
        exposure_status, exposure_value = self.check_total_exposure_limit()
        if exposure_status != RiskStatus.NORMAL:
            limit = self.risk_limits["total_crypto_exposure"]
            violation = RiskViolation(
                timestamp=current_time,
                limit_name="total_crypto_exposure",
                limit_type=RiskLimitType.TOTAL_EXPOSURE,
                current_value=exposure_value,
                threshold_value=limit.threshold_value,
                severity=exposure_status,
                action_taken=self._get_action_for_status(limit, exposure_status),
                portfolio_value=self.current_portfolio_value
            )
            violations.append(violation)
        
        # Store violations
        self.violations.extend(violations)
        
        # Update last check time
        self.last_check_time = current_time
        
        return violations
    
    def _get_action_for_status(self, limit: RiskLimit, status: RiskStatus) -> ActionType:
        """Get appropriate action for risk status"""
        if status == RiskStatus.WARNING:
            return limit.warning_action
        elif status == RiskStatus.CRITICAL:
            return limit.critical_action
        elif status == RiskStatus.BREACH:
            return limit.breach_action
        else:
            return ActionType.MONITOR
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        self.kill_switch_triggered = True
        
        emergency_violation = RiskViolation(
            timestamp=datetime.now(),
            limit_name="emergency_stop",
            limit_type=RiskLimitType.DAILY_LOSS,  # Placeholder
            current_value=0.0,
            threshold_value=0.0,
            severity=RiskStatus.BREACH,
            action_taken=ActionType.EMERGENCY_STOP,
            portfolio_value=self.current_portfolio_value
        )
        
        self.violations.append(emergency_violation)
        
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def clear_emergency_stop(self, authorization_code: str = "CLEAR_EMERGENCY"):
        """Clear emergency stop (requires authorization)"""
        if authorization_code == "CLEAR_EMERGENCY":
            self.emergency_stop_active = False
            self.kill_switch_triggered = False
            logger.warning("Emergency stop cleared")
            return True
        else:
            logger.error("Invalid authorization code for emergency stop clear")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        # Check all limits
        current_violations = self.check_all_limits()
        
        # Calculate risk metrics
        daily_loss_status, daily_loss_value = self.check_daily_loss_limit()
        drawdown_status, drawdown_value = self.check_drawdown_limit()
        exposure_status, exposure_value = self.check_total_exposure_limit()
        
        # Position size violations
        position_violations = self.check_position_size_limits()
        cluster_violations = self.check_cluster_exposure_limits()
        
        # Overall risk level
        all_statuses = [daily_loss_status, drawdown_status, exposure_status]
        all_statuses.extend([status for _, status, _ in position_violations])
        all_statuses.extend([status for _, status, _ in cluster_violations])
        
        if RiskStatus.BREACH in all_statuses:
            overall_risk = RiskStatus.BREACH
        elif RiskStatus.CRITICAL in all_statuses:
            overall_risk = RiskStatus.CRITICAL
        elif RiskStatus.WARNING in all_statuses:
            overall_risk = RiskStatus.WARNING
        else:
            overall_risk = RiskStatus.NORMAL
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_risk_status": overall_risk.value,
            "emergency_stop_active": self.emergency_stop_active,
            "kill_switch_triggered": self.kill_switch_triggered,
            
            # Portfolio metrics
            "portfolio_value": self.current_portfolio_value,
            "high_water_mark": self.high_water_mark,
            "current_drawdown_pct": self.current_drawdown * 100,
            "daily_pnl": self.daily_pnl_history[-1][1] if self.daily_pnl_history else 0.0,
            
            # Risk limit status
            "daily_loss": {
                "status": daily_loss_status.value,
                "current_pct": daily_loss_value * 100,
                "limit_pct": self.risk_limits["daily_loss_limit"].threshold_value * 100
            },
            "max_drawdown": {
                "status": drawdown_status.value,
                "current_pct": drawdown_value * 100,
                "limit_pct": self.risk_limits["max_drawdown_limit"].threshold_value * 100
            },
            "total_exposure": {
                "status": exposure_status.value,
                "current_pct": exposure_value * 100,
                "limit_pct": self.risk_limits["total_crypto_exposure"].threshold_value * 100
            },
            
            # Violations
            "active_violations": len(current_violations),
            "position_violations": len(position_violations),
            "cluster_violations": len(cluster_violations),
            
            # Historical
            "total_violations_30d": len([
                v for v in self.violations 
                if v.timestamp >= datetime.now() - timedelta(days=30)
            ])
        }
    
    def export_risk_report(self, filepath: str, days_back: int = 30):
        """Export detailed risk report"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_violations = [
                v for v in self.violations if v.timestamp >= cutoff_date
            ]
            
            report_data = {
                "report_timestamp": datetime.now().isoformat(),
                "period_days": days_back,
                "risk_summary": self.get_risk_summary(),
                
                # Risk limits configuration
                "risk_limits": {
                    name: {
                        "limit_type": limit.limit_type.value,
                        "threshold_value": limit.threshold_value,
                        "warning_threshold": limit.warning_threshold,
                        "critical_threshold": limit.critical_threshold,
                        "current_value": limit.current_value,
                        "symbol": limit.symbol,
                        "cluster": limit.cluster
                    }
                    for name, limit in self.risk_limits.items()
                },
                
                # Violation history
                "violation_history": [
                    {
                        "timestamp": v.timestamp.isoformat(),
                        "limit_name": v.limit_name,
                        "limit_type": v.limit_type.value,
                        "current_value": v.current_value,
                        "threshold_value": v.threshold_value,
                        "severity": v.severity.value,
                        "action_taken": v.action_taken.value,
                        "symbol": v.symbol,
                        "cluster": v.cluster,
                        "resolved": v.resolved
                    }
                    for v in recent_violations
                ],
                
                # Daily PnL history
                "daily_pnl_history": [
                    {"date": d.isoformat(), "pnl": pnl}
                    for d, pnl in self.daily_pnl_history
                ],
                
                # Current positions
                "current_positions": {
                    symbol: {
                        "value": value,
                        "percentage": (value / self.current_portfolio_value * 100) 
                                    if self.current_portfolio_value > 0 else 0
                    }
                    for symbol, value in self.current_positions.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Risk report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export risk report: {e}")
    
    def is_trading_allowed(self, symbol: Optional[str] = None, 
                          size: float = 0.0) -> Tuple[bool, str]:
        """Check if trading is allowed given current risk status"""
        
        if self.emergency_stop_active or self.kill_switch_triggered:
            return False, "Emergency stop active"
        
        # Check recent violations for halt conditions
        recent_violations = [
            v for v in self.violations 
            if v.timestamp >= datetime.now() - timedelta(minutes=15)
            and not v.resolved
        ]
        
        for violation in recent_violations:
            if violation.action_taken in [ActionType.HALT_NEW_POSITIONS, ActionType.EMERGENCY_STOP]:
                return False, f"Trading halted due to {violation.limit_name}"
            
            if (violation.action_taken == ActionType.CLOSE_POSITIONS and 
                violation.symbol == symbol):
                return False, f"Position closure required for {symbol}"
        
        # Check if new position would violate limits
        if symbol and size > 0:
            # Simulate new position
            current_value = self.current_positions.get(symbol, 0)
            new_position_value = current_value + size
            
            # Check position size limit
            if self.current_portfolio_value > 0:
                new_position_pct = new_position_value / self.current_portfolio_value
                position_limit = self.risk_limits.get("single_position_limit")
                
                if (position_limit and 
                    new_position_pct >= position_limit.threshold_value):
                    return False, f"Position size limit would be breached for {symbol}"
        
        return True, "Trading allowed"