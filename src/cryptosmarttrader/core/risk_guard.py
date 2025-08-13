"""Enterprise RiskGuard System - Comprehensive risk management and kill-switch functionality."""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading
from contextvars import ContextVar

from .structured_logger import get_logger


class RiskLevel(Enum):
    """Risk escalation levels."""
    NORMAL = "normal"           # < 2% daily loss
    CONSERVATIVE = "conservative"  # 2-3% daily loss
    DEFENSIVE = "defensive"     # 3-5% daily loss
    EMERGENCY = "emergency"     # 5-8% daily loss
    SHUTDOWN = "shutdown"       # > 8% daily loss


class TradingMode(Enum):
    """Trading mode states."""
    LIVE = "live"
    PAPER = "paper"
    DISABLED = "disabled"


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    daily_pnl: float
    daily_pnl_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    total_exposure: float
    position_count: int
    largest_position_percent: float
    correlation_risk: float
    data_quality_score: float
    last_signal_age_minutes: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_daily_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0
    max_position_size_percent: float = 2.0
    max_total_exposure_percent: float = 95.0
    max_correlation_cluster_percent: float = 20.0
    max_position_count: int = 50
    min_data_quality_score: float = 0.7
    max_signal_age_minutes: int = 30


@dataclass
class AlertConfig:
    """Alert configuration."""
    daily_loss_warning: float = 2.0  # %
    daily_loss_critical: float = 3.5  # %
    drawdown_warning: float = 5.0  # %
    drawdown_critical: float = 8.0  # %
    data_gap_warning_minutes: int = 15
    data_gap_critical_minutes: int = 30


class RiskGuard:
    """Enterprise risk management system with kill-switch functionality."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize RiskGuard system."""
        self.logger = get_logger("risk_guard")

        # Load configuration
        self.risk_limits = self._load_risk_limits(config_path)
        self.alert_config = AlertConfig()

        # Current state
        self.current_risk_level = RiskLevel.NORMAL
        self.trading_mode = TradingMode.LIVE
        self.kill_switch_active = False

        # Metrics tracking
        self.risk_metrics_history: List[RiskMetrics] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.position_tracker: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.session_start_time = datetime.now()
        self.session_start_portfolio_value = 0.0
        self.daily_start_portfolio_value = 0.0
        self.max_portfolio_value = 0.0

        # Data quality monitoring
        self.last_data_update = datetime.now()
        self.data_sources_status: Dict[str, datetime] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Alert callbacks
        self.alert_callbacks: List[callable] = []

        # Persistence
        self.data_path = Path("data/risk_guard")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("RiskGuard system initialized",
                        risk_limits=self.risk_limits.__dict__,
                        alert_config=self.alert_config.__dict__)

    def _load_risk_limits(self, config_path: Optional[str]) -> RiskLimits:
        """Load risk limits from configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return RiskLimits(**config.get('risk_limits', {}))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self.logger.warning(f"Failed to load risk config: {e}")

        return RiskLimits()  # Use defaults

    def register_alert_callback(self, callback: callable) -> None:
        """Register callback for risk alerts."""
        self.alert_callbacks.append(callback)
        self.logger.info("Alert callback registered")

    def update_portfolio_value(self, total_value: float) -> None:
        """Update current portfolio value for risk calculations."""
        with self._lock:
            if self.session_start_portfolio_value == 0.0:
                self.session_start_portfolio_value = total_value
                self.daily_start_portfolio_value = total_value

            # Update daily start value at midnight
            now = datetime.now()
            if now.hour == 0 and now.minute < 5:  # Near midnight
                self.daily_start_portfolio_value = total_value

            # Track maximum value for drawdown calculation
            if total_value > self.max_portfolio_value:
                self.max_portfolio_value = total_value

    def update_position(self, symbol: str, size: float, value: float,
                       entry_price: float, current_price: float) -> None:
        """Update position for risk tracking."""
        with self._lock:
            self.position_tracker[symbol] = {
                'size': size,
                'value': value,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': (current_price - entry_price) * size,
                'timestamp': datetime.now()
            }

    def remove_position(self, symbol: str) -> None:
        """Remove closed position from tracking."""
        with self._lock:
            if symbol in self.position_tracker:
                del self.position_tracker[symbol]

    def update_data_source_status(self, source: str) -> None:
        """Update data source last seen timestamp."""
        with self._lock:
            self.data_sources_status[source] = datetime.now()
            self.last_data_update = datetime.now()

    def calculate_current_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Calculate current risk metrics."""
        with self._lock:
            # Daily P&L calculation
            daily_pnl = portfolio_value - self.daily_start_portfolio_value
            daily_pnl_percent = (daily_pnl / self.daily_start_portfolio_value * 100
                               if self.daily_start_portfolio_value > 0 else 0.0)

            # Drawdown calculation
            max_drawdown = self.max_portfolio_value - portfolio_value
            max_drawdown_percent = (max_drawdown / self.max_portfolio_value * 100
                                  if self.max_portfolio_value > 0 else 0.0)

            # Position metrics
            total_exposure = sum(abs(pos['value']) for pos in self.position_tracker.values())
            position_count = len(self.position_tracker)

            largest_position = max([abs(pos['value']) for pos in self.position_tracker.values()],
                                 default=0.0)
            largest_position_percent = (largest_position / portfolio_value * 100
                                     if portfolio_value > 0 else 0.0)

            # Data quality assessment
            data_age_minutes = (datetime.now() - self.last_data_update).total_seconds() / 60
            data_quality_score = max(0.0, 1.0 - (data_age_minutes / 60))  # Degrade over 1 hour

            # Signal freshness
            last_signal_age = int(data_age_minutes)

            # Correlation risk (simplified - would need correlation matrix in practice)
            correlation_risk = min(50.0, position_count * 2.0)  # Approximate

            return RiskMetrics(
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                max_drawdown=max_drawdown,
                max_drawdown_percent=max_drawdown_percent,
                total_exposure=total_exposure,
                position_count=position_count,
                largest_position_percent=largest_position_percent,
                correlation_risk=correlation_risk,
                data_quality_score=data_quality_score,
                last_signal_age_minutes=last_signal_age
            )

    def assess_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """Assess current risk level based on metrics."""
        # Check for emergency conditions
        if (metrics.daily_pnl_percent <= -8.0 or
            metrics.max_drawdown_percent >= 15.0 or
            metrics.data_quality_score < 0.3):
            return RiskLevel.SHUTDOWN

        if (metrics.daily_pnl_percent <= -5.0 or
            metrics.max_drawdown_percent >= 10.0 or
            metrics.last_signal_age_minutes > 60):
            return RiskLevel.EMERGENCY

        if (metrics.daily_pnl_percent <= -3.0 or
            metrics.max_drawdown_percent >= 7.0 or
            metrics.total_exposure > self.risk_limits.max_total_exposure_percent):
            return RiskLevel.DEFENSIVE

        if (metrics.daily_pnl_percent <= -2.0 or
            metrics.max_drawdown_percent >= 5.0 or
            metrics.position_count > self.risk_limits.max_position_count * 0.8):
            return RiskLevel.CONSERVATIVE

        return RiskLevel.NORMAL

    def check_violation(self, metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Check for risk limit violations."""
        violations = []

        # Daily loss limit
        if metrics.daily_pnl_percent <= -self.risk_limits.max_daily_loss_percent:
            violations.append({
                'type': 'daily_loss_limit',
                'severity': 'critical',
                'message': f"Daily loss {metrics.daily_pnl_percent:.2f}% exceeds limit {self.risk_limits.max_daily_loss_percent}%",
                'value': metrics.daily_pnl_percent,
                'limit': -self.risk_limits.max_daily_loss_percent
            })

        # Drawdown limit
        if metrics.max_drawdown_percent >= self.risk_limits.max_drawdown_percent:
            violations.append({
                'type': 'drawdown_limit',
                'severity': 'critical',
                'message': f"Drawdown {metrics.max_drawdown_percent:.2f}% exceeds limit {self.risk_limits.max_drawdown_percent}%",
                'value': metrics.max_drawdown_percent,
                'limit': self.risk_limits.max_drawdown_percent
            })

        # Position size limit
        if metrics.largest_position_percent > self.risk_limits.max_position_size_percent:
            violations.append({
                'type': 'position_size_limit',
                'severity': 'warning',
                'message': f"Largest position {metrics.largest_position_percent:.2f}% exceeds limit {self.risk_limits.max_position_size_percent}%",
                'value': metrics.largest_position_percent,
                'limit': self.risk_limits.max_position_size_percent
            })

        # Data quality limit
        if metrics.data_quality_score < self.risk_limits.min_data_quality_score:
            violations.append({
                'type': 'data_quality',
                'severity': 'critical',
                'message': f"Data quality {metrics.data_quality_score:.2f} below minimum {self.risk_limits.min_data_quality_score}",
                'value': metrics.data_quality_score,
                'limit': self.risk_limits.min_data_quality_score
            })

        return violations

    def trigger_kill_switch(self, reason: str, auto_trigger: bool = True) -> None:
        """Trigger emergency kill switch."""
        with self._lock:
            if self.kill_switch_active:
                return  # Already active

            self.kill_switch_active = True
            self.trading_mode = TradingMode.DISABLED
            self.current_risk_level = RiskLevel.SHUTDOWN

        alert = {
            'type': 'kill_switch_activated',
            'severity': 'critical',
            'reason': reason,
            'auto_trigger': auto_trigger,
            'timestamp': datetime.now(),
            'positions_count': len(self.position_tracker)
        }

        self.alert_history.append(alert)

        self.logger.critical("KILL SWITCH ACTIVATED",
                           reason=reason,
                           auto_trigger=auto_trigger,
                           alert=alert)

        # Notify all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def reset_kill_switch(self, manual_override: bool = False) -> bool:
        """Reset kill switch if conditions are safe."""
        with self._lock:
            if not self.kill_switch_active:
                return True

            # Check if safe to reset
            current_metrics = self.calculate_current_metrics(
                self.daily_start_portfolio_value  # Use safe baseline
            )

            if not manual_override:
                if (current_metrics.data_quality_score < 0.8 or
                    current_metrics.last_signal_age_minutes > 30):
                    self.logger.warning("Kill switch reset denied - conditions not safe")
                    return False

            self.kill_switch_active = False
            self.trading_mode = TradingMode.PAPER  # Start in paper mode
            self.current_risk_level = RiskLevel.CONSERVATIVE

        self.logger.info("Kill switch reset", manual_override=manual_override)
        return True

    def run_risk_check(self, portfolio_value: float) -> Dict[str, Any]:
        """Run comprehensive risk check and return status."""
        metrics = self.calculate_current_metrics(portfolio_value)
        risk_level = self.assess_risk_level(metrics)
        violations = self.check_violation(metrics)

        # Store metrics history
        self.risk_metrics_history.append(metrics)
        if len(self.risk_metrics_history) > 1000:  # Keep last 1000
            self.risk_metrics_history = self.risk_metrics_history[-1000:]

        # Update risk level
        previous_level = self.current_risk_level
        self.current_risk_level = risk_level

        # Check for auto kill switch triggers
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        if critical_violations and not self.kill_switch_active:
            self.trigger_kill_switch(f"Critical violations: {len(critical_violations)}")

        # Risk level escalation
        if risk_level != previous_level:
            self.logger.warning(f"Risk level changed: {previous_level.value} → {risk_level.value}")

            # Auto-adjust trading mode based on risk level
            if risk_level == RiskLevel.SHUTDOWN:
                self.trading_mode = TradingMode.DISABLED
            elif risk_level in [RiskLevel.EMERGENCY, RiskLevel.DEFENSIVE]:
                self.trading_mode = TradingMode.PAPER
            elif risk_level == RiskLevel.CONSERVATIVE and self.trading_mode == TradingMode.DISABLED:
                self.trading_mode = TradingMode.PAPER

        return {
            'risk_level': risk_level.value,
            'trading_mode': self.trading_mode.value,
            'kill_switch_active': self.kill_switch_active,
            'metrics': metrics,
            'violations': violations,
            'timestamp': datetime.now()
        }

    def get_trading_constraints(self) -> Dict[str, Any]:
        """Get current trading constraints based on risk level."""
        base_constraints = {
            'max_position_size_percent': self.risk_limits.max_position_size_percent,
            'max_exposure_percent': self.risk_limits.max_total_exposure_percent,
            'max_positions': self.risk_limits.max_position_count
        }

        # Adjust constraints based on risk level
        risk_multipliers = {
            RiskLevel.NORMAL: 1.0,
            RiskLevel.CONSERVATIVE: 0.75,
            RiskLevel.DEFENSIVE: 0.5,
            RiskLevel.EMERGENCY: 0.25,
            RiskLevel.SHUTDOWN: 0.0
        }

        multiplier = risk_multipliers[self.current_risk_level]

        return {
            'max_position_size_percent': base_constraints['max_position_size_percent'] * multiplier,
            'max_exposure_percent': base_constraints['max_exposure_percent'] * multiplier,
            'max_positions': int(base_constraints['max_positions'] * multiplier),
            'trading_enabled': self.trading_mode != TradingMode.DISABLED,
            'paper_only': self.trading_mode == TradingMode.PAPER,
            'risk_level': self.current_risk_level.value
        }

    def save_state(self) -> None:
        """Save current risk state to disk."""
        state = {
            'current_risk_level': self.current_risk_level.value,
            'trading_mode': self.trading_mode.value,
            'kill_switch_active': self.kill_switch_active,
            'session_start_portfolio_value': self.session_start_portfolio_value,
            'daily_start_portfolio_value': self.daily_start_portfolio_value,
            'max_portfolio_value': self.max_portfolio_value,
            'alert_history': self.alert_history[-100:],  # Keep last 100
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(self.data_path / "risk_state.json", 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save risk state: {e}")

    def load_state(self) -> None:
        """Load previous risk state from disk."""
        try:
            state_file = self.data_path / "risk_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self.current_risk_level = RiskLevel(state['current_risk_level'])
                self.trading_mode = TradingMode(state['trading_mode'])
                self.kill_switch_active = state['kill_switch_active']
                self.session_start_portfolio_value = state['session_start_portfolio_value']
                self.daily_start_portfolio_value = state['daily_start_portfolio_value']
                self.max_portfolio_value = state['max_portfolio_value']
                self.alert_history = state.get('alert_history', [])

                self.logger.info("Risk state loaded from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load risk state: {e}")


def create_risk_guard(config_path: Optional[str] = None) -> RiskGuard:
    """Factory function to create RiskGuard instance."""
    return RiskGuard(config_path=config_path)
