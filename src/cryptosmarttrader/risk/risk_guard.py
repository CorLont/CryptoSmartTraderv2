"""
Enterprise Risk Management System - Hard Blockers and Kill Switch
Comprehensive risk controls with real-time monitoring and automatic trading halt.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk escalation levels with automatic actions."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class TradingMode(Enum):
    """Trading mode based on risk level."""
    ACTIVE = "active"
    CONSERVATIVE = "conservative"
    DEFENSIVE = "defensive"
    EMERGENCY_ONLY = "emergency_only"
    SHUTDOWN = "shutdown"


@dataclass
class RiskEvent:
    """Risk event data structure."""
    timestamp: datetime
    event_type: str
    severity: RiskLevel
    description: str
    trigger_value: float
    threshold: float
    action_taken: str
    position_data: Optional[Dict] = None


class RiskLimits(BaseModel):
    """Configurable risk limits with hard thresholds."""
    
    # Daily loss limits
    daily_loss_warning: float = Field(default=0.03, ge=0, le=1, description="3% daily loss warning")
    daily_loss_critical: float = Field(default=0.05, ge=0, le=1, description="5% daily loss critical")
    daily_loss_emergency: float = Field(default=0.08, ge=0, le=1, description="8% daily loss emergency stop")
    
    # Maximum drawdown limits
    max_drawdown_warning: float = Field(default=0.05, ge=0, le=1, description="5% max drawdown warning")
    max_drawdown_critical: float = Field(default=0.10, ge=0, le=1, description="10% max drawdown critical")
    max_drawdown_emergency: float = Field(default=0.15, ge=0, le=1, description="15% max drawdown kill switch")
    
    # Position exposure limits
    max_position_size: float = Field(default=0.02, ge=0, le=1, description="2% max position size")
    max_asset_exposure: float = Field(default=0.05, ge=0, le=1, description="5% max single asset exposure")
    max_cluster_exposure: float = Field(default=0.20, ge=0, le=1, description="20% max cluster exposure")
    max_total_positions: int = Field(default=50, ge=1, le=200, description="Maximum total positions")
    
    # Data quality limits
    max_data_gap_seconds: int = Field(default=300, ge=60, description="5 minute max data gap")
    max_latency_ms: int = Field(default=5000, ge=100, description="5 second max latency")
    min_api_success_rate: float = Field(default=0.90, ge=0.5, le=1.0, description="90% min API success rate")
    
    # Volatility limits
    max_portfolio_volatility: float = Field(default=0.25, ge=0, le=2.0, description="25% max portfolio vol")
    max_correlation_exposure: float = Field(default=0.30, ge=0, le=1.0, description="30% max correlation exposure")


class RiskGuard:
    """
    Enterprise Risk Management System with hard blockers and kill switch.
    
    Features:
    - Real-time risk monitoring
    - Automatic trading halt on breaches
    - Progressive risk escalation
    - Kill switch activation
    - Comprehensive logging and alerts
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/risk_limits.json")
        self.limits = self._load_limits()
        
        # Risk state tracking
        self.current_risk_level = RiskLevel.NORMAL
        self.trading_mode = TradingMode.ACTIVE
        self.kill_switch_active = False
        self.last_update = datetime.now()
        
        # Performance tracking
        self.daily_start_equity = 100000.0  # Will be updated from portfolio
        self.current_equity = 100000.0
        self.peak_equity = 100000.0
        self.daily_pnl = 0.0
        self.total_drawdown = 0.0
        
        # Position tracking
        self.positions: Dict[str, float] = {}
        self.asset_exposures: Dict[str, float] = {}
        self.cluster_exposures: Dict[str, float] = {}
        
        # Data quality tracking
        self.last_data_timestamp: Optional[datetime] = None
        self.api_success_count = 0
        self.api_total_count = 0
        self.recent_latencies: List[float] = []
        
        # Risk events log
        self.risk_events: List[RiskEvent] = []
        self.alerts_sent: List[str] = []
        
        logger.info("RiskGuard initialized with comprehensive hard blockers")
    
    def _load_limits(self) -> RiskLimits:
        """Load risk limits from configuration file."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return RiskLimits(**config_data)
            except Exception as e:
                logger.warning(f"Failed to load risk config: {e}, using defaults")
        
        return RiskLimits()
    
    def save_limits(self):
        """Save current risk limits to configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.limits.dict(), f, indent=2)
        logger.info(f"Risk limits saved to {self.config_path}")
    
    def update_portfolio_state(self, equity: float, positions: Dict[str, float], 
                             asset_exposures: Dict[str, float] = None,
                             cluster_exposures: Dict[str, float] = None):
        """Update portfolio state for risk monitoring."""
        self.current_equity = equity
        self.positions = positions.copy()
        
        if asset_exposures is not None:
            self.asset_exposures = asset_exposures.copy()
        if cluster_exposures is not None:
            self.cluster_exposures = cluster_exposures.copy()
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.total_drawdown = (self.peak_equity - equity) / self.peak_equity
        self.daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity
        
        # Check all risk conditions
        self._check_all_risks()
        self.last_update = datetime.now()
    
    def update_data_quality(self, timestamp: datetime, api_success: bool, latency_ms: float):
        """Update data quality metrics."""
        self.last_data_timestamp = timestamp
        
        # Track API success rate
        self.api_total_count += 1
        if api_success:
            self.api_success_count += 1
        
        # Track latency (keep last 100 measurements)
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > 100:
            self.recent_latencies.pop(0)
        
        # Check data quality risks
        self._check_data_quality_risks()
    
    def _check_all_risks(self):
        """Comprehensive risk checking with progressive escalation."""
        previous_level = self.current_risk_level
        
        # Check all risk categories
        daily_risk = self._check_daily_loss_risk()
        drawdown_risk = self._check_drawdown_risk()
        position_risk = self._check_position_risk()
        exposure_risk = self._check_exposure_risk()
        
        # Determine highest risk level
        risk_levels = [daily_risk, drawdown_risk, position_risk, exposure_risk]
        max_risk = max(risk_levels, key=lambda x: list(RiskLevel).index(x))
        
        # Update risk level and trading mode
        self._update_risk_level(max_risk)
        
        # Log risk level changes
        if self.current_risk_level != previous_level:
            self._log_risk_event(
                "risk_level_change",
                self.current_risk_level,
                f"Risk level changed from {previous_level.value} to {self.current_risk_level.value}",
                0.0, 0.0,
                f"Trading mode: {self.trading_mode.value}"
            )
    
    def _check_daily_loss_risk(self) -> RiskLevel:
        """Check daily loss limits."""
        if abs(self.daily_pnl) >= self.limits.daily_loss_emergency:
            self._log_risk_event(
                "daily_loss_emergency",
                RiskLevel.EMERGENCY,
                f"Daily loss exceeded emergency threshold: {self.daily_pnl:.2%}",
                abs(self.daily_pnl), self.limits.daily_loss_emergency,
                "KILL SWITCH ACTIVATED"
            )
            return RiskLevel.EMERGENCY
        
        elif abs(self.daily_pnl) >= self.limits.daily_loss_critical:
            self._log_risk_event(
                "daily_loss_critical",
                RiskLevel.CRITICAL,
                f"Daily loss exceeded critical threshold: {self.daily_pnl:.2%}",
                abs(self.daily_pnl), self.limits.daily_loss_critical,
                "Emergency trading mode activated"
            )
            return RiskLevel.CRITICAL
        
        elif abs(self.daily_pnl) >= self.limits.daily_loss_warning:
            return RiskLevel.WARNING
        
        return RiskLevel.NORMAL
    
    def _check_drawdown_risk(self) -> RiskLevel:
        """Check maximum drawdown limits."""
        if self.total_drawdown >= self.limits.max_drawdown_emergency:
            self._log_risk_event(
                "drawdown_emergency",
                RiskLevel.EMERGENCY,
                f"Drawdown exceeded emergency threshold: {self.total_drawdown:.2%}",
                self.total_drawdown, self.limits.max_drawdown_emergency,
                "KILL SWITCH ACTIVATED"
            )
            return RiskLevel.EMERGENCY
        
        elif self.total_drawdown >= self.limits.max_drawdown_critical:
            self._log_risk_event(
                "drawdown_critical",
                RiskLevel.CRITICAL,
                f"Drawdown exceeded critical threshold: {self.total_drawdown:.2%}",
                self.total_drawdown, self.limits.max_drawdown_critical,
                "Emergency trading mode activated"
            )
            return RiskLevel.CRITICAL
        
        elif self.total_drawdown >= self.limits.max_drawdown_warning:
            return RiskLevel.WARNING
        
        return RiskLevel.NORMAL
    
    def _check_position_risk(self) -> RiskLevel:
        """Check position size and count limits."""
        # Check individual position sizes
        max_position = max(abs(pos) for pos in self.positions.values()) if self.positions else 0
        
        if max_position > self.limits.max_position_size:
            self._log_risk_event(
                "position_size_exceeded",
                RiskLevel.CRITICAL,
                f"Position size exceeded limit: {max_position:.2%}",
                max_position, self.limits.max_position_size,
                "Position size reduction required"
            )
            return RiskLevel.CRITICAL
        
        # Check total position count
        if len(self.positions) > self.limits.max_total_positions:
            self._log_risk_event(
                "position_count_exceeded",
                RiskLevel.WARNING,
                f"Position count exceeded: {len(self.positions)}",
                len(self.positions), self.limits.max_total_positions,
                "Position consolidation recommended"
            )
            return RiskLevel.WARNING
        
        return RiskLevel.NORMAL
    
    def _check_exposure_risk(self) -> RiskLevel:
        """Check asset and cluster exposure limits."""
        # Check asset exposure
        max_asset_exposure = max(abs(exp) for exp in self.asset_exposures.values()) if self.asset_exposures else 0
        
        if max_asset_exposure > self.limits.max_asset_exposure:
            self._log_risk_event(
                "asset_exposure_exceeded",
                RiskLevel.CRITICAL,
                f"Asset exposure exceeded: {max_asset_exposure:.2%}",
                max_asset_exposure, self.limits.max_asset_exposure,
                "Asset exposure reduction required"
            )
            return RiskLevel.CRITICAL
        
        # Check cluster exposure
        max_cluster_exposure = max(abs(exp) for exp in self.cluster_exposures.values()) if self.cluster_exposures else 0
        
        if max_cluster_exposure > self.limits.max_cluster_exposure:
            self._log_risk_event(
                "cluster_exposure_exceeded",
                RiskLevel.WARNING,
                f"Cluster exposure exceeded: {max_cluster_exposure:.2%}",
                max_cluster_exposure, self.limits.max_cluster_exposure,
                "Cluster diversification required"
            )
            return RiskLevel.WARNING
        
        return RiskLevel.NORMAL
    
    def _check_data_quality_risks(self):
        """Check data gaps, latency, and API reliability."""
        now = datetime.now()
        
        # Check data gap
        if self.last_data_timestamp:
            data_gap = (now - self.last_data_timestamp).total_seconds()
            if data_gap > self.limits.max_data_gap_seconds:
                self._activate_kill_switch(
                    f"Data gap detected: {data_gap:.0f} seconds",
                    "data_gap"
                )
                return
        
        # Check API success rate
        if self.api_total_count > 10:  # Need minimum samples
            success_rate = self.api_success_count / self.api_total_count
            if success_rate < self.limits.min_api_success_rate:
                self._activate_kill_switch(
                    f"API success rate too low: {success_rate:.2%}",
                    "api_reliability"
                )
                return
        
        # Check latency
        if len(self.recent_latencies) > 5:  # Need minimum samples
            avg_latency = sum(self.recent_latencies[-10:]) / min(10, len(self.recent_latencies))
            if avg_latency > self.limits.max_latency_ms:
                self._activate_kill_switch(
                    f"High latency detected: {avg_latency:.0f}ms",
                    "high_latency"
                )
                return
    
    def _update_risk_level(self, new_level: RiskLevel):
        """Update risk level and corresponding trading mode."""
        self.current_risk_level = new_level
        
        # Map risk level to trading mode
        if new_level == RiskLevel.NORMAL:
            self.trading_mode = TradingMode.ACTIVE
        elif new_level == RiskLevel.WARNING:
            self.trading_mode = TradingMode.CONSERVATIVE
        elif new_level == RiskLevel.CRITICAL:
            self.trading_mode = TradingMode.DEFENSIVE
        elif new_level == RiskLevel.EMERGENCY:
            self.trading_mode = TradingMode.EMERGENCY_ONLY
            self._activate_kill_switch("Emergency risk level reached", "emergency_risk")
        elif new_level == RiskLevel.SHUTDOWN:
            self.trading_mode = TradingMode.SHUTDOWN
            self._activate_kill_switch("Shutdown risk level reached", "shutdown_risk")
    
    def _activate_kill_switch(self, reason: str, trigger_type: str):
        """Activate kill switch and halt all trading."""
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.trading_mode = TradingMode.SHUTDOWN
            
            self._log_risk_event(
                f"kill_switch_{trigger_type}",
                RiskLevel.SHUTDOWN,
                f"KILL SWITCH ACTIVATED: {reason}",
                1.0, 0.0,
                "ALL TRADING HALTED"
            )
            
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            self._send_critical_alert(reason)
    
    def _log_risk_event(self, event_type: str, severity: RiskLevel, description: str,
                       trigger_value: float, threshold: float, action: str):
        """Log risk event with full details."""
        event = RiskEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            description=description,
            trigger_value=trigger_value,
            threshold=threshold,
            action_taken=action,
            position_data={
                'equity': self.current_equity,
                'daily_pnl': self.daily_pnl,
                'drawdown': self.total_drawdown,
                'positions': len(self.positions)
            }
        )
        
        self.risk_events.append(event)
        
        # Log based on severity
        if severity in [RiskLevel.EMERGENCY, RiskLevel.SHUTDOWN]:
            logger.critical(f"🚨 {description} - {action}")
        elif severity == RiskLevel.CRITICAL:
            logger.error(f"⚠️ {description} - {action}")
        elif severity == RiskLevel.WARNING:
            logger.warning(f"⚡ {description} - {action}")
        else:
            logger.info(f"ℹ️ {description} - {action}")
    
    def _send_critical_alert(self, reason: str):
        """Send critical alert for kill switch activation."""
        alert_key = f"kill_switch_{datetime.now().strftime('%Y%m%d_%H')}"
        
        if alert_key not in self.alerts_sent:
            # In a real system, this would send alerts via:
            # - Email, SMS, Slack, PagerDuty, etc.
            logger.critical(f"📧 CRITICAL ALERT: Kill switch activated - {reason}")
            self.alerts_sent.append(alert_key)
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self.kill_switch_active and self.trading_mode != TradingMode.SHUTDOWN
    
    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status."""
        return {
            'risk_level': self.current_risk_level.value,
            'trading_mode': self.trading_mode.value,
            'kill_switch_active': self.kill_switch_active,
            'daily_pnl': self.daily_pnl,
            'total_drawdown': self.total_drawdown,
            'current_equity': self.current_equity,
            'position_count': len(self.positions),
            'last_update': self.last_update.isoformat(),
            'limits': self.limits.dict(),
            'recent_events': len([e for e in self.risk_events if e.timestamp > datetime.now() - timedelta(hours=1)])
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking at market open."""
        self.daily_start_equity = self.current_equity
        self.daily_pnl = 0.0
        self.api_success_count = 0
        self.api_total_count = 0
        logger.info("Daily risk tracking reset")
    
    def manual_kill_switch(self, reason: str = "Manual activation"):
        """Manually activate kill switch."""
        self._activate_kill_switch(reason, "manual")
    
    def reset_kill_switch(self, reason: str = "Manual reset"):
        """Reset kill switch (requires manual intervention)."""
        if self.kill_switch_active:
            self.kill_switch_active = False
            self.current_risk_level = RiskLevel.NORMAL
            self.trading_mode = TradingMode.ACTIVE
            
            self._log_risk_event(
                "kill_switch_reset",
                RiskLevel.NORMAL,
                f"Kill switch reset: {reason}",
                0.0, 0.0,
                "Trading resumed"
            )
            
            logger.info(f"🔄 Kill switch reset: {reason}")


# Risk monitoring utilities
class RiskMonitor:
    """Risk monitoring service for continuous surveillance."""
    
    def __init__(self, risk_guard: RiskGuard):
        self.risk_guard = risk_guard
        self.monitoring_active = False
        self._monitoring_task = None
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous risk monitoring."""
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info(f"Risk monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop risk monitoring."""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for stale data
                if self.risk_guard.last_update:
                    time_since_update = datetime.now() - self.risk_guard.last_update
                    if time_since_update.total_seconds() > 300:  # 5 minutes
                        self.risk_guard._activate_kill_switch(
                            f"Stale portfolio data: {time_since_update.total_seconds():.0f}s",
                            "stale_data"
                        )
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(interval_seconds)