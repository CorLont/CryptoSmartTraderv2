"""
Risk Limits Manager

Comprehensive risk limit system with daily loss limits,
maximum drawdown guards, and exposure controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Types of risk limits"""
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_SIZE = "position_size"
    EXPOSURE_CONCENTRATION = "exposure_concentration"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    LEVERAGE_LIMIT = "leverage_limit"


class LimitStatus(Enum):
    """Risk limit status"""
    SAFE = "safe"
    WARNING = "warning"
    BREACHED = "breached"
    EMERGENCY = "emergency"


@dataclass
class RiskLimit:
    """Individual risk limit definition"""
    limit_id: str
    limit_type: LimitType
    
    # Limit parameters
    soft_limit: float           # Warning threshold
    hard_limit: float           # Absolute limit
    emergency_limit: float      # Emergency stop threshold
    
    # Current values
    current_value: float = 0.0
    current_status: LimitStatus = LimitStatus.SAFE
    
    # Configuration
    enabled: bool = True
    auto_action: str = "alert"  # "alert", "reduce_positions", "stop_trading"
    
    # Metadata
    description: str = ""
    last_check: Optional[datetime] = None
    breach_count_today: int = 0
    
    @property
    def utilization_pct(self) -> float:
        """Current limit utilization as percentage"""
        if self.hard_limit == 0:
            return 0.0
        return abs(self.current_value / self.hard_limit) * 100
    
    @property
    def is_breached(self) -> bool:
        """Check if limit is breached"""
        return self.current_status in [LimitStatus.BREACHED, LimitStatus.EMERGENCY]
    
    def update_status(self, value: float) -> LimitStatus:
        """Update limit status based on current value"""
        self.current_value = value
        self.last_check = datetime.now()
        
        # Determine status based on thresholds
        if abs(value) >= abs(self.emergency_limit):
            self.current_status = LimitStatus.EMERGENCY
        elif abs(value) >= abs(self.hard_limit):
            self.current_status = LimitStatus.BREACHED
            self.breach_count_today += 1
        elif abs(value) >= abs(self.soft_limit):
            self.current_status = LimitStatus.WARNING
        else:
            self.current_status = LimitStatus.SAFE
        
        return self.current_status


class RiskLimitManager:
    """
    Enterprise risk limit management system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.limits: Dict[str, RiskLimit] = {}
        self.limit_history = []
        self.daily_reset_time = "00:00"
        
        # Default risk limits
        self._setup_default_limits()
        
        # Load custom configuration if provided
        if config_file:
            self._load_config(config_file)
        
        # Callbacks for limit breaches
        self.breach_callbacks = []
        
        # Daily tracking
        self.daily_start_equity = 0.0
        self.daily_peak_equity = 0.0
        self.last_reset_date = date.today()
    
    def _setup_default_limits(self):
        """Setup default risk limits"""
        
        # Daily loss limit (5% of starting equity)
        self.add_limit(RiskLimit(
            limit_id="daily_loss",
            limit_type=LimitType.DAILY_LOSS,
            soft_limit=-3.0,      # 3% warning
            hard_limit=-5.0,      # 5% stop trading
            emergency_limit=-7.5, # 7.5% emergency stop
            description="Daily portfolio loss limit",
            auto_action="stop_trading"
        ))
        
        # Maximum drawdown limit (10% from peak)
        self.add_limit(RiskLimit(
            limit_id="max_drawdown",
            limit_type=LimitType.MAX_DRAWDOWN,
            soft_limit=-8.0,      # 8% warning
            hard_limit=-10.0,     # 10% reduce positions
            emergency_limit=-15.0, # 15% emergency stop
            description="Maximum drawdown from peak equity",
            auto_action="reduce_positions"
        ))
        
        # Position size limit (2% per position)
        self.add_limit(RiskLimit(
            limit_id="position_size",
            limit_type=LimitType.POSITION_SIZE,
            soft_limit=1.5,       # 1.5% warning
            hard_limit=2.0,       # 2% hard limit
            emergency_limit=3.0,  # 3% emergency
            description="Maximum position size per asset",
            auto_action="reject_order"
        ))
        
        # Exposure concentration (20% per asset cluster)
        self.add_limit(RiskLimit(
            limit_id="exposure_concentration",
            limit_type=LimitType.EXPOSURE_CONCENTRATION,
            soft_limit=15.0,      # 15% warning
            hard_limit=20.0,      # 20% hard limit
            emergency_limit=25.0, # 25% emergency
            description="Maximum exposure per asset cluster",
            auto_action="reject_order"
        ))
        
        # Correlation limit (max 70% correlated exposure)
        self.add_limit(RiskLimit(
            limit_id="correlation",
            limit_type=LimitType.CORRELATION_LIMIT,
            soft_limit=60.0,      # 60% warning
            hard_limit=70.0,      # 70% hard limit
            emergency_limit=80.0, # 80% emergency
            description="Maximum correlated exposure",
            auto_action="reduce_positions"
        ))
        
        # Volatility limit (max 25% portfolio volatility)
        self.add_limit(RiskLimit(
            limit_id="volatility",
            limit_type=LimitType.VOLATILITY_LIMIT,
            soft_limit=20.0,      # 20% warning
            hard_limit=25.0,      # 25% hard limit
            emergency_limit=30.0, # 30% emergency
            description="Maximum portfolio volatility",
            auto_action="reduce_positions"
        ))
    
    def add_limit(self, limit: RiskLimit):
        """Add or update a risk limit"""
        self.limits[limit.limit_id] = limit
        logger.info(f"Added risk limit: {limit.limit_id} ({limit.limit_type.value})")
    
    def update_daily_metrics(self, current_equity: float, current_positions: Dict[str, float]):
        """Update daily risk metrics"""
        try:
            # Check if we need to reset daily tracking
            today = date.today()
            if today != self.last_reset_date:
                self._reset_daily_tracking(current_equity)
                self.last_reset_date = today
            
            # Update daily peak
            if current_equity > self.daily_peak_equity:
                self.daily_peak_equity = current_equity
            
            # Calculate daily PnL
            if self.daily_start_equity > 0:
                daily_pnl_pct = (current_equity - self.daily_start_equity) / self.daily_start_equity * 100
            else:
                daily_pnl_pct = 0.0
            
            # Calculate drawdown from peak
            if self.daily_peak_equity > 0:
                drawdown_pct = (current_equity - self.daily_peak_equity) / self.daily_peak_equity * 100
            else:
                drawdown_pct = 0.0
            
            # Update limits
            self._update_limit("daily_loss", daily_pnl_pct)
            self._update_limit("max_drawdown", drawdown_pct)
            
            # Calculate position and exposure metrics
            self._update_position_limits(current_positions, current_equity)
            
        except Exception as e:
            logger.error(f"Daily metrics update failed: {e}")
    
    def _reset_daily_tracking(self, current_equity: float):
        """Reset daily tracking metrics"""
        self.daily_start_equity = current_equity
        self.daily_peak_equity = current_equity
        
        # Reset daily breach counts
        for limit in self.limits.values():
            limit.breach_count_today = 0
        
        logger.info(f"Daily risk tracking reset - starting equity: {current_equity:.2f}")
    
    def _update_limit(self, limit_id: str, value: float) -> bool:
        """Update a specific risk limit and check for breaches"""
        if limit_id not in self.limits:
            return False
        
        limit = self.limits[limit_id]
        if not limit.enabled:
            return True
        
        old_status = limit.current_status
        new_status = limit.update_status(value)
        
        # Check for status changes
        if new_status != old_status:
            self._handle_limit_change(limit, old_status, new_status)
        
        # Record in history
        self.limit_history.append({
            "timestamp": datetime.now(),
            "limit_id": limit_id,
            "value": value,
            "status": new_status.value,
            "utilization_pct": limit.utilization_pct
        })
        
        return new_status == LimitStatus.SAFE
    
    def _update_position_limits(self, positions: Dict[str, float], total_equity: float):
        """Update position-based risk limits"""
        try:
            if total_equity <= 0:
                return
            
            # Calculate individual position sizes
            position_sizes = {asset: abs(position) / total_equity * 100 
                            for asset, position in positions.items()}
            
            # Check maximum position size
            max_position_size = max(position_sizes.values()) if position_sizes else 0
            self._update_limit("position_size", max_position_size)
            
            # Calculate cluster exposure (simplified - would use actual clustering)
            # For now, group by asset type prefix
            cluster_exposure = {}
            for asset, size in position_sizes.items():
                cluster = asset.split('/')[0] if '/' in asset else asset[:3]  # Simplified clustering
                cluster_exposure[cluster] = cluster_exposure.get(cluster, 0) + size
            
            max_cluster_exposure = max(cluster_exposure.values()) if cluster_exposure else 0
            self._update_limit("exposure_concentration", max_cluster_exposure)
            
            # Calculate total exposure
            total_exposure = sum(position_sizes.values())
            
            # Estimate correlation impact (simplified)
            # In reality, would use correlation matrix
            avg_correlation = 0.3  # Assume 30% average correlation
            correlation_adjusted_exposure = total_exposure * avg_correlation
            self._update_limit("correlation", correlation_adjusted_exposure)
            
        except Exception as e:
            logger.error(f"Position limit update failed: {e}")
    
    def _handle_limit_change(self, limit: RiskLimit, old_status: LimitStatus, new_status: LimitStatus):
        """Handle risk limit status changes"""
        
        logger.warning(f"Risk limit {limit.limit_id} status changed: {old_status.value} → {new_status.value}")
        
        # Execute auto-actions based on new status
        if new_status == LimitStatus.EMERGENCY:
            self._execute_emergency_action(limit)
        elif new_status == LimitStatus.BREACHED:
            self._execute_breach_action(limit)
        elif new_status == LimitStatus.WARNING:
            self._execute_warning_action(limit)
        
        # Notify callbacks
        for callback in self.breach_callbacks:
            try:
                callback(limit, old_status, new_status)
            except Exception as e:
                logger.error(f"Risk limit callback failed: {e}")
    
    def _execute_emergency_action(self, limit: RiskLimit):
        """Execute emergency actions for limit breach"""
        logger.critical(f"EMERGENCY: Risk limit {limit.limit_id} breached - immediate action required")
        
        if limit.auto_action == "stop_trading":
            self._trigger_emergency_stop(f"Emergency limit breach: {limit.limit_id}")
        elif limit.auto_action == "reduce_positions":
            self._trigger_position_reduction(0.5, f"Emergency reduction: {limit.limit_id}")
    
    def _execute_breach_action(self, limit: RiskLimit):
        """Execute actions for limit breach"""
        logger.error(f"BREACH: Risk limit {limit.limit_id} exceeded")
        
        if limit.auto_action == "stop_trading":
            self._trigger_trading_halt(f"Limit breach: {limit.limit_id}")
        elif limit.auto_action == "reduce_positions":
            self._trigger_position_reduction(0.25, f"Breach reduction: {limit.limit_id}")
        elif limit.auto_action == "reject_order":
            logger.warning(f"New orders will be rejected due to {limit.limit_id} breach")
    
    def _execute_warning_action(self, limit: RiskLimit):
        """Execute warning actions"""
        logger.warning(f"WARNING: Risk limit {limit.limit_id} approaching threshold")
        # Warning actions are typically just alerts
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency trading stop"""
        # This would integrate with the kill switch system
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def _trigger_trading_halt(self, reason: str):
        """Trigger trading halt"""
        # This would integrate with the trading engine
        logger.error(f"TRADING HALT TRIGGERED: {reason}")
    
    def _trigger_position_reduction(self, reduction_factor: float, reason: str):
        """Trigger position size reduction"""
        # This would integrate with the portfolio manager
        logger.warning(f"POSITION REDUCTION TRIGGERED ({reduction_factor*100:.0f}%): {reason}")
    
    def add_breach_callback(self, callback):
        """Add callback function for limit breaches"""
        self.breach_callbacks.append(callback)
    
    def check_order_against_limits(self, asset: str, order_size: float, current_equity: float) -> Tuple[bool, List[str]]:
        """Check if a new order would violate risk limits"""
        violations = []
        
        try:
            # Check position size limit
            position_size_pct = abs(order_size) / current_equity * 100
            position_limit = self.limits.get("position_size")
            
            if position_limit and position_size_pct > position_limit.hard_limit:
                violations.append(f"Order size ({position_size_pct:.1f}%) exceeds position limit ({position_limit.hard_limit:.1f}%)")
            
            # Check if trading is currently halted
            daily_loss_limit = self.limits.get("daily_loss")
            if daily_loss_limit and daily_loss_limit.is_breached:
                violations.append("Trading halted due to daily loss limit breach")
            
            drawdown_limit = self.limits.get("max_drawdown")
            if drawdown_limit and drawdown_limit.current_status == LimitStatus.EMERGENCY:
                violations.append("Trading halted due to emergency drawdown limit")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Order limit check failed: {e}")
            return False, [f"Risk check error: {e}"]
    
    def get_limit_status(self) -> Dict[str, Any]:
        """Get current status of all risk limits"""
        status = {}
        
        for limit_id, limit in self.limits.items():
            status[limit_id] = {
                "type": limit.limit_type.value,
                "current_value": limit.current_value,
                "status": limit.current_status.value,
                "utilization_pct": limit.utilization_pct,
                "soft_limit": limit.soft_limit,
                "hard_limit": limit.hard_limit,
                "emergency_limit": limit.emergency_limit,
                "breach_count_today": limit.breach_count_today,
                "last_check": limit.last_check.isoformat() if limit.last_check else None
            }
        
        return status
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        breached_limits = [limit for limit in self.limits.values() if limit.is_breached]
        warning_limits = [limit for limit in self.limits.values() if limit.current_status == LimitStatus.WARNING]
        
        overall_status = "EMERGENCY" if any(l.current_status == LimitStatus.EMERGENCY for l in self.limits.values()) else \
                        "BREACHED" if breached_limits else \
                        "WARNING" if warning_limits else \
                        "SAFE"
        
        return {
            "overall_status": overall_status,
            "total_limits": len(self.limits),
            "breached_limits": len(breached_limits),
            "warning_limits": len(warning_limits),
            "daily_pnl_pct": self.limits.get("daily_loss", RiskLimit("", LimitType.DAILY_LOSS, 0, 0, 0)).current_value,
            "max_drawdown_pct": self.limits.get("max_drawdown", RiskLimit("", LimitType.MAX_DRAWDOWN, 0, 0, 0)).current_value,
            "trading_allowed": overall_status not in ["EMERGENCY", "BREACHED"],
            "last_update": datetime.now().isoformat()
        }
    
    def _load_config(self, config_file: str):
        """Load risk limit configuration from file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update limits from configuration
                for limit_config in config.get("limits", []):
                    limit = RiskLimit(
                        limit_id=limit_config["limit_id"],
                        limit_type=LimitType(limit_config["limit_type"]),
                        soft_limit=limit_config["soft_limit"],
                        hard_limit=limit_config["hard_limit"],
                        emergency_limit=limit_config["emergency_limit"],
                        description=limit_config.get("description", ""),
                        auto_action=limit_config.get("auto_action", "alert"),
                        enabled=limit_config.get("enabled", True)
                    )
                    self.add_limit(limit)
                
                logger.info(f"Loaded risk configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load risk configuration: {e}")
    
    def save_config(self, config_file: str):
        """Save current risk limit configuration to file"""
        try:
            config = {
                "limits": [
                    {
                        "limit_id": limit.limit_id,
                        "limit_type": limit.limit_type.value,
                        "soft_limit": limit.soft_limit,
                        "hard_limit": limit.hard_limit,
                        "emergency_limit": limit.emergency_limit,
                        "description": limit.description,
                        "auto_action": limit.auto_action,
                        "enabled": limit.enabled
                    }
                    for limit in self.limits.values()
                ],
                "daily_reset_time": self.daily_reset_time
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved risk configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save risk configuration: {e}")