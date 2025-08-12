"""
Kill Switch System

Emergency trading halt system that monitors multiple risk factors
and can immediately stop all trading activities.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class KillSwitchLevel(Enum):
    """Kill switch severity levels"""
    ACTIVE = "active"           # Normal operation
    WARNING = "warning"         # Warning level - reduce risk
    HALT = "halt"              # Halt new positions
    EMERGENCY = "emergency"     # Emergency stop all activity


class KillSwitchTrigger(Enum):
    """Types of kill switch triggers"""
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS = "daily_loss"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SHOCK = "volatility_shock"
    DATA_QUALITY = "data_quality"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_SIGNAL = "external_signal"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class KillSwitchEvent:
    """Kill switch activation event"""
    event_id: str
    timestamp: datetime
    trigger: KillSwitchTrigger
    level: KillSwitchLevel
    
    # Trigger details
    trigger_value: float
    threshold_value: float
    description: str
    
    # System state
    positions_at_trigger: int
    portfolio_value: float
    
    # Actions taken
    actions_taken: List[str]
    
    # Resolution
    resolved_at: Optional[datetime] = None
    resolution_reason: Optional[str] = None
    manual_override: bool = False
    
    @property
    def is_active(self) -> bool:
        """Check if kill switch is still active"""
        return self.resolved_at is None
    
    @property
    def duration_minutes(self) -> float:
        """Get duration of kill switch activation"""
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.timestamp).total_seconds() / 60


@dataclass 
class KillSwitchConfig:
    """Kill switch configuration parameters"""
    # Drawdown limits
    max_daily_loss_pct: float = 0.05        # 5%
    max_portfolio_dd_pct: float = 0.15       # 15%
    max_position_loss_pct: float = 0.20      # 20% per position
    
    # Correlation limits
    max_correlation_threshold: float = 0.9   # 90% max correlation
    correlation_window_hours: int = 24       # 24 hour window
    
    # Volatility limits
    volatility_shock_multiplier: float = 3.0  # 3x normal volatility
    volatility_window_hours: int = 4          # 4 hour window
    
    # Data quality limits
    max_missing_data_pct: float = 0.1        # 10% missing data
    max_stale_data_minutes: int = 30         # 30 minutes stale
    min_data_quality_score: float = 0.7     # 70% quality score
    
    # System limits
    max_consecutive_errors: int = 5          # 5 consecutive errors
    max_error_rate_per_hour: int = 20        # 20 errors per hour
    
    # Auto-resolution
    auto_resolve_after_minutes: int = 60     # Auto resolve after 1 hour
    require_manual_resolution: bool = False   # Require manual resolution


class KillSwitch:
    """
    Advanced kill switch system for emergency trading halts
    """
    
    def __init__(self, config: Optional[KillSwitchConfig] = None):
        self.config = config or KillSwitchConfig()
        
        self.current_level = KillSwitchLevel.ACTIVE
        self.active_events = []
        self.event_history = []
        
        # Monitoring state
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        self.hourly_error_count = []
        
        # Portfolio tracking
        self.portfolio_value_history = []
        self.position_count_history = []
        
        # External monitoring
        self.external_signals = {}
        
    def monitor_portfolio_risk(self, 
                              portfolio_data: Dict[str, Any],
                              timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Monitor portfolio-level risk factors"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            portfolio_value = portfolio_data.get("total_value", 0)
            position_count = portfolio_data.get("position_count", 0)
            daily_pnl_pct = portfolio_data.get("daily_pnl_pct", 0)
            max_dd_pct = portfolio_data.get("max_drawdown_pct", 0)
            
            # Store history
            self.portfolio_value_history.append((timestamp, portfolio_value))
            self.position_count_history.append((timestamp, position_count))
            
            triggers = []
            
            # Check daily loss limit
            if daily_pnl_pct < -self.config.max_daily_loss_pct:
                triggers.append({
                    "trigger": KillSwitchTrigger.DAILY_LOSS,
                    "level": KillSwitchLevel.EMERGENCY,
                    "value": daily_pnl_pct,
                    "threshold": -self.config.max_daily_loss_pct,
                    "description": f"Daily loss {daily_pnl_pct:.1%} exceeds limit {self.config.max_daily_loss_pct:.1%}"
                })
            
            # Check portfolio drawdown
            if max_dd_pct > self.config.max_portfolio_dd_pct:
                level = KillSwitchLevel.EMERGENCY if max_dd_pct > self.config.max_portfolio_dd_pct * 1.5 else KillSwitchLevel.HALT
                triggers.append({
                    "trigger": KillSwitchTrigger.DRAWDOWN_LIMIT,
                    "level": level,
                    "value": max_dd_pct,
                    "threshold": self.config.max_portfolio_dd_pct,
                    "description": f"Portfolio drawdown {max_dd_pct:.1%} exceeds limit {self.config.max_portfolio_dd_pct:.1%}"
                })
            
            # Process triggers
            for trigger_info in triggers:
                self._activate_kill_switch(
                    trigger_info["trigger"],
                    trigger_info["level"],
                    trigger_info["value"],
                    trigger_info["threshold"],
                    trigger_info["description"],
                    portfolio_data
                )
            
            return {
                "timestamp": timestamp,
                "current_level": self.current_level.value,
                "triggers_detected": len(triggers),
                "active_events": len(self.active_events),
                "monitoring_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk monitoring failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def monitor_correlation_risk(self, 
                                correlation_matrix: np.ndarray,
                                pair_names: List[str],
                                timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Monitor correlation risk across positions"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if correlation_matrix.size == 0:
                return {"status": "no_data"}
            
            # Find maximum correlation (excluding diagonal)
            np.fill_diagonal(correlation_matrix, 0)  # Remove self-correlation
            max_correlation = np.max(np.abs(correlation_matrix))
            
            # Find pairs with highest correlation
            max_indices = np.unravel_index(np.argmax(np.abs(correlation_matrix)), correlation_matrix.shape)
            max_pair = (pair_names[max_indices[0]], pair_names[max_indices[1]]) if len(pair_names) > max(max_indices) else ("Unknown", "Unknown")
            
            # Check correlation spike
            if max_correlation > self.config.max_correlation_threshold:
                level = KillSwitchLevel.EMERGENCY if max_correlation > 0.95 else KillSwitchLevel.WARNING
                
                self._activate_kill_switch(
                    KillSwitchTrigger.CORRELATION_SPIKE,
                    level,
                    max_correlation,
                    self.config.max_correlation_threshold,
                    f"Correlation spike: {max_pair[0]}-{max_pair[1]} correlation {max_correlation:.2%}",
                    {"correlation_matrix": correlation_matrix.tolist(), "max_pair": max_pair}
                )
            
            return {
                "timestamp": timestamp,
                "max_correlation": max_correlation,
                "max_correlation_pair": max_pair,
                "threshold": self.config.max_correlation_threshold,
                "risk_level": "high" if max_correlation > 0.8 else "normal"
            }
            
        except Exception as e:
            logger.error(f"Correlation risk monitoring failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def monitor_volatility_shock(self, 
                                current_volatility: float,
                                baseline_volatility: float,
                                timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Monitor for volatility shocks"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if baseline_volatility <= 0:
                return {"status": "invalid_baseline"}
            
            volatility_ratio = current_volatility / baseline_volatility
            
            # Check for volatility shock
            if volatility_ratio > self.config.volatility_shock_multiplier:
                level = KillSwitchLevel.EMERGENCY if volatility_ratio > 5.0 else KillSwitchLevel.HALT
                
                self._activate_kill_switch(
                    KillSwitchTrigger.VOLATILITY_SHOCK,
                    level,
                    volatility_ratio,
                    self.config.volatility_shock_multiplier,
                    f"Volatility shock: {volatility_ratio:.1f}x baseline volatility",
                    {"current_vol": current_volatility, "baseline_vol": baseline_volatility}
                )
            
            return {
                "timestamp": timestamp,
                "current_volatility": current_volatility,
                "baseline_volatility": baseline_volatility,
                "volatility_ratio": volatility_ratio,
                "threshold": self.config.volatility_shock_multiplier,
                "shock_detected": volatility_ratio > self.config.volatility_shock_multiplier
            }
            
        except Exception as e:
            logger.error(f"Volatility shock monitoring failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def monitor_data_quality(self, 
                           data_quality_metrics: Dict[str, Any],
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Monitor data quality for kill switch triggers"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            missing_data_pct = data_quality_metrics.get("missing_data_pct", 0)
            stale_data_minutes = data_quality_metrics.get("stale_data_minutes", 0)
            quality_score = data_quality_metrics.get("quality_score", 1.0)
            
            triggers = []
            
            # Check missing data
            if missing_data_pct > self.config.max_missing_data_pct:
                triggers.append({
                    "type": "missing_data",
                    "value": missing_data_pct,
                    "threshold": self.config.max_missing_data_pct,
                    "level": KillSwitchLevel.HALT
                })
            
            # Check stale data
            if stale_data_minutes > self.config.max_stale_data_minutes:
                triggers.append({
                    "type": "stale_data",
                    "value": stale_data_minutes,
                    "threshold": self.config.max_stale_data_minutes,
                    "level": KillSwitchLevel.WARNING
                })
            
            # Check quality score
            if quality_score < self.config.min_data_quality_score:
                triggers.append({
                    "type": "low_quality",
                    "value": quality_score,
                    "threshold": self.config.min_data_quality_score,
                    "level": KillSwitchLevel.HALT
                })
            
            # Process triggers
            for trigger_info in triggers:
                self._activate_kill_switch(
                    KillSwitchTrigger.DATA_QUALITY,
                    trigger_info["level"],
                    trigger_info["value"],
                    trigger_info["threshold"],
                    f"Data quality issue: {trigger_info['type']} {trigger_info['value']} exceeds threshold {trigger_info['threshold']}",
                    data_quality_metrics
                )
            
            return {
                "timestamp": timestamp,
                "data_quality_ok": len(triggers) == 0,
                "triggers_detected": len(triggers),
                "quality_metrics": data_quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Data quality monitoring failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def record_system_error(self, 
                           error_type: str,
                           error_description: str,
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Record system error and check for kill switch triggers"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            self.error_count += 1
            
            # Track consecutive errors
            if self.last_error_time and (timestamp - self.last_error_time).total_seconds() < 300:  # 5 minutes
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 1
            
            self.last_error_time = timestamp
            
            # Track hourly error rate
            self.hourly_error_count.append(timestamp)
            
            # Keep only last hour
            hour_ago = timestamp - timedelta(hours=1)
            self.hourly_error_count = [t for t in self.hourly_error_count if t >= hour_ago]
            
            # Check consecutive error threshold
            if self.consecutive_errors >= self.config.max_consecutive_errors:
                self._activate_kill_switch(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    KillSwitchLevel.HALT,
                    self.consecutive_errors,
                    self.config.max_consecutive_errors,
                    f"Consecutive system errors: {self.consecutive_errors}",
                    {"error_type": error_type, "description": error_description}
                )
            
            # Check hourly error rate
            elif len(self.hourly_error_count) >= self.config.max_error_rate_per_hour:
                self._activate_kill_switch(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    KillSwitchLevel.WARNING,
                    len(self.hourly_error_count),
                    self.config.max_error_rate_per_hour,
                    f"High error rate: {len(self.hourly_error_count)} errors in last hour",
                    {"error_type": error_type, "description": error_description}
                )
            
            return {
                "timestamp": timestamp,
                "total_errors": self.error_count,
                "consecutive_errors": self.consecutive_errors,
                "hourly_errors": len(self.hourly_error_count),
                "kill_switch_triggered": self.current_level != KillSwitchLevel.ACTIVE
            }
            
        except Exception as e:
            logger.error(f"System error recording failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def manual_trigger(self, 
                      level: KillSwitchLevel,
                      reason: str,
                      operator: str = "System") -> bool:
        """Manually trigger kill switch"""
        try:
            self._activate_kill_switch(
                KillSwitchTrigger.MANUAL_TRIGGER,
                level,
                1.0,
                0.0,
                f"Manual trigger by {operator}: {reason}",
                {"operator": operator, "reason": reason}
            )
            
            logger.warning(f"Kill switch manually triggered: {level.value} - {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Manual kill switch trigger failed: {e}")
            return False
    
    def resolve_kill_switch(self, 
                           event_id: str,
                           reason: str,
                           manual_override: bool = False) -> bool:
        """Resolve a kill switch event"""
        try:
            # Find active event
            event_to_resolve = None
            for event in self.active_events:
                if event.event_id == event_id:
                    event_to_resolve = event
                    break
            
            if not event_to_resolve:
                logger.warning(f"Kill switch event not found: {event_id}")
                return False
            
            # Resolve the event
            event_to_resolve.resolved_at = datetime.now()
            event_to_resolve.resolution_reason = reason
            event_to_resolve.manual_override = manual_override
            
            # Move to history
            self.active_events.remove(event_to_resolve)
            self.event_history.append(event_to_resolve)
            
            # Update current level
            self._update_current_level()
            
            logger.info(f"Kill switch event resolved: {event_id} - {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Kill switch resolution failed: {e}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        try:
            status = {
                "current_level": self.current_level.value,
                "active_events": len(self.active_events),
                "trading_allowed": self.current_level in [KillSwitchLevel.ACTIVE, KillSwitchLevel.WARNING],
                "new_positions_allowed": self.current_level == KillSwitchLevel.ACTIVE,
                "event_details": []
            }
            
            # Add active event details
            for event in self.active_events:
                status["event_details"].append({
                    "event_id": event.event_id,
                    "trigger": event.trigger.value,
                    "level": event.level.value,
                    "description": event.description,
                    "duration_minutes": event.duration_minutes,
                    "actions_taken": event.actions_taken
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_kill_switch_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get kill switch performance analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent events
            recent_events = [
                event for event in self.event_history
                if event.timestamp >= cutoff_time
            ]
            
            if not recent_events:
                return {"status": "no_events"}
            
            analytics = {
                "total_events": len(recent_events),
                "events_by_trigger": {},
                "events_by_level": {},
                "average_duration_minutes": np.mean([e.duration_minutes for e in recent_events]),
                "longest_event_minutes": max(e.duration_minutes for e in recent_events),
                "manual_overrides": sum(1 for e in recent_events if e.manual_override),
                "auto_resolutions": sum(1 for e in recent_events if not e.manual_override and e.resolved_at),
                "false_positive_rate": self._calculate_false_positive_rate(recent_events)
            }
            
            # Trigger distribution
            for trigger in KillSwitchTrigger:
                count = sum(1 for e in recent_events if e.trigger == trigger)
                analytics["events_by_trigger"][trigger.value] = count
            
            # Level distribution
            for level in KillSwitchLevel:
                count = sum(1 for e in recent_events if e.level == level)
                analytics["events_by_level"][level.value] = count
            
            return analytics
            
        except Exception as e:
            logger.error(f"Kill switch analytics failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _activate_kill_switch(self, 
                             trigger: KillSwitchTrigger,
                             level: KillSwitchLevel,
                             trigger_value: float,
                             threshold_value: float,
                             description: str,
                             context_data: Dict[str, Any]) -> None:
        """Activate kill switch with specified parameters"""
        try:
            # Check if similar event is already active
            for active_event in self.active_events:
                if (active_event.trigger == trigger and 
                    active_event.level == level and
                    (datetime.now() - active_event.timestamp).total_seconds() < 300):  # 5 minutes
                    return  # Don't create duplicate
            
            event_id = f"ks_{trigger.value}_{datetime.now().timestamp()}"
            
            # Determine actions to take
            actions_taken = self._determine_kill_switch_actions(level)
            
            event = KillSwitchEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                trigger=trigger,
                level=level,
                trigger_value=trigger_value,
                threshold_value=threshold_value,
                description=description,
                positions_at_trigger=context_data.get("position_count", 0),
                portfolio_value=context_data.get("total_value", 0),
                actions_taken=actions_taken
            )
            
            self.active_events.append(event)
            
            # Update current level
            self._update_current_level()
            
            # Execute actions
            self._execute_kill_switch_actions(actions_taken)
            
            logger.critical(f"Kill switch activated: {trigger.value} - {level.value} - {description}")
            
        except Exception as e:
            logger.error(f"Kill switch activation failed: {e}")
    
    def _update_current_level(self) -> None:
        """Update current kill switch level based on active events"""
        if not self.active_events:
            self.current_level = KillSwitchLevel.ACTIVE
        else:
            # Use highest severity level from active events
            levels = [event.level for event in self.active_events]
            level_priority = {
                KillSwitchLevel.ACTIVE: 0,
                KillSwitchLevel.WARNING: 1,
                KillSwitchLevel.HALT: 2,
                KillSwitchLevel.EMERGENCY: 3
            }
            
            highest_priority = max(level_priority[level] for level in levels)
            for level, priority in level_priority.items():
                if priority == highest_priority:
                    self.current_level = level
                    break
    
    def _determine_kill_switch_actions(self, level: KillSwitchLevel) -> List[str]:
        """Determine actions to take for kill switch level"""
        actions = []
        
        if level == KillSwitchLevel.WARNING:
            actions.extend([
                "reduce_position_sizes",
                "increase_monitoring",
                "alert_operators"
            ])
        elif level == KillSwitchLevel.HALT:
            actions.extend([
                "halt_new_positions",
                "reduce_existing_positions",
                "enable_close_only_mode",
                "alert_operators"
            ])
        elif level == KillSwitchLevel.EMERGENCY:
            actions.extend([
                "emergency_stop_trading",
                "close_all_positions",
                "disable_new_orders",
                "immediate_operator_notification"
            ])
        
        return actions
    
    def _execute_kill_switch_actions(self, actions: List[str]) -> None:
        """Execute kill switch actions"""
        for action in actions:
            try:
                if action == "alert_operators":
                    logger.critical("KILL SWITCH: Operator alert triggered")
                elif action == "halt_new_positions":
                    logger.critical("KILL SWITCH: New positions halted")
                elif action == "emergency_stop_trading":
                    logger.critical("KILL SWITCH: Emergency trading stop activated")
                # Add more action implementations as needed
                
            except Exception as e:
                logger.error(f"Failed to execute kill switch action {action}: {e}")
    
    def _calculate_false_positive_rate(self, events: List[KillSwitchEvent]) -> float:
        """Calculate false positive rate for kill switch events"""
        try:
            if not events:
                return 0.0
            
            # Simple heuristic: events resolved quickly might be false positives
            false_positives = sum(
                1 for event in events
                if event.duration_minutes < 10 and event.manual_override
            )
            
            return false_positives / len(events)
            
        except Exception as e:
            logger.error(f"False positive rate calculation failed: {e}")
            return 0.0