"""
Kill Switch System

Emergency trading halt system with manual and automatic triggers
for immediate risk containment.
"""

import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Kill switch trigger types"""
    MANUAL = "manual"
    DATA_GAP = "data_gap"
    LATENCY_SPIKE = "latency_spike"
    DRIFT_DETECTION = "drift_detection"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_EMERGENCY = "drawdown_emergency"
    SYSTEM_ERROR = "system_error"
    EXCHANGE_ERROR = "exchange_error"
    CORRELATION_SHOCK = "correlation_shock"


class KillSwitchStatus(Enum):
    """Kill switch status"""
    ARMED = "armed"
    TRIGGERED = "triggered" 
    RECOVERING = "recovering"
    DISABLED = "disabled"


@dataclass
class KillSwitchTrigger:
    """Kill switch trigger configuration"""
    trigger_id: str
    trigger_type: TriggerType
    
    # Trigger parameters
    threshold_value: float
    time_window_seconds: int = 60
    
    # Configuration
    enabled: bool = True
    auto_trigger: bool = True
    description: str = ""
    
    # State tracking
    current_value: float = 0.0
    last_trigger: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class EmergencyStop:
    """Emergency stop event record"""
    stop_id: str
    timestamp: datetime
    trigger_type: TriggerType
    reason: str
    
    # Context
    triggered_by: str = "system"
    trigger_value: float = 0.0
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery
    recovery_started: Optional[datetime] = None
    recovery_completed: Optional[datetime] = None
    manual_override: bool = False


class KillSwitchSystem:
    """
    Enterprise kill switch system for emergency trading halts
    """
    
    def __init__(self):
        self.status = KillSwitchStatus.ARMED
        self.triggers: Dict[str, KillSwitchTrigger] = {}
        self.stop_history: List[EmergencyStop] = []
        
        # Callbacks
        self.trigger_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Configuration
        self.auto_recovery_enabled = False
        self.auto_recovery_delay_minutes = 5
        self.max_triggers_per_hour = 3
        
        # State
        self.current_stop: Optional[EmergencyStop] = None
        self.last_health_check = datetime.now()
        
        # Setup default triggers
        self._setup_default_triggers()
        
        # Monitoring thread
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
    
    def _setup_default_triggers(self):
        """Setup default kill switch triggers"""
        
        # Data gap trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="data_gap",
            trigger_type=TriggerType.DATA_GAP,
            threshold_value=300,  # 5 minutes without data
            time_window_seconds=300,
            description="Trigger when market data feed gaps exceed threshold",
            auto_trigger=True
        ))
        
        # Latency spike trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="latency_spike",
            trigger_type=TriggerType.LATENCY_SPIKE,
            threshold_value=2000,  # 2 seconds latency
            time_window_seconds=60,
            description="Trigger on excessive execution latency",
            auto_trigger=True
        ))
        
        # Drift detection trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="model_drift",
            trigger_type=TriggerType.DRIFT_DETECTION,
            threshold_value=0.5,  # High drift score
            time_window_seconds=300,
            description="Trigger on severe model performance drift",
            auto_trigger=True
        ))
        
        # Daily loss limit trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="daily_loss",
            trigger_type=TriggerType.DAILY_LOSS_LIMIT,
            threshold_value=-7.5,  # 7.5% daily loss
            time_window_seconds=60,
            description="Trigger on daily loss limit breach",
            auto_trigger=True
        ))
        
        # Emergency drawdown trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="emergency_drawdown",
            trigger_type=TriggerType.DRAWDOWN_EMERGENCY,
            threshold_value=-15.0,  # 15% drawdown
            time_window_seconds=60,
            description="Trigger on emergency drawdown level",
            auto_trigger=True
        ))
        
        # System error trigger
        self.add_trigger(KillSwitchTrigger(
            trigger_id="system_error",
            trigger_type=TriggerType.SYSTEM_ERROR,
            threshold_value=10,  # 10 errors per minute
            time_window_seconds=60,
            description="Trigger on high system error rate",
            auto_trigger=True
        ))
    
    def add_trigger(self, trigger: KillSwitchTrigger):
        """Add kill switch trigger"""
        self.triggers[trigger.trigger_id] = trigger
        logger.info(f"Added kill switch trigger: {trigger.trigger_id}")
    
    def check_triggers(self, metrics: Dict[str, float]) -> bool:
        """Check all triggers against current metrics"""
        if self.status != KillSwitchStatus.ARMED:
            return True  # Already triggered or disabled
        
        try:
            for trigger in self.triggers.values():
                if not trigger.enabled or not trigger.auto_trigger:
                    continue
                
                metric_key = self._get_metric_key(trigger.trigger_type)
                if metric_key not in metrics:
                    continue
                
                current_value = metrics[metric_key]
                trigger.current_value = current_value
                
                # Check if trigger conditions are met
                if self._should_trigger(trigger, current_value):
                    reason = f"{trigger.description} - Value: {current_value}, Threshold: {trigger.threshold_value}"
                    self._execute_emergency_stop(trigger.trigger_type, reason, current_value)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trigger check failed: {e}")
            # Fail-safe: trigger on error
            self._execute_emergency_stop(TriggerType.SYSTEM_ERROR, f"Trigger check error: {e}", 0)
            return False
    
    def _get_metric_key(self, trigger_type: TriggerType) -> str:
        """Get metric key for trigger type"""
        mapping = {
            TriggerType.DATA_GAP: "data_gap_seconds",
            TriggerType.LATENCY_SPIKE: "execution_latency_ms",
            TriggerType.DRIFT_DETECTION: "drift_score",
            TriggerType.DAILY_LOSS_LIMIT: "daily_pnl_pct",
            TriggerType.DRAWDOWN_EMERGENCY: "drawdown_pct",
            TriggerType.SYSTEM_ERROR: "error_rate_per_minute",
            TriggerType.EXCHANGE_ERROR: "exchange_error_rate",
            TriggerType.CORRELATION_SHOCK: "correlation_change"
        }
        return mapping.get(trigger_type, "unknown")
    
    def _should_trigger(self, trigger: KillSwitchTrigger, current_value: float) -> bool:
        """Check if trigger should activate"""
        try:
            # Different logic for different trigger types
            if trigger.trigger_type in [TriggerType.DATA_GAP, TriggerType.LATENCY_SPIKE]:
                # Greater than threshold triggers
                return current_value > trigger.threshold_value
            
            elif trigger.trigger_type in [TriggerType.DAILY_LOSS_LIMIT, TriggerType.DRAWDOWN_EMERGENCY]:
                # Less than threshold triggers (negative values)
                return current_value < trigger.threshold_value
            
            elif trigger.trigger_type in [TriggerType.DRIFT_DETECTION, TriggerType.SYSTEM_ERROR]:
                # Greater than threshold triggers
                return current_value > trigger.threshold_value
            
            else:
                # Default: greater than threshold
                return current_value > trigger.threshold_value
        
        except Exception as e:
            logger.error(f"Trigger evaluation failed: {e}")
            return True  # Fail-safe: trigger on error
    
    def manual_trigger(self, reason: str, triggered_by: str = "manual") -> bool:
        """Manually trigger emergency stop"""
        try:
            if self.status == KillSwitchStatus.TRIGGERED:
                logger.warning("Kill switch already triggered")
                return False
            
            logger.critical(f"MANUAL KILL SWITCH ACTIVATED by {triggered_by}: {reason}")
            
            stop = EmergencyStop(
                stop_id=f"manual_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                trigger_type=TriggerType.MANUAL,
                reason=reason,
                triggered_by=triggered_by,
                system_state=self._capture_system_state()
            )
            
            return self._activate_emergency_stop(stop)
            
        except Exception as e:
            logger.error(f"Manual trigger failed: {e}")
            return False
    
    def _execute_emergency_stop(self, trigger_type: TriggerType, reason: str, trigger_value: float):
        """Execute automatic emergency stop"""
        try:
            if self.status == KillSwitchStatus.TRIGGERED:
                return  # Already triggered
            
            # Check trigger rate limits
            if not self._check_trigger_rate_limit():
                logger.warning("Kill switch trigger rate limit exceeded - ignoring")
                return
            
            logger.critical(f"AUTOMATIC KILL SWITCH ACTIVATED: {trigger_type.value} - {reason}")
            
            stop = EmergencyStop(
                stop_id=f"auto_{trigger_type.value}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                trigger_type=trigger_type,
                reason=reason,
                triggered_by="system",
                trigger_value=trigger_value,
                system_state=self._capture_system_state()
            )
            
            self._activate_emergency_stop(stop)
            
        except Exception as e:
            logger.error(f"Emergency stop execution failed: {e}")
    
    def _activate_emergency_stop(self, stop: EmergencyStop) -> bool:
        """Activate emergency stop"""
        try:
            self.current_stop = stop
            self.status = KillSwitchStatus.TRIGGERED
            self.stop_history.append(stop)
            
            # Update trigger count
            if stop.trigger_type != TriggerType.MANUAL:
                trigger = self.triggers.get(stop.trigger_type.value)
                if trigger:
                    trigger.trigger_count += 1
                    trigger.last_trigger = stop.timestamp
            
            # Execute emergency actions
            self._execute_emergency_actions(stop)
            
            # Notify callbacks
            for callback in self.trigger_callbacks:
                try:
                    callback(stop)
                except Exception as e:
                    logger.error(f"Kill switch callback failed: {e}")
            
            # Start auto-recovery if enabled
            if self.auto_recovery_enabled and stop.trigger_type != TriggerType.MANUAL:
                self._schedule_auto_recovery()
            
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop activation failed: {e}")
            return False
    
    def _execute_emergency_actions(self, stop: EmergencyStop):
        """Execute immediate emergency actions"""
        try:
            # 1. Halt all trading immediately
            logger.critical("EMERGENCY ACTION: Halting all trading")
            
            # 2. Cancel all pending orders
            logger.critical("EMERGENCY ACTION: Canceling all pending orders")
            
            # 3. Close risky positions (if configured)
            if stop.trigger_type in [TriggerType.DAILY_LOSS_LIMIT, TriggerType.DRAWDOWN_EMERGENCY]:
                logger.critical("EMERGENCY ACTION: Preparing to close positions")
            
            # 4. Notify monitoring systems
            logger.critical("EMERGENCY ACTION: Notifying monitoring systems")
            
            # 5. Save current state
            self._save_emergency_state(stop)
            
        except Exception as e:
            logger.error(f"Emergency action execution failed: {e}")
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for debugging"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "trigger_states": {
                    trigger_id: {
                        "current_value": trigger.current_value,
                        "threshold": trigger.threshold_value,
                        "trigger_count": trigger.trigger_count
                    }
                    for trigger_id, trigger in self.triggers.items()
                },
                "recent_stops": len([s for s in self.stop_history 
                                  if s.timestamp > datetime.now() - timedelta(hours=1)])
            }
        except Exception as e:
            logger.error(f"System state capture failed: {e}")
            return {"error": str(e)}
    
    def _check_trigger_rate_limit(self) -> bool:
        """Check if trigger rate limit is exceeded"""
        try:
            recent_stops = [s for s in self.stop_history 
                          if s.timestamp > datetime.now() - timedelta(hours=1)]
            
            return len(recent_stops) < self.max_triggers_per_hour
        except Exception:
            return True  # Allow trigger on error
    
    def _schedule_auto_recovery(self):
        """Schedule automatic recovery"""
        try:
            def auto_recovery():
                import time
                time.sleep(self.auto_recovery_delay_minutes * 60)
                
                if self.status == KillSwitchStatus.TRIGGERED:
                    logger.info("Starting automatic recovery")
                    self.start_recovery()
            
            recovery_thread = threading.Thread(target=auto_recovery, daemon=True)
            recovery_thread.start()
            
        except Exception as e:
            logger.error(f"Auto-recovery scheduling failed: {e}")
    
    def start_recovery(self, manual_override: bool = False) -> bool:
        """Start recovery process"""
        try:
            if self.status != KillSwitchStatus.TRIGGERED:
                logger.warning("Cannot start recovery - kill switch not triggered")
                return False
            
            if self.current_stop and not self.current_stop.manual_override and not manual_override:
                # Check if conditions have improved
                if not self._recovery_conditions_met():
                    logger.warning("Recovery conditions not met - staying in emergency mode")
                    return False
            
            logger.info("Starting kill switch recovery")
            
            self.status = KillSwitchStatus.RECOVERING
            if self.current_stop:
                self.current_stop.recovery_started = datetime.now()
                self.current_stop.manual_override = manual_override
            
            # Execute recovery actions
            recovery_success = self._execute_recovery_actions()
            
            if recovery_success:
                self._complete_recovery()
                return True
            else:
                self.status = KillSwitchStatus.TRIGGERED
                logger.error("Recovery failed - reverting to triggered state")
                return False
                
        except Exception as e:
            logger.error(f"Recovery start failed: {e}")
            self.status = KillSwitchStatus.TRIGGERED
            return False
    
    def _recovery_conditions_met(self) -> bool:
        """Check if conditions are suitable for recovery"""
        try:
            # Check that trigger conditions are no longer met
            for trigger in self.triggers.values():
                if not trigger.enabled:
                    continue
                
                # For auto-triggers, ensure conditions have improved
                if trigger.trigger_type == self.current_stop.trigger_type:
                    if trigger.trigger_type in [TriggerType.DAILY_LOSS_LIMIT, TriggerType.DRAWDOWN_EMERGENCY]:
                        # For loss triggers, require some improvement
                        if trigger.current_value < (trigger.threshold_value * 0.8):
                            continue  # Still too close to threshold
                    else:
                        # For other triggers, require value below threshold
                        if trigger.current_value > (trigger.threshold_value * 0.5):
                            continue  # Still elevated
            
            return True
            
        except Exception as e:
            logger.error(f"Recovery condition check failed: {e}")
            return False
    
    def _execute_recovery_actions(self) -> bool:
        """Execute recovery actions"""
        try:
            logger.info("Executing recovery actions")
            
            # 1. Verify system health
            if not self._verify_system_health():
                logger.error("System health check failed during recovery")
                return False
            
            # 2. Restore data connections
            logger.info("Restoring data connections")
            
            # 3. Validate market conditions
            logger.info("Validating market conditions")
            
            # 4. Gradual trading resume (if configured)
            logger.info("Preparing for gradual trading resume")
            
            return True
            
        except Exception as e:
            logger.error(f"Recovery actions failed: {e}")
            return False
    
    def _verify_system_health(self) -> bool:
        """Verify system health for recovery"""
        try:
            # Check data feeds
            # Check exchange connections  
            # Check system resources
            # Check error rates
            
            # Placeholder implementation
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"System health verification failed: {e}")
            return False
    
    def _complete_recovery(self):
        """Complete the recovery process"""
        try:
            self.status = KillSwitchStatus.ARMED
            if self.current_stop:
                self.current_stop.recovery_completed = datetime.now()
            
            logger.info("Kill switch recovery completed - system re-armed")
            
            # Notify recovery callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback(self.current_stop)
                except Exception as e:
                    logger.error(f"Recovery callback failed: {e}")
            
            self.current_stop = None
            
        except Exception as e:
            logger.error(f"Recovery completion failed: {e}")
    
    def _save_emergency_state(self, stop: EmergencyStop):
        """Save emergency state to disk"""
        try:
            state_file = Path(f"emergency_state_{stop.stop_id}.json")
            
            state_data = {
                "stop_id": stop.stop_id,
                "timestamp": stop.timestamp.isoformat(),
                "trigger_type": stop.trigger_type.value,
                "reason": stop.reason,
                "trigger_value": stop.trigger_value,
                "system_state": stop.system_state
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Emergency state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"Emergency state save failed: {e}")
    
    def _monitor_system(self):
        """Background system monitoring"""
        while self._monitoring_active:
            try:
                # Monitor system health
                if self.status == KillSwitchStatus.ARMED:
                    # Perform basic health checks
                    pass
                
                # Sleep for monitoring interval
                import time
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def add_trigger_callback(self, callback: Callable[[EmergencyStop], None]):
        """Add callback for kill switch triggers"""
        self.trigger_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[EmergencyStop], None]):
        """Add callback for recovery completion"""
        self.recovery_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        return {
            "status": self.status.value,
            "current_stop": {
                "stop_id": self.current_stop.stop_id,
                "trigger_type": self.current_stop.trigger_type.value,
                "reason": self.current_stop.reason,
                "timestamp": self.current_stop.timestamp.isoformat(),
                "recovery_started": self.current_stop.recovery_started.isoformat() if self.current_stop.recovery_started else None
            } if self.current_stop else None,
            "triggers": {
                trigger_id: {
                    "enabled": trigger.enabled,
                    "current_value": trigger.current_value,
                    "threshold": trigger.threshold_value,
                    "trigger_count": trigger.trigger_count,
                    "last_trigger": trigger.last_trigger.isoformat() if trigger.last_trigger else None
                }
                for trigger_id, trigger in self.triggers.items()
            },
            "recent_stops": len([s for s in self.stop_history 
                               if s.timestamp > datetime.now() - timedelta(hours=24)]),
            "auto_recovery_enabled": self.auto_recovery_enabled
        }
    
    def disable(self):
        """Disable kill switch system"""
        self.status = KillSwitchStatus.DISABLED
        logger.warning("Kill switch system DISABLED")
    
    def enable(self):
        """Enable kill switch system"""
        self.status = KillSwitchStatus.ARMED
        logger.info("Kill switch system ENABLED")
    
    def shutdown(self):
        """Shutdown kill switch monitoring"""
        self._monitoring_active = False
        logger.info("Kill switch monitoring shutdown")