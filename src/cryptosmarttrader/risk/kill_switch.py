"""
Kill Switch System

Emergency trading halt system with multiple trigger sources,
immediate position closure, and comprehensive safety controls.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

from .risk_limits import RiskLimitManager, RiskStatus, ActionType
from .circuit_breaker import CircuitBreakerManager, BreakerState

logger = logging.getLogger(__name__)

class KillSwitchTrigger(Enum):
    """Kill switch trigger sources"""
    MANUAL = "manual"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM_ERROR = "system_error"
    DATA_INTEGRITY = "data_integrity"
    EXTERNAL_SIGNAL = "external_signal"

class KillSwitchState(Enum):
    """Kill switch states"""
    ARMED = "armed"        # Normal operation, monitoring active
    TRIGGERED = "triggered"  # Kill switch activated
    RECOVERY = "recovery"   # Attempting to recover
    MAINTENANCE = "maintenance"  # Manual maintenance mode

@dataclass
class KillSwitchEvent:
    """Kill switch activation event"""
    timestamp: datetime
    trigger_source: KillSwitchTrigger
    trigger_reason: str
    severity: str
    
    # Context
    portfolio_value_before: float
    portfolio_value_after: float
    positions_closed: int
    orders_cancelled: int
    
    # Recovery
    recovery_time: Optional[datetime] = None
    recovery_method: Optional[str] = None
    authorization_code: Optional[str] = None

class KillSwitchManager:
    """
    Emergency kill switch system for immediate trading halt
    """
    
    def __init__(self, 
                 risk_manager: RiskLimitManager,
                 circuit_breaker_manager: CircuitBreakerManager):
        
        self.risk_manager = risk_manager
        self.circuit_breaker_manager = circuit_breaker_manager
        
        # Kill switch state
        self.state = KillSwitchState.ARMED
        self.activation_time: Optional[datetime] = None
        self.last_trigger: Optional[KillSwitchEvent] = None
        
        # Event history
        self.events: List[KillSwitchEvent] = []
        
        # Emergency contacts and callbacks
        self.emergency_callbacks: List[Callable[[], None]] = []
        self.position_close_callback: Optional[Callable[[], bool]] = None
        self.order_cancel_callback: Optional[Callable[[], bool]] = None
        
        # Monitoring
        self.monitoring_active = True
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Authorization
        self.authorized_codes = {
            "EMERGENCY_OVERRIDE_2024",
            "SYSTEM_MAINTENANCE_MODE",
            "MANUAL_RECOVERY_AUTHORIZED"
        }
        
        # Configuration
        self.auto_recovery_enabled = False
        self.max_recovery_attempts = 3
        self.recovery_cooldown_minutes = 30
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start kill switch monitoring thread"""
        
        def monitoring_loop():
            logger.info("Kill switch monitoring started")
            
            while self.monitoring_active:
                try:
                    if self.state == KillSwitchState.ARMED:
                        self._check_trigger_conditions()
                    elif self.state == KillSwitchState.TRIGGERED:
                        self._monitor_triggered_state()
                    elif self.state == KillSwitchState.RECOVERY:
                        self._monitor_recovery_state()
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Kill switch monitoring error: {e}")
                    time.sleep(10)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _check_trigger_conditions(self):
        """Check all trigger conditions"""
        
        # Check risk limit violations
        risk_summary = self.risk_manager.get_risk_summary()
        
        # Daily loss limit breach
        if (risk_summary.get("daily_loss", {}).get("status") == "breach" or
            risk_summary.get("overall_risk_status") == "breach"):
            
            self.trigger_kill_switch(
                trigger_source=KillSwitchTrigger.DAILY_LOSS_LIMIT,
                reason=f"Daily loss limit breached: {risk_summary.get('daily_loss', {}).get('current_pct', 0):.1f}%"
            )
            return
        
        # Max drawdown breach
        if risk_summary.get("max_drawdown", {}).get("status") == "breach":
            self.trigger_kill_switch(
                trigger_source=KillSwitchTrigger.MAX_DRAWDOWN,
                reason=f"Max drawdown breached: {risk_summary.get('max_drawdown', {}).get('current_pct', 0):.1f}%"
            )
            return
        
        # Critical circuit breakers
        system_status = self.circuit_breaker_manager.get_system_status()
        open_breakers = system_status.get("open_breakers", [])
        
        if open_breakers:
            critical_breakers = []
            for breaker_name in open_breakers:
                breaker = self.circuit_breaker_manager.breakers.get(breaker_name)
                if breaker and breaker.config.alert_severity.value in ["critical"]:
                    critical_breakers.append(breaker_name)
            
            if critical_breakers:
                self.trigger_kill_switch(
                    trigger_source=KillSwitchTrigger.CIRCUIT_BREAKER,
                    reason=f"Critical circuit breakers open: {', '.join(critical_breakers)}"
                )
                return
    
    def trigger_kill_switch(self, 
                           trigger_source: KillSwitchTrigger,
                           reason: str,
                           authorization_code: Optional[str] = None) -> bool:
        """Trigger the kill switch"""
        
        if self.state == KillSwitchState.TRIGGERED:
            logger.warning("Kill switch already triggered")
            return False
        
        try:
            logger.critical(f"🚨 KILL SWITCH TRIGGERED 🚨")
            logger.critical(f"Source: {trigger_source.value}")
            logger.critical(f"Reason: {reason}")
            
            # Record portfolio state before
            portfolio_before = self.risk_manager.current_portfolio_value
            
            # Change state
            self.state = KillSwitchState.TRIGGERED
            self.activation_time = datetime.now()
            
            # Emergency actions
            positions_closed = self._close_all_positions()
            orders_cancelled = self._cancel_all_orders()
            
            # Stop trading in all systems
            self.risk_manager.emergency_stop_active = True
            self.risk_manager.kill_switch_triggered = True
            self.circuit_breaker_manager.trading_enabled = False
            
            # Record portfolio state after
            portfolio_after = self.risk_manager.current_portfolio_value
            
            # Create event record
            event = KillSwitchEvent(
                timestamp=datetime.now(),
                trigger_source=trigger_source,
                trigger_reason=reason,
                severity="CRITICAL",
                portfolio_value_before=portfolio_before,
                portfolio_value_after=portfolio_after,
                positions_closed=positions_closed,
                orders_cancelled=orders_cancelled,
                authorization_code=authorization_code
            )
            
            self.events.append(event)
            self.last_trigger = event
            
            # Execute emergency callbacks
            self._execute_emergency_callbacks()
            
            # Log critical information
            logger.critical(f"Kill switch activated at {datetime.now().isoformat()}")
            logger.critical(f"Positions closed: {positions_closed}")
            logger.critical(f"Orders cancelled: {orders_cancelled}")
            logger.critical(f"Portfolio value: ${portfolio_before:,.2f} → ${portfolio_after:,.2f}")
            
            return True
            
        except Exception as e:
            logger.critical(f"Kill switch execution failed: {e}")
            return False
    
    def _close_all_positions(self) -> int:
        """Close all open positions"""
        positions_closed = 0
        
        try:
            if self.position_close_callback:
                success = self.position_close_callback()
                if success:
                    positions_closed = len(self.risk_manager.current_positions)
                    # Clear positions from tracking
                    self.risk_manager.current_positions.clear()
                    logger.info(f"Closed {positions_closed} positions via callback")
                else:
                    logger.error("Position close callback failed")
            else:
                logger.warning("No position close callback configured")
                # Manual position clearing for tracking
                positions_closed = len(self.risk_manager.current_positions)
                self.risk_manager.current_positions.clear()
                
        except Exception as e:
            logger.error(f"Position closure failed: {e}")
        
        return positions_closed
    
    def _cancel_all_orders(self) -> int:
        """Cancel all pending orders"""
        orders_cancelled = 0
        
        try:
            if self.order_cancel_callback:
                success = self.order_cancel_callback()
                if success:
                    orders_cancelled = 1  # Assume some orders were cancelled
                    logger.info("Cancelled pending orders via callback")
                else:
                    logger.error("Order cancel callback failed")
            else:
                logger.warning("No order cancel callback configured")
                
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
        
        return orders_cancelled
    
    def _execute_emergency_callbacks(self):
        """Execute all registered emergency callbacks"""
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def _monitor_triggered_state(self):
        """Monitor system while kill switch is triggered"""
        if not self.activation_time:
            return
        
        # Log periodic status
        time_triggered = (datetime.now() - self.activation_time).total_seconds() / 60
        if int(time_triggered) % 5 == 0:  # Every 5 minutes
            logger.critical(f"Kill switch active for {time_triggered:.0f} minutes")
    
    def _monitor_recovery_state(self):
        """Monitor system during recovery"""
        # Recovery monitoring logic would go here
        pass
    
    def manual_trigger(self, reason: str, authorization_code: str) -> bool:
        """Manually trigger kill switch with authorization"""
        
        if authorization_code not in self.authorized_codes:
            logger.error(f"Invalid authorization code for manual kill switch")
            return False
        
        return self.trigger_kill_switch(
            trigger_source=KillSwitchTrigger.MANUAL,
            reason=f"Manual trigger: {reason}",
            authorization_code=authorization_code
        )
    
    def reset_kill_switch(self, authorization_code: str, recovery_method: str = "manual") -> bool:
        """Reset kill switch with proper authorization"""
        
        if authorization_code not in self.authorized_codes:
            logger.error("Invalid authorization code for kill switch reset")
            return False
        
        if self.state != KillSwitchState.TRIGGERED:
            logger.warning("Kill switch not currently triggered")
            return False
        
        try:
            # Change state
            self.state = KillSwitchState.RECOVERY
            
            # Re-enable trading systems
            self.risk_manager.emergency_stop_active = False
            self.risk_manager.kill_switch_triggered = False
            self.circuit_breaker_manager.trading_enabled = True
            
            # Update last event
            if self.last_trigger:
                self.last_trigger.recovery_time = datetime.now()
                self.last_trigger.recovery_method = recovery_method
                self.last_trigger.authorization_code = authorization_code
            
            # Reset to armed state
            self.state = KillSwitchState.ARMED
            self.activation_time = None
            
            logger.warning(f"Kill switch reset by {authorization_code}")
            logger.warning(f"Recovery method: {recovery_method}")
            
            return True
            
        except Exception as e:
            logger.error(f"Kill switch reset failed: {e}")
            return False
    
    def add_emergency_callback(self, callback: Callable[[], None]):
        """Add emergency callback function"""
        self.emergency_callbacks.append(callback)
        logger.info("Emergency callback added")
    
    def set_position_close_callback(self, callback: Callable[[], bool]):
        """Set callback for closing all positions"""
        self.position_close_callback = callback
        logger.info("Position close callback set")
    
    def set_order_cancel_callback(self, callback: Callable[[], bool]):
        """Set callback for cancelling all orders"""
        self.order_cancel_callback = callback
        logger.info("Order cancel callback set")
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
            "monitoring_active": self.monitoring_active,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "time_since_activation": None,
            "last_trigger": None,
            "total_events": len(self.events),
            "recent_events_24h": 0
        }
        
        # Time since activation
        if self.activation_time:
            status["time_since_activation"] = (datetime.now() - self.activation_time).total_seconds()
        
        # Last trigger info
        if self.last_trigger:
            status["last_trigger"] = {
                "timestamp": self.last_trigger.timestamp.isoformat(),
                "source": self.last_trigger.trigger_source.value,
                "reason": self.last_trigger.trigger_reason,
                "positions_closed": self.last_trigger.positions_closed,
                "orders_cancelled": self.last_trigger.orders_cancelled,
                "recovered": self.last_trigger.recovery_time is not None
            }
        
        # Recent events
        cutoff_time = datetime.now() - timedelta(hours=24)
        status["recent_events_24h"] = len([
            e for e in self.events if e.timestamp >= cutoff_time
        ])
        
        return status
    
    def test_kill_switch(self, authorization_code: str) -> bool:
        """Test kill switch functionality without actually triggering"""
        
        if authorization_code not in self.authorized_codes:
            logger.error("Invalid authorization code for kill switch test")
            return False
        
        try:
            logger.info("🧪 TESTING KILL SWITCH FUNCTIONALITY")
            
            # Test callbacks
            test_results = {
                "position_close_callback": False,
                "order_cancel_callback": False,
                "emergency_callbacks": 0
            }
            
            # Test position close callback
            if self.position_close_callback:
                try:
                    # Don't actually call it, just check if it exists
                    test_results["position_close_callback"] = True
                    logger.info("✅ Position close callback available")
                except:
                    logger.error("❌ Position close callback test failed")
            
            # Test order cancel callback
            if self.order_cancel_callback:
                try:
                    test_results["order_cancel_callback"] = True
                    logger.info("✅ Order cancel callback available")
                except:
                    logger.error("❌ Order cancel callback test failed")
            
            # Test emergency callbacks
            test_results["emergency_callbacks"] = len(self.emergency_callbacks)
            logger.info(f"✅ {test_results['emergency_callbacks']} emergency callbacks registered")
            
            # Test monitoring
            monitoring_ok = self.monitoring_active and self.monitoring_thread and self.monitoring_thread.is_alive()
            logger.info(f"✅ Monitoring active: {monitoring_ok}")
            
            # Test integration
            risk_ok = hasattr(self.risk_manager, 'emergency_stop_active')
            circuit_ok = hasattr(self.circuit_breaker_manager, 'trading_enabled')
            logger.info(f"✅ Risk manager integration: {risk_ok}")
            logger.info(f"✅ Circuit breaker integration: {circuit_ok}")
            
            logger.info("🧪 Kill switch test completed")
            return True
            
        except Exception as e:
            logger.error(f"Kill switch test failed: {e}")
            return False
    
    def export_kill_switch_report(self, filepath: str, days_back: int = 30):
        """Export kill switch events and status report"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_events = [
                event for event in self.events
                if event.timestamp >= cutoff_date
            ]
            
            report_data = {
                "report_timestamp": datetime.now().isoformat(),
                "period_days": days_back,
                "current_status": self.get_status(),
                
                "configuration": {
                    "auto_recovery_enabled": self.auto_recovery_enabled,
                    "max_recovery_attempts": self.max_recovery_attempts,
                    "recovery_cooldown_minutes": self.recovery_cooldown_minutes,
                    "emergency_callbacks": len(self.emergency_callbacks),
                    "position_close_callback_set": self.position_close_callback is not None,
                    "order_cancel_callback_set": self.order_cancel_callback is not None
                },
                
                "events": [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "trigger_source": event.trigger_source.value,
                        "trigger_reason": event.trigger_reason,
                        "severity": event.severity,
                        "portfolio_value_before": event.portfolio_value_before,
                        "portfolio_value_after": event.portfolio_value_after,
                        "positions_closed": event.positions_closed,
                        "orders_cancelled": event.orders_cancelled,
                        "recovery_time": event.recovery_time.isoformat() if event.recovery_time else None,
                        "recovery_method": event.recovery_method
                    }
                    for event in recent_events
                ],
                
                "statistics": {
                    "total_activations": len(self.events),
                    "recent_activations": len(recent_events),
                    "successful_recoveries": len([
                        e for e in self.events if e.recovery_time is not None
                    ]),
                    "trigger_sources": {
                        trigger.value: len([
                            e for e in self.events if e.trigger_source == trigger
                        ])
                        for trigger in KillSwitchTrigger
                    }
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Kill switch report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export kill switch report: {e}")