#!/usr/bin/env python3
"""
FASE D - Advanced AlertManager Implementation
HighOrderErrorRate, DrawdownTooHigh, NoSignals(30m) alerts with Prometheus integration
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertState(Enum):
    """Alert state enumeration"""
    RESOLVED = "resolved"
    FIRING = "firing"
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AlertCondition:
    """Alert condition definition"""
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    duration_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True


@dataclass  
class Alert:
    """Active alert instance"""
    condition: AlertCondition
    state: AlertState
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    current_value: Optional[float] = None
    message: str = ""
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class FaseDAlertManager:
    """
    FASE D Alert Manager
    Monitors HighOrderErrorRate, DrawdownTooHigh, NoSignals(30m) alerts
    """
    
    def __init__(self, metrics_instance=None):
        from .metrics import get_metrics
        self.metrics = metrics_instance or get_metrics()
        
        # Alert conditions for FASE D requirements
        self.alert_conditions = self._initialize_fase_d_conditions()
        
        # Active alerts tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert state persistence
        self.alert_state_file = Path("logs/alert_state.json")
        self.alert_state_file.parent.mkdir(exist_ok=True)
        
        # Load persistent state
        self._load_alert_state()
        
        logger.info("FASE D AlertManager initialized with 3 core alert conditions")
    
    def _initialize_fase_d_conditions(self) -> Dict[str, AlertCondition]:
        """Initialize FASE D alert conditions"""
        conditions = {}
        
        # 1. HighOrderErrorRate - Critical when order error rate > 5%
        conditions['high_order_error_rate'] = AlertCondition(
            name="HighOrderErrorRate",
            description="Order error rate exceeds 5% threshold",
            metric_name="alert_high_order_error_rate",
            threshold=1.0,  # Alert gauge set to 1 when firing
            comparison=">=",
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL
        )
        
        # 2. DrawdownTooHigh - High when portfolio drawdown > 10%
        conditions['drawdown_too_high'] = AlertCondition(
            name="DrawdownTooHigh", 
            description="Portfolio drawdown exceeds 10% threshold",
            metric_name="alert_drawdown_too_high",
            threshold=1.0,  # Alert gauge set to 1 when firing
            comparison=">=",
            duration_seconds=60,
            severity=AlertSeverity.HIGH
        )
        
        # 3. NoSignals - Medium when no signals for 30 minutes
        conditions['no_signals_timeout'] = AlertCondition(
            name="NoSignals30m",
            description="No trading signals received for 30 minutes",
            metric_name="alert_no_signals_timeout", 
            threshold=1.0,  # Alert gauge set to 1 when firing
            comparison=">=",
            duration_seconds=120,  # 2 minute confirmation
            severity=AlertSeverity.MEDIUM
        )
        
        return conditions
    
    def evaluate_alerts(self) -> Dict[str, Any]:
        """Evaluate all alert conditions and update states"""
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'alerts_evaluated': len(self.alert_conditions),
            'alerts_firing': 0,
            'alerts_resolved': 0,
            'new_alerts': 0
        }
        
        for condition_name, condition in self.alert_conditions.items():
            if not condition.enabled:
                continue
                
            # Get current metric value
            current_value = self._get_metric_value(condition.metric_name)
            
            # Evaluate condition
            is_breached = self._evaluate_condition(condition, current_value)
            
            # Update alert state
            if is_breached:
                if condition_name not in self.active_alerts:
                    # New alert
                    alert = Alert(
                        condition=condition,
                        state=AlertState.FIRING,
                        triggered_at=datetime.now(),
                        current_value=current_value,
                        message=f"{condition.description} (value: {current_value})"
                    )
                    self.active_alerts[condition_name] = alert
                    self.alert_history.append(alert)
                    evaluation_results['new_alerts'] += 1
                    evaluation_results['alerts_firing'] += 1
                    
                    logger.warning(f"🚨 ALERT FIRING: {condition.name} - {alert.message}")
                    
                else:
                    # Existing alert still firing
                    self.active_alerts[condition_name].current_value = current_value
                    evaluation_results['alerts_firing'] += 1
                    
            else:
                if condition_name in self.active_alerts:
                    # Resolve existing alert
                    alert = self.active_alerts[condition_name]
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = datetime.now()
                    alert.current_value = current_value
                    
                    del self.active_alerts[condition_name]
                    evaluation_results['alerts_resolved'] += 1
                    
                    logger.info(f"✅ ALERT RESOLVED: {condition.name}")
        
        # Save alert state
        self._save_alert_state()
        
        return evaluation_results
    
    def _evaluate_condition(self, condition: AlertCondition, current_value: float) -> bool:
        """Evaluate if condition is breached"""
        if current_value is None:
            return False
            
        threshold = condition.threshold
        comparison = condition.comparison
        
        if comparison == ">":
            return current_value > threshold
        elif comparison == ">=":
            return current_value >= threshold
        elif comparison == "<":
            return current_value < threshold
        elif comparison == "<=":
            return current_value <= threshold
        elif comparison == "==":
            return current_value == threshold
        elif comparison == "!=":
            return current_value != threshold
        else:
            logger.error(f"Unknown comparison operator: {comparison}")
            return False
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of metric from metrics instance"""
        try:
            if hasattr(self.metrics, metric_name.replace('alert_', '')):
                metric = getattr(self.metrics, metric_name.replace('alert_', ''))
                return self.metrics._get_gauge_value(metric)
            else:
                # Try with alert_ prefix
                if hasattr(self.metrics, metric_name):
                    metric = getattr(self.metrics, metric_name)
                    return self.metrics._get_gauge_value(metric)
                else:
                    logger.warning(f"Metric not found: {metric_name}")
                    return None
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get comprehensive alert status"""
        return {
            'total_conditions': len(self.alert_conditions),
            'active_alerts': len(self.active_alerts),
            'alert_history_count': len(self.alert_history),
            'conditions': {
                name: {
                    'enabled': condition.enabled,
                    'severity': condition.severity.value,
                    'threshold': condition.threshold,
                    'description': condition.description
                }
                for name, condition in self.alert_conditions.items()
            },
            'active_alerts_detail': {
                name: {
                    'state': alert.state.value,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'current_value': alert.current_value,
                    'message': alert.message,
                    'severity': alert.condition.severity.value
                }
                for name, alert in self.active_alerts.items()
            }
        }
    
    def acknowledge_alert(self, alert_name: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        if alert_name in self.active_alerts:
            self.active_alerts[alert_name].state = AlertState.ACKNOWLEDGED
            self.active_alerts[alert_name].labels['acknowledged_by'] = acknowledged_by
            self.active_alerts[alert_name].labels['acknowledged_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert acknowledged: {alert_name} by {acknowledged_by}")
            self._save_alert_state()
            return True
        return False
    
    def _save_alert_state(self):
        """Save alert state to persistent storage"""
        try:
            state_data = {
                'active_alerts': {
                    name: {
                        'condition_name': alert.condition.name,
                        'state': alert.state.value,
                        'triggered_at': alert.triggered_at.isoformat(),
                        'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                        'current_value': alert.current_value,
                        'message': alert.message,
                        'labels': alert.labels
                    }
                    for name, alert in self.active_alerts.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.alert_state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save alert state: {e}")
    
    def _load_alert_state(self):
        """Load alert state from persistent storage"""
        try:
            if self.alert_state_file.exists():
                with open(self.alert_state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Reconstruct active alerts (simplified - would need full condition reconstruction)
                logger.info(f"Loaded {len(state_data.get('active_alerts', {}))} persistent alerts")
                
        except Exception as e:
            logger.error(f"Failed to load alert state: {e}")
    
    def export_prometheus_rules(self) -> str:
        """Export alert rules in Prometheus AlertManager format"""
        rules = {
            'groups': [
                {
                    'name': 'cryptosmarttrader_fase_d_alerts',
                    'rules': []
                }
            ]
        }
        
        for condition_name, condition in self.alert_conditions.items():
            rule = {
                'alert': condition.name,
                'expr': f'{condition.metric_name} {condition.comparison} {condition.threshold}',
                'for': f'{condition.duration_seconds}s',
                'labels': {
                    'severity': condition.severity.value,
                    'component': 'cryptosmarttrader'
                },
                'annotations': {
                    'summary': condition.description,
                    'description': f'{condition.description} (current value: {{{{ $value }}}})'
                }
            }
            rules['groups'][0]['rules'].append(rule)
        
        import yaml
        return yaml.dump(rules, default_flow_style=False)


# Global alert manager instance
_alert_manager_instance = None
_alert_manager_lock = threading.Lock()


def get_alert_manager() -> FaseDAlertManager:
    """Get global alert manager instance"""
    global _alert_manager_instance
    with _alert_manager_lock:
        if _alert_manager_instance is None:
            _alert_manager_instance = FaseDAlertManager()
        return _alert_manager_instance


def reset_alert_manager():
    """Reset alert manager instance (for testing)"""
    global _alert_manager_instance
    _alert_manager_instance = None