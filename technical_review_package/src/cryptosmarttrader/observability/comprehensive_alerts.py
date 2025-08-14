"""
Comprehensive Alert System for Fase C - Guardrails & Observability
HighErrorRate, DrawdownTooHigh, NoSignals 30m alerts with enforcement
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..core.structured_logger import get_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Alert rule definition."""
    
    name: str
    description: str
    condition: str  # Condition expression
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 15
    evaluation_window_minutes: int = 5
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance."""
    
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: float
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveAlertManager:
    """
    ENTERPRISE ALERT SYSTEM - MANDATORY ENFORCEMENT
    
    Implements required alerts for Fase C:
    - HighOrderErrorRate
    - DrawdownTooHigh  
    - NoSignals (30 minutes)
    """
    
    def __init__(self):
        """Initialize with mandatory Fase C alert rules."""
        self.logger = get_logger("alert_manager")
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.metrics_history: Dict[str, List[tuple]] = {}  # metric_name -> [(timestamp, value)]
        self.last_evaluation = {}  # rule_name -> timestamp
        
        # Initialize mandatory Fase C rules
        self._setup_fase_c_rules()
        
        self.logger.info("Comprehensive Alert Manager initialized with Fase C enforcement")
    
    def _setup_fase_c_rules(self):
        """Setup mandatory alert rules for Fase C compliance."""
        
        # 1. HighOrderErrorRate - Critical for execution quality
        self.add_rule(AlertRule(
            name="HighOrderErrorRate",
            description="Order error rate exceeds acceptable threshold",
            condition="order_error_rate > threshold",
            threshold=0.05,  # 5% error rate
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=10,
            evaluation_window_minutes=5,
            tags={"category": "execution", "fase": "C"}
        ))
        
        # 2. DrawdownTooHigh - Risk management enforcement  
        self.add_rule(AlertRule(
            name="DrawdownTooHigh",
            description="Portfolio drawdown exceeds risk limits",
            condition="max_drawdown_percent > threshold",
            threshold=10.0,  # 10% maximum drawdown
            severity=AlertSeverity.EMERGENCY,
            cooldown_minutes=5,
            evaluation_window_minutes=1,
            tags={"category": "risk", "fase": "C"}
        ))
        
        # 3. NoSignals - Signal flow monitoring
        self.add_rule(AlertRule(
            name="NoSignals",
            description="No trading signals received in 30 minutes",
            condition="minutes_since_last_signal > threshold",
            threshold=30.0,  # 30 minutes
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15,
            evaluation_window_minutes=5,
            tags={"category": "signals", "fase": "C"}
        ))
        
        # 4. HighSlippage - P95 slippage budget enforcement
        self.add_rule(AlertRule(
            name="HighSlippage",
            description="P95 slippage exceeds budget allocation",
            condition="p95_slippage_percent > threshold", 
            threshold=0.3,  # 0.3% slippage budget
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=10,
            evaluation_window_minutes=5,
            tags={"category": "execution", "fase": "C"}
        ))
        
        # 5. ExchangeConnectivityLost - Data integrity enforcement
        self.add_rule(AlertRule(
            name="ExchangeConnectivityLost",
            description="Exchange API connectivity issues detected",
            condition="api_success_rate < threshold",
            threshold=0.90,  # 90% success rate minimum
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
            evaluation_window_minutes=2,
            tags={"category": "connectivity", "fase": "C"}
        ))
        
        # 6. LowLiquidity - Tradability gate enforcement
        self.add_rule(AlertRule(
            name="LowLiquidity",
            description="Market liquidity below execution thresholds",
            condition="avg_liquidity_score < threshold",
            threshold=0.6,  # 60% minimum liquidity score
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15,
            evaluation_window_minutes=10,
            tags={"category": "liquidity", "fase": "C"}
        ))
        
        # 7. HighResourceUsage - System monitoring
        self.add_rule(AlertRule(
            name="HighResourceUsage",
            description="System resource usage exceeds operational limits",
            condition="max_resource_usage > threshold",
            threshold=85.0,  # 85% resource usage
            severity=AlertSeverity.WARNING,
            cooldown_minutes=20,
            evaluation_window_minutes=5,
            tags={"category": "system", "fase": "C"}
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule to manager."""
        self.rules[rule.name] = rule
        self.logger.info(f"Alert rule added: {rule.name}")
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record metric value for alert evaluation."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append((timestamp, value))
        
        # Keep only recent history (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics_history[metric_name] = [
            (ts, val) for ts, val in self.metrics_history[metric_name] 
            if ts > cutoff
        ]
    
    def evaluate_rules(self, current_metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        new_alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.last_evaluation:
                last_eval = self.last_evaluation[rule_name]
                if (datetime.now() - last_eval).total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            try:
                alert = self._evaluate_rule(rule, current_metrics)
                if alert:
                    new_alerts.append(alert)
                    self.alerts[alert.id] = alert
                    self.logger.error(f"ALERT TRIGGERED: {alert.rule_name}", 
                                    value=alert.value, 
                                    threshold=alert.threshold,
                                    severity=alert.severity.value)
                
                self.last_evaluation[rule_name] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Rule evaluation failed: {rule_name}: {e}")
        
        return new_alerts
    
    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, float]) -> Optional[Alert]:
        """Evaluate single rule against current metrics."""
        
        # Get metric value based on rule condition
        metric_value = None
        
        if rule.name == "HighOrderErrorRate":
            metric_value = metrics.get("order_error_rate", 0.0)
        elif rule.name == "DrawdownTooHigh":
            metric_value = metrics.get("max_drawdown_percent", 0.0)
        elif rule.name == "NoSignals":
            last_signal_time = metrics.get("last_signal_timestamp", time.time())
            metric_value = (time.time() - last_signal_time) / 60.0  # minutes
        elif rule.name == "HighSlippage":
            metric_value = metrics.get("p95_slippage_percent", 0.0)
        elif rule.name == "ExchangeConnectivityLost":
            metric_value = metrics.get("api_success_rate", 1.0)
        elif rule.name == "LowLiquidity":
            metric_value = metrics.get("avg_liquidity_score", 1.0)
        elif rule.name == "HighResourceUsage":
            cpu = metrics.get("cpu_usage_percent", 0.0)
            memory = metrics.get("memory_usage_percent", 0.0)
            metric_value = max(cpu, memory)
        
        if metric_value is None:
            return None
        
        # Evaluate condition
        triggered = False
        if ">" in rule.condition:
            triggered = metric_value > rule.threshold
        elif "<" in rule.condition:
            triggered = metric_value < rule.threshold
        
        if triggered:
            # Check if alert already exists and is active
            existing_alert_id = f"{rule.name}_{int(time.time() // 3600)}"  # Hour-based deduplication
            if existing_alert_id in self.alerts and self.alerts[existing_alert_id].status == AlertStatus.ACTIVE:
                return None
            
            # Create new alert
            alert = Alert(
                id=existing_alert_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=f"{rule.description}: {metric_value:.3f} (threshold: {rule.threshold})",
                value=metric_value,
                threshold=rule.threshold,
                triggered_at=datetime.now(),
                metadata={"rule_tags": rule.tags}
            )
            
            return alert
        
        return None
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            if alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                self.logger.info(f"Alert acknowledged: {alert.rule_name}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            self.logger.info(f"Alert resolved: {alert.rule_name}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        active_alerts = self.get_active_alerts()
        
        summary = {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "emergency_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.EMERGENCY]),
            "rules_configured": len(self.rules),
            "rules_enabled": len([r for r in self.rules.values() if r.enabled]),
            "fase_c_compliance": {
                "HighOrderErrorRate": "HighOrderErrorRate" in self.rules,
                "DrawdownTooHigh": "DrawdownTooHigh" in self.rules,
                "NoSignals": "NoSignals" in self.rules,
                "mandatory_rules_active": all([
                    "HighOrderErrorRate" in self.rules,
                    "DrawdownTooHigh" in self.rules, 
                    "NoSignals" in self.rules
                ])
            }
        }
        
        return summary


def create_alert_manager() -> ComprehensiveAlertManager:
    """Factory function to create alert manager with Fase C compliance."""
    return ComprehensiveAlertManager()