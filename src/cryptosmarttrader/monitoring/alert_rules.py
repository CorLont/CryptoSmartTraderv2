"""Alert rules and thresholds for comprehensive system monitoring."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path

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
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Individual alert rule definition."""
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    duration_minutes: int = 5  # How long condition must persist
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Alert:
    """Active alert instance."""
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_value: float
    threshold: float
    started_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize alert manager."""
        self.logger = get_logger("alert_manager")
        
        # Alert rules and active alerts
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Persistence
        self.data_path = Path("data/alerts")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load default rules
        self._load_default_rules()
        
        # Load custom rules if provided
        if config_path:
            self._load_custom_rules(config_path)
        
        self.logger.info(f"Alert manager initialized with {len(self.rules)} rules")
    
    def _load_default_rules(self) -> None:
        """Load default alert rules."""
        default_rules = [
            # Trading alerts
            AlertRule(
                name="HighOrderErrorRate",
                description="Order error rate is too high",
                severity=AlertSeverity.CRITICAL,
                metric_name="cst_order_error_rate",
                operator=">",
                threshold=0.1,  # 10% error rate
                duration_minutes=5,
                annotations={
                    "summary": "High order error rate detected",
                    "description": "Order error rate has exceeded 10% for 5 minutes"
                }
            ),
            
            AlertRule(
                name="DrawdownTooHigh",
                description="Portfolio drawdown exceeds safe limits",
                severity=AlertSeverity.EMERGENCY,
                metric_name="cst_max_drawdown_percent",
                operator=">",
                threshold=10.0,  # 10% drawdown
                duration_minutes=1,
                annotations={
                    "summary": "Excessive portfolio drawdown",
                    "description": "Maximum drawdown has exceeded 10%"
                }
            ),
            
            AlertRule(
                name="DailyLossLimit",
                description="Daily loss limit reached",
                severity=AlertSeverity.CRITICAL,
                metric_name="cst_daily_pnl_percent",
                operator="<",
                threshold=-5.0,  # -5% daily loss
                duration_minutes=1,
                annotations={
                    "summary": "Daily loss limit reached",
                    "description": "Daily PnL has fallen below -5%"
                }
            ),
            
            AlertRule(
                name="NoSignals",
                description="No trading signals generated recently",
                severity=AlertSeverity.WARNING,
                metric_name="cst_last_signal_age_minutes",
                operator=">",
                threshold=60.0,  # 1 hour without signals
                duration_minutes=10,
                annotations={
                    "summary": "No recent trading signals",
                    "description": "No trading signals generated in the last hour"
                }
            ),
            
            AlertRule(
                name="HighSlippage",
                description="Trading slippage is excessive",
                severity=AlertSeverity.WARNING,
                metric_name="cst_average_slippage_percent",
                operator=">",
                threshold=0.5,  # 0.5% average slippage
                duration_minutes=15,
                annotations={
                    "summary": "High trading slippage detected",
                    "description": "Average slippage has exceeded 0.5% for 15 minutes"
                }
            ),
            
            # Data quality alerts
            AlertRule(
                name="DataGapDetected",
                description="Data feed gap detected",
                severity=AlertSeverity.CRITICAL,
                metric_name="cst_data_gap_minutes",
                operator=">",
                threshold=5.0,  # 5 minutes data gap
                duration_minutes=1,
                annotations={
                    "summary": "Data feed interruption",
                    "description": "Data feed has been interrupted for more than 5 minutes"
                }
            ),
            
            AlertRule(
                name="LowDataQuality",
                description="Data quality score is low",
                severity=AlertSeverity.WARNING,
                metric_name="cst_data_quality_score",
                operator="<",
                threshold=0.7,  # 70% quality threshold
                duration_minutes=10,
                annotations={
                    "summary": "Low data quality detected",
                    "description": "Data quality score has fallen below 70%"
                }
            ),
            
            # System health alerts
            AlertRule(
                name="AgentDown",
                description="Critical agent is not running",
                severity=AlertSeverity.CRITICAL,
                metric_name="cst_agent_status",
                operator="==",
                threshold=0.0,  # Agent stopped
                duration_minutes=2,
                annotations={
                    "summary": "Critical agent stopped",
                    "description": "A critical trading agent has stopped running"
                }
            ),
            
            AlertRule(
                name="HighMemoryUsage",
                description="Memory usage is too high",
                severity=AlertSeverity.WARNING,
                metric_name="cst_memory_usage_percent",
                operator=">",
                threshold=85.0,  # 85% memory usage
                duration_minutes=10,
                annotations={
                    "summary": "High memory usage",
                    "description": "System memory usage has exceeded 85%"
                }
            ),
            
            AlertRule(
                name="HighCPUUsage",
                description="CPU usage is too high",
                severity=AlertSeverity.WARNING,
                metric_name="cst_cpu_usage_percent",
                operator=">",
                threshold=90.0,  # 90% CPU usage
                duration_minutes=5,
                annotations={
                    "summary": "High CPU usage",
                    "description": "System CPU usage has exceeded 90%"
                }
            ),
            
            # API and connectivity alerts
            AlertRule(
                name="APIErrorRate",
                description="API error rate is too high",
                severity=AlertSeverity.WARNING,
                metric_name="cst_api_error_rate_percent",
                operator=">",
                threshold=5.0,  # 5% API error rate
                duration_minutes=5,
                annotations={
                    "summary": "High API error rate",
                    "description": "API error rate has exceeded 5% for 5 minutes"
                }
            ),
            
            AlertRule(
                name="SlowAPIResponse",
                description="API response time is too slow",
                severity=AlertSeverity.WARNING,
                metric_name="cst_api_response_time_seconds",
                operator=">",
                threshold=10.0,  # 10 seconds
                duration_minutes=5,
                annotations={
                    "summary": "Slow API responses",
                    "description": "API response time has exceeded 10 seconds"
                }
            ),
            
            # Risk management alerts
            AlertRule(
                name="KillSwitchActivated",
                description="Emergency kill switch has been activated",
                severity=AlertSeverity.EMERGENCY,
                metric_name="cst_kill_switch_active",
                operator="==",
                threshold=1.0,  # Kill switch active
                duration_minutes=0,  # Immediate alert
                annotations={
                    "summary": "Emergency kill switch activated",
                    "description": "The emergency kill switch has been triggered"
                }
            ),
            
            AlertRule(
                name="RiskLevelEscalation",
                description="Risk level has escalated",
                severity=AlertSeverity.WARNING,
                metric_name="cst_risk_level_numeric",
                operator=">=",
                threshold=3.0,  # Defensive level or higher
                duration_minutes=1,
                annotations={
                    "summary": "Risk level escalation",
                    "description": "Trading risk level has escalated to defensive or higher"
                }
            ),
            
            # Performance alerts
            AlertRule(
                name="LowPredictionAccuracy",
                description="Prediction accuracy has dropped",
                severity=AlertSeverity.WARNING,
                metric_name="cst_prediction_accuracy_percent",
                operator="<",
                threshold=60.0,  # 60% accuracy threshold
                duration_minutes=30,
                annotations={
                    "summary": "Low prediction accuracy",
                    "description": "Model prediction accuracy has fallen below 60%"
                }
            ),
            
            AlertRule(
                name="ExcessiveRetries",
                description="Too many order retries",
                severity=AlertSeverity.WARNING,
                metric_name="cst_order_retry_rate",
                operator=">",
                threshold=0.2,  # 20% retry rate
                duration_minutes=15,
                annotations={
                    "summary": "Excessive order retries",
                    "description": "Order retry rate has exceeded 20%"
                }
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    def _load_custom_rules(self, config_path: str) -> None:
        """Load custom alert rules from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            custom_rules = config.get('alert_rules', [])
            for rule_data in custom_rules:
                rule = AlertRule(**rule_data)
                self.rules[rule.name] = rule
                self.logger.info(f"Loaded custom alert rule: {rule.name}")
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to load custom alert rules: {e}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)
        self.logger.info("Alert callback registered")
    
    def check_metric(self, metric_name: str, value: float, 
                    labels: Optional[Dict[str, str]] = None) -> None:
        """Check metric value against alert rules."""
        labels = labels or {}
        
        for rule in self.rules.values():
            if not rule.enabled or rule.metric_name != metric_name:
                continue
            
            if self._evaluate_condition(value, rule.operator, rule.threshold):
                self._trigger_alert(rule, value, labels)
            else:
                self._resolve_alert(rule.name)
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            self.logger.warning(f"Unknown operator: {operator}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, value: float, 
                      labels: Dict[str, str]) -> None:
        """Trigger alert if conditions are met."""
        alert_key = f"{rule.name}_{hash(frozenset(labels.items()))}"
        
        now = datetime.now()
        
        if alert_key in self.active_alerts:
            # Alert already active, check if duration threshold met
            alert = self.active_alerts[alert_key]
            if alert.status == AlertStatus.ACTIVE:
                duration = (now - alert.started_at).total_seconds() / 60
                if duration < rule.duration_minutes:
                    return  # Not enough time passed
        else:
            # New alert
            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=self._format_alert_message(rule, value),
                metric_value=value,
                threshold=rule.threshold,
                started_at=now,
                labels={**rule.labels, **labels},
                annotations=rule.annotations.copy()
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            self.logger.warning(f"Alert triggered: {rule.name}",
                              severity=rule.severity.value,
                              metric_value=value,
                              threshold=rule.threshold)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def _resolve_alert(self, rule_name: str) -> None:
        """Resolve active alert."""
        alerts_to_resolve = [
            key for key, alert in self.active_alerts.items()
            if alert.rule_name == rule_name and alert.status == AlertStatus.ACTIVE
        ]
        
        for alert_key in alerts_to_resolve:
            alert = self.active_alerts[alert_key]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            self.logger.info(f"Alert resolved: {rule_name}")
            
            # Remove from active alerts
            del self.active_alerts[alert_key]
    
    def _format_alert_message(self, rule: AlertRule, value: float) -> str:
        """Format alert message."""
        return (f"{rule.description}. "
                f"Current value: {value:.2f}, "
                f"Threshold: {rule.threshold:.2f}")
    
    def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            
            self.logger.info(f"Alert acknowledged: {alert.rule_name}")
            return True
        
        return False
    
    def silence_alert(self, rule_name: str, duration_minutes: int = 60) -> None:
        """Silence alerts for a specific rule."""
        # Implementation would add silencing logic
        self.logger.info(f"Alert rule silenced: {rule_name} for {duration_minutes} minutes")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1 for alert in active_alerts if alert.severity == severity
            )
        
        return {
            'total_active': len(active_alerts),
            'by_severity': severity_counts,
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for rule in self.rules.values() if rule.enabled),
            'total_historical': len(self.alert_history)
        }
    
    def export_rules(self, file_path: str) -> None:
        """Export alert rules to JSON file."""
        rules_data = []
        for rule in self.rules.values():
            rules_data.append({
                'name': rule.name,
                'description': rule.description,
                'severity': rule.severity.value,
                'metric_name': rule.metric_name,
                'operator': rule.operator,
                'threshold': rule.threshold,
                'duration_minutes': rule.duration_minutes,
                'labels': rule.labels,
                'annotations': rule.annotations,
                'enabled': rule.enabled
            })
        
        with open(file_path, 'w') as f:
            json.dump({'alert_rules': rules_data}, f, indent=2)
        
        self.logger.info(f"Alert rules exported to {file_path}")


def create_alert_manager(config_path: Optional[str] = None) -> AlertManager:
    """Factory function to create AlertManager instance."""
    return AlertManager(config_path=config_path)